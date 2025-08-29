import os
import time
import io
import re
import requests
from typing import Optional, Tuple, List
from dotenv import load_dotenv
import yt_dlp  # ✅ switched from pytube
from murf import MurfDub
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from types import SimpleNamespace

load_dotenv()

MURFDUB_API_KEY = os.getenv("MURFDUB_API_KEY")
murf_client = MurfDub(api_key=MURFDUB_API_KEY)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task="text-generation"
)
chat = ChatHuggingFace(llm=llm)

# Target locales per Murf docs
TARGET_LOCALES = [
    "en_US","en_UK","en_IN","en_SCOTT","en_AU",
    "fr_FR","de_DE","es_ES","es_MX","it_IT","pt_BR","pl_PL",
    "hi_IN","ko_KR","ta_IN","bn_IN","ja_JP","zh_CN","nl_NL","fi_FI",
    "ru_RU","tr_TR","uk_UA","da_DK","id_ID","ro_RO","nb_NO",
]

# ---------- YT download ----------

def clean_youtube_url(url: str) -> str:
    """Return base YouTube video URL (remove &t=, &si=, etc)."""
    if "&" in url:
        url = url.split("&")[0]
    return url

def download_youtube_highest_mp4(url: str, out_dir: str = "downloads") -> str:
    """
    Downloads the highest quality MP4 (video+audio merged) using yt-dlp.
    Returns the file path to the saved video.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        url = clean_youtube_url(url)
        output_template = os.path.join(out_dir, "%(title)s.%(ext)s")

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': output_template,
            'quiet': True,
            # 'cookiesfrombrowser': ('chrome'),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info).replace(".webm", ".mp4")
            return file_path

    except Exception as e:
        raise Exception(f"Failed to download Youtube video with yt-dlp: {e}")

# ---------- Murf Dub API wrappers ----------

def create_dub_job(file_path: str, target_locale: str, priority: str = "LOW") -> dict:
    if target_locale not in TARGET_LOCALES:
        raise ValueError(f"target_locale '{target_locale}' not in supported TARGET_LOCALES.")
    with open(file_path, "rb") as f:
        res = murf_client.dubbing.jobs.create(
            target_locales=[target_locale],
            file_name=os.path.basename(file_path),
            file=f,
            priority=priority
        )

    if isinstance(res, dict):
        return SimpleNamespace(
            id=res.get("id") or res.get("job_id"),
            raw=res
        )
    else:
        try:
            return SimpleNamespace(
                id=getattr(res, "id", None) or getattr(res, "job_id", None),
                raw=res.__dict__
            )
        except Exception as e:
            raise ValueError(f"Unexpected response format: {e}")

def poll_job_until_complete(job_id: str, poll_interval: float = 3.0, timeout_sec: int = 1800) -> dict:
    start = time.time()
    attempt = 0

    while True:
        status = murf_client.dubbing.jobs.get_status(job_id=job_id)

        if hasattr(status, "to_dict"):
            status_dict = status.to_dict()
        elif hasattr(status, "__dict__"):
            status_dict = status.__dict__
        else:
            status_dict = dict(status)
        
        s = str(status_dict.get("status", "")).upper()
        if s in ("COMPLETED", "FAILED", "ERROR"):
            return SimpleNamespace(**status_dict)
        
        if time.time() - start > timeout_sec:
            raise TimeoutError("Polling timed out.")

        attempt+=1
        sleep_time = min(poll_interval * (1 + attempt // 10), 15)
        time.sleep(sleep_time)

def download_url_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

# ---------- Subtitles helpers ----------

def srt_to_plain_text(srt_bytes: bytes) -> str:
    text = srt_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"(?m)^\d+\s*$", "", text)
    text = text.strip()
    return text

# ---------- Notes via LLM ----------

NOTES_PROMPT = PromptTemplate(
    template="""
You are a helpful teacher. Create compact class notes in bullet points from the following transcript text.

Rules:
- 4–7 concise bullets.
- Keep each bullet <= 2 lines.
- No new facts not present in text.
- Use plain language.

Transcript:
{text}

Notes:
""",
    input_variables=["text"]
)

def generate_notes_from_text(text: str) -> str:
    prompt_text = NOTES_PROMPT.format(text=text)
    try:
        resp = chat.invoke(prompt_text)
        return resp.content.strip()
    except Exception as e:
        return f"- (Error generating notes) {e}"

# ---------- Utility ----------

def save_bytes_to_tmpfile(b: bytes, suffix: str) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(b)
    return path
