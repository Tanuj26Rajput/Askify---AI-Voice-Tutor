from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from core import workflow, agentstate
import io
from dub import (
    create_dub_job,
    poll_job_until_complete,
    download_url_bytes,
    srt_to_plain_text,
    generate_notes_from_text
)

app = FastAPI(title="Askify API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    query: str

@app.post("/api/ask")
async def api_ask(payload: AskIn):
    state: agentstate = {
        "query": payload.query,
        "lang": "english",
        "explanation": "",
        "audio_url": None,
        "summary": "",
    }

    result = await workflow.ainvoke(state)

    explanation = result.get("explanation", "") or ""
    summary = result.get("summary", "") or ""
    audio_b64: Optional[str] = None
    audio_obj = result.get("audio_url", None)
    if isinstance(audio_obj, io.BytesIO):
        import base64
        audio_b64 = base64.b64encode(audio_obj.read()).decode("utf-8")

    return {
        "explanation": explanation,
        "summary": summary,
        "audio_b64": audio_b64
    }

class DubIn(BaseModel):
    youtube_url: str
    target_locale: str

@app.post("/api/dub")
def api_dub_start(payload: DubIn):
    from dub import download_youtube_highest_mp4
    mp4_path = download_youtube_highest_mp4(payload.youtube_url)
    job = create_dub_job(file_path=mp4_path, target_locale=payload.target_locale) 
    return {"job_id": job.id}

@app.get("/api/dub_status")
def api_dub_status(job_id: str):
    status = poll_job_until_complete(job_id=job_id)
    s = str(getattr(status, "status", "")).upper()

    if s not in ("COMPLETED", "FAILED", "ERROR"):
        return {"status": s}

    # Adapt to MurfDub response fields (names can vary; handle generously)
    result = getattr(status, "result", None) or {}
    dubbed_video_url = result.get("dubbed_url") or result.get("video_url") or result.get("output_url")
    subtitles_url = result.get("subtitles_url") or result.get("srt_url") or result.get("captions_url")

    notes = None
    if subtitles_url:
        try:
            srt_bytes = download_url_bytes(subtitles_url)
            transcript_text = srt_to_plain_text(srt_bytes)
            notes = generate_notes_from_text(transcript_text)
        except Exception as e:
            notes = f"- (Could not generate notes) {e}"

    return {
        "status": s.lower(),
        "dubbed_video_url": dubbed_video_url,
        "subtitles_url": subtitles_url,
        "notes": notes
    }