import os
import io
import base64
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import tempfile
import speech_recognition as sr

from core import agentstate, generate_explanation, murf_stream_tts, generate_summary
from dub import (
    download_youtube_highest_mp4,
    create_dub_job,
    poll_job_until_complete,
    download_url_bytes,
    srt_to_plain_text,
    generate_notes_from_text,
    save_bytes_to_tmpfile,
    TARGET_LOCALES,
)

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

def gemini_asr(audio_path: str) -> str:
    """
    Use Gemini for *transcription only*. We force 'transcribe verbatim'
    so it doesn't generate explanations.
    """
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        prompt = (
            "TRANSCRIBE VERBATIM.\n"
            "Return ONLY the user's spoken words as plain text.\n"
            "Do NOT summarize, translate, explain, or add anything.\n"
            "No punctuation cleanup beyond what was actually spoken."
        )

        resp = model.generate_content(
            [
                prompt,
                {"mime_type": "audio/wav", "data": audio_bytes},
            ]
        )

        text = (resp.text or "").strip()
        # extra guardrails: keep it short; if Gemini slips into an essay, fallback to first line
        if text.count("\n") > 2 or len(text) > 500:
            text = text.splitlines()[0].strip()
        return text
    except Exception as e:
        return f"[Transcription failed: {e}]"

# ---- Streamlit page config ----
st.set_page_config(page_title="AI Teacher", page_icon="🧑‍🏫", layout="wide")
st.title("🧑‍🏫 AI Teacher")

if "query" not in st.session_state:
    st.session_state["query"] = ""

# Tabs: Q&A teacher vs Video dubbing
tab1, tab2 = st.tabs(["❓ Ask AI Teacher", "🎬 Video Dubbing + Notes"])

# -----------------------------
# TAB 1: Q&A TEACHER
# -----------------------------
with tab1:
    st.subheader("Ask AI Teacher (Voice + Notes)")

    col_a, col_b = st.columns([1, 3])

    with col_a:
        if st.button("🎤 Speak now"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening... Please speak into the microphone.")
                audio_data = r.listen(source)
                st.info("Processing...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        wav_path = tmp.name
                        with open(wav_path, "wb") as f:
                            f.write(audio_data.get_wav_data())

                    with st.spinner("Transcribing..."):
                        transcript = gemini_asr(wav_path)

                    if not transcript or transcript.startswith("[Transcription failed"):
                        st.error(transcript or "Could not transcribe audio. Please try again.")
                    else:
                        # ✅ Write ONLY the transcript into the input box
                        st.session_state["query"] = transcript
                        st.success(f"Captured question: “{transcript}”")
                except Exception as e:
                    st.error(f"Speech input failed: {e}")

    # with col_b:
        # ✅ Input bound to session_state (no more weird overwrites)
    st.text_input("Ask your question here:", key="query", placeholder="Type or use 'Speak now'")

    # Generate when user clicks Explain
    if st.button("Explain", key="explain_btn"):
        query = st.session_state["query"].strip()
        if not query:
            st.warning("Please enter a question first.")
        else:
            # Initial state
            state: agentstate = {
                "query": query,
                "lang": "english",
                "explanation": "",
                "audio_url": None,
                "summary": ""
            }

            # Pipeline: Explanation (not shown), TTS of explanation, Summary (shown)
            state = generate_explanation(state)
            state = murf_stream_tts(state)     # 🔊 explanation → speech
            state = generate_summary(state)    # 📝 show notes

            # UI
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📖 Notes")
                st.markdown(state["summary"] or "_No notes generated._")

            with col2:
                st.subheader("🔊 Listen")
                audio_io = state.get("audio_url")
                if audio_io:
                    # Convert BytesIO/bytes to base64 for HTML audio (autoplay)
                    if isinstance(audio_io, io.BytesIO):
                        audio_bytes = audio_io.getvalue()
                    elif isinstance(audio_io, bytes):
                        audio_bytes = audio_io
                    else:
                        audio_bytes = None

                    if audio_bytes:
                        b64_audio = base64.b64encode(audio_bytes).decode()
                        audio_html = f"""
                            <audio id="tts_audio" autoplay>
                                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
                            </audio>
                            <div style="margin-top:10px;">
                                <button onclick="document.getElementById('tts_audio').play()">🔁 Listen Again</button>
                                <button onclick="document.getElementById('tts_audio').pause()">⏹ Stop</button>
                            </div>
                        """
                        components.html(audio_html, height=120)
                    else:
                        st.error("Audio format not recognized.")
                else:
                    st.error("No audio was generated.")

# -----------------------------
# TAB 2: VIDEO DUBBING
# -----------------------------
with tab2:
    st.subheader("🎬 Video Dubbing + Notes")

    with st.expander("How it works", expanded=False):
        st.markdown("""
1. Paste a **YouTube link** *or* upload a video  
2. Choose a **Target Language**  
3. Click **Dub** → We send it to Murf Dub API  
4. Compare **Original vs Dubbed** side-by-side  
5. Download **Dubbed MP4 + Subtitles**  
6. Generate **Class Notes** automatically  
        """)

    yt_url = st.text_input("YouTube URL (optional)")
    uploaded_file = st.file_uploader("Or upload a video file (.mp4, .mov, .mkv, .webm)",
                                     type=["mp4","mov","mkv","webm"])
    target_locale = st.selectbox("Target Language", TARGET_LOCALES, index=0)

    if st.button("🎬 Dub", key="dub_btn"):
        original_video_path = None
        dubbed_video_bytes = None
        srt_bytes = None

        # Step 1: Get source video
        with st.status("Preparing source video...", expanded=True) as status:
            try:
                if yt_url:
                    st.write("Downloading from YouTube…")
                    original_video_path = download_youtube_highest_mp4(yt_url)
                elif uploaded_file is not None:
                    st.write("Saving uploaded file…")
                    os.makedirs("uploads", exist_ok=True)
                    original_video_path = os.path.join("uploads", uploaded_file.name)
                    with open(original_video_path, "wb") as f:
                        f.write(uploaded_file.read())
                else:
                    st.error("Please paste a YouTube link or upload a file.")
                    st.stop()

                status.update(label="Source ready ✅", state="complete")
            except Exception as e:
                st.error(f"Failed to prepare source: {e}")
                st.stop()

        # Step 2: Create Dub Job
        with st.status("Creating dub job…", expanded=True) as status:
            try:
                create_res = create_dub_job(original_video_path, target_locale=target_locale)
                job_id = create_res.id
                st.write(f"Job ID: `{job_id}`")
                status.update(label="Dub job created ✅", state="complete")
            except Exception as e:
                st.error(f"Failed to create dub job: {e}")
                st.stop()

        # Step 3: Poll job until complete
        with st.status("Dubbing in progress…", expanded=True) as status:
            try:
                final = poll_job_until_complete(job_id)

                if final.status.upper() != "COMPLETED":
                    st.error(f"Dub job did not complete. Status: {final.status}")
                    st.stop()

                details = final.download_details
                if not details:
                    st.error("No download details returned.")
                    st.stop()

                first = details[0]
                download_url = first.download_url
                srt_url = getattr(first, "download_srt_url", None)

                if not download_url:
                    st.error("No dubbed video found.")
                    st.stop()

                dubbed_video_bytes = download_url_bytes(download_url)
                if srt_url:
                    srt_bytes = download_url_bytes(srt_url)

                status.update(label="Dubbing complete ✅", state="complete")
            except TimeoutError:
                st.error("Dubbing took too long and timed out. Please retry later.")
                st.stop()
            except Exception as e:
                st.error(f"Error during dubbing: {e}")
                st.stop()

        # Step 4: Show results
        if dubbed_video_bytes:
            st.divider()
            st.subheader("▶️ Compare Videos")
            col1, col2 = st.columns(2)

            with col1:
                st.caption("Original")
                if yt_url:
                    st.video(yt_url)
                else:
                    st.video(original_video_path)

            with col2:
                st.caption(f"Dubbed ({target_locale})")
                dubbed_path = save_bytes_to_tmpfile(dubbed_video_bytes, suffix=".mp4")
                st.video(dubbed_path)

            st.subheader("⬇️ Downloads")
            st.download_button("Download Dubbed MP4", data=dubbed_video_bytes,
                               file_name="dubbed.mp4", mime="video/mp4")
            if srt_bytes:
                st.download_button("Download Subtitles (SRT)", data=srt_bytes,
                                   file_name="dubbed.srt", mime="text/plain")

            st.subheader("🗒️ Study Notes")
            if st.button("Generate Notes from Subtitles", key="notes_btn"):
                plain = srt_to_plain_text(srt_bytes) if srt_bytes else "(No subtitles available)"
                with st.spinner("Generating notes…"):
                    notes = generate_notes_from_text(plain)
                st.markdown(notes)
