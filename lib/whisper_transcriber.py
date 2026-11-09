"""Transcription locale (openai-whisper) avec repli sur l'API OpenAI."""
import os


def segments_to_transcript(segments):
    return [{"text": s["text"].strip(), "start": float(s["start"]), "duration": float(s["end"]) - float(s["start"])}
            for s in segments if s["text"].strip()]


def transcribe_file(path, language=""):
    try:
        import whisper
        model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
        result = model.transcribe(path, language=language or None)
        return segments_to_transcript(result["segments"])
    except ImportError:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("Whisper indisponible : installez openai-whisper ou définissez OPENAI_API_KEY")
        import httpx
        with open(path, "rb") as f:
            r = httpx.post("https://api.openai.com/v1/audio/transcriptions",
                           headers={"Authorization": f"Bearer {key}"},
                           files={"file": f}, data={"model": "whisper-1", "response_format": "verbose_json"}, timeout=600)
        r.raise_for_status()
        return segments_to_transcript(r.json()["segments"])
