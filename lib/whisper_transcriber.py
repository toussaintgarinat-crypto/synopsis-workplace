"""Transcription locale (openai-whisper) avec repli sur l'API OpenAI."""
import os

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large"]


def default_model_size() -> str:
    return os.getenv("WHISPER_MODEL", "base") or "base"


def segments_to_transcript(segments):
    return [{"text": s["text"].strip(), "start": float(s["start"]), "duration": float(s["end"]) - float(s["start"])}
            for s in segments if s["text"].strip()]


API_MODEL = "whisper-1"


def transcribe_file(path, language="", model_size=""):
    """`model_size` : une des WHISPER_SIZES (ou la valeur serveur WHISPER_MODEL) ; vide = WHISPER_MODEL/base."""
    return transcribe_file_with_model(path, language, model_size)[0]


def transcribe_file_with_model(path, language="", model_size=""):
    """Comme transcribe_file, mais renvoie (segments, modèle réellement utilisé) : la taille locale
    (« base », « small »…) ou « whisper-1 » si le repli sur l'API OpenAI a servi."""
    model_size = (model_size or "").strip() or default_model_size()
    if model_size not in WHISPER_SIZES and model_size != default_model_size():
        raise ValueError("Modèle Whisper invalide : choisissez parmi " + ", ".join(WHISPER_SIZES) + ".")
    try:
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(path, language=language or None)
        return segments_to_transcript(result["segments"]), model_size
    except ImportError:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("Whisper indisponible : installez openai-whisper ou définissez OPENAI_API_KEY")
        import httpx
        with open(path, "rb") as f:
            r = httpx.post("https://api.openai.com/v1/audio/transcriptions",
                           headers={"Authorization": f"Bearer {key}"},
                           files={"file": f}, data={"model": API_MODEL, "response_format": "verbose_json"}, timeout=600)
        r.raise_for_status()
        return segments_to_transcript(r.json()["segments"]), API_MODEL
