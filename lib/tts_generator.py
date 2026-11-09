"""TTS Generator — gTTS-based, adapted for Synopsis."""

import os
import tempfile


def generate_tts(text: str, method: str = "gtts", lang: str = "fr") -> dict:
    if method == "gtts":
        return _gtts(text, lang)
    return {"success": False, "error": f"Méthode TTS inconnue: {method}"}


def _gtts(text: str, lang: str = "fr") -> dict:
    try:
        from gtts import gTTS
    except ImportError:
        return {"success": False, "error": "gTTS non installé. pip install gtts"}
    try:
        tts = gTTS(text=text[:5000], lang=lang, slow=False)
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        tts.save(path)
        return {"success": True, "audio_path": path}
    except Exception as e:
        return {"success": False, "error": f"gTTS: {e}"}
