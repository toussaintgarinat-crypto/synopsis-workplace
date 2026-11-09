"""Détecte les fonctions disponibles selon le mode et l'environnement."""
import importlib.util
import os
import shutil

LIGHT_REASON = "indisponible en ligne (mode léger)"
HEAVY = ("whisper", "reel", "llm_local", "llm_custom", "obsidian", "drive", "excalidraw", "images", "tts")


def is_light_mode() -> bool:
    if os.getenv("VERCEL"):  # Vercel définit VERCEL=1 : mode léger automatique
        return True
    return os.getenv("SYNOPSIS_MODE", "full").lower() == "light"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def get_capabilities() -> dict:
    # transcription_video : appel OpenRouter avec la clé du client, possible dans les deux modes.
    caps = {"resume": {"ok": True, "raison": ""}, "transcription_video": {"ok": True, "raison": ""}}
    if is_light_mode():
        for k in HEAVY:
            caps[k] = {"ok": False, "raison": LIGHT_REASON}
        return caps
    ffmpeg = shutil.which("ffmpeg") is not None
    caps["reel"] = {"ok": ffmpeg, "raison": "" if ffmpeg else "nécessite ffmpeg"}
    caps["whisper"] = {"ok": ffmpeg and (_has("whisper") or bool(os.getenv("OPENAI_API_KEY"))),
                       "raison": "nécessite ffmpeg et openai-whisper ou OPENAI_API_KEY"}
    caps["llm_local"] = {"ok": True, "raison": ""}
    caps["llm_custom"] = {"ok": bool(os.getenv("CUSTOM_LLM_URL")), "raison": "nécessite CUSTOM_LLM_URL"}
    caps["obsidian"] = {"ok": True, "raison": ""}
    caps["excalidraw"] = {"ok": True, "raison": ""}
    caps["drive"] = {"ok": bool(os.getenv("GOOGLE_DRIVE_CREDENTIALS")), "raison": "nécessite GOOGLE_DRIVE_CREDENTIALS"}
    caps["images"] = {"ok": True, "raison": ""}
    caps["tts"] = {"ok": _has("gtts"), "raison": "nécessite gtts"}
    for v in caps.values():
        if v["ok"]:
            v["raison"] = ""
    return caps
