"""Catalogue des modèles par tâche et résolution des modèles demandés par le client.

Tâches : resume (analyse des segments), fusion (fusion des analyses), qa (questions-réponses),
transcription_video (Gemini, vidéos sans sous-titres), transcription_audio (taille Whisper),
image (génération d'image).
"""
import hashlib
import os
import re
import threading
import time

import httpx

from lib import image_generator
from lib.video_transcriber import DEFAULT_MODEL as DEFAULT_VIDEO_MODEL, valider_modele_video
from lib.whisper_transcriber import WHISPER_SIZES, default_model_size

TACHES = ("resume", "fusion", "qa", "transcription_video", "transcription_audio", "image")
TACHES_TEXTE = ("resume", "fusion", "qa")
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,100}$")
CACHE_TTL = 600.0
CACHE_MAX = 256
# Fournisseurs dont la liste de modèles dépend de la clé (sinon : liste publique, entrée partagée).
LISTE_PAR_CLE = ("openai",)

PROVIDER_MODELS_API = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "opencode": None,
}

PROVIDER_FALLBACK_MODELS = {
    "openrouter": [
        "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro",
        "google/gemini-2.5-pro", "google/gemini-2.5-flash",
        "anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4.5",
        "openai/gpt-5.2", "openai/gpt-5.2-mini",
        "qwen/qwen3-coder", "mistral/pi-zero-2",
        "x-ai/grok-4", "amazon/nova-pro-v2",
    ],
    "openai": ["gpt-5.2", "gpt-5.2-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini", "o3"],
    "opencode": ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"],
}

VIDEO_FALLBACK = ["google/gemini-2.5-flash", "google/gemini-2.5-flash-lite", "google/gemini-2.5-pro"]
IMAGE_FALLBACK = [m for cfg in image_generator.IMAGE_PROVIDERS.values() for m in cfg["models"]]

# Filtres de l'API OpenRouter vérifiés (input_modalities=video, output_modalities=image).
_FILTRES_OPENROUTER = {
    "transcription_video": "?input_modalities=video",
    "image": "?output_modalities=image",
}

RAISON_VIDEO = "Transcription vidéo disponible uniquement avec OpenRouter (modèles Gemini)."
RAISON_IMAGE = "Génération d'image disponible uniquement avec OpenRouter."

_cache: dict = {}
_verrou = threading.Lock()


def vider_cache() -> None:
    with _verrou:
        _cache.clear()


def _cle_cache(provider: str, task: str, api_key: str) -> tuple:
    groupe = "texte" if task in TACHES_TEXTE else task
    # La liste d'OpenAI dépend de la clé : on ne la partage pas entre clés (empreinte, jamais la clé).
    # Les listes publiques (OpenRouter) ont une seule entrée, quelle que soit la clé.
    par_cle = provider in LISTE_PAR_CLE and groupe == "texte"
    empreinte = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key and par_cle else ""
    return (provider, groupe, empreinte)


def _ranger(cle: tuple, now: float, valeur: dict) -> None:
    """Insère dans le cache (verrou tenu) en le gardant sous CACHE_MAX entrées :
    purge des entrées expirées, puis des plus anciennes."""
    if cle not in _cache and len(_cache) >= CACHE_MAX:
        for k in [k for k, (t, _) in _cache.items() if now - t >= CACHE_TTL]:
            del _cache[k]
        while len(_cache) >= CACHE_MAX:
            del _cache[min(_cache, key=lambda k: _cache[k][0])]
    _cache[cle] = (now, valeur)


def _interroger(url: str, api_key: str):
    """Liste (ids triés, noms) depuis une API /models ; None en cas d'échec."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = httpx.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()["data"]
    except Exception:
        return None
    ids, noms = [], {}
    for m in data:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if isinstance(mid, str) and MODEL_RE.fullmatch(mid):
            ids.append(mid)
            nom = m.get("name")
            if isinstance(nom, str) and nom.strip() and nom != mid:
                noms[mid] = nom.strip()[:120]
    return sorted(set(ids)), noms


def fetch_models(provider: str, task: str = "resume", api_key: str = "") -> dict:
    """{"models": [ids], "source": "api"|"fallback", "noms"?: {id: nom}, "raison"?: str}."""
    if task not in TACHES:
        raise ValueError(f"Tâche inconnue : {task}")
    provider = provider or "openrouter"
    api_key = api_key or ""

    if task == "transcription_audio":
        return {"models": list(WHISPER_SIZES), "source": "fallback"}
    if task in ("transcription_video", "image") and provider != "openrouter":
        return {"models": [], "source": "fallback", "raison": RAISON_VIDEO if task == "transcription_video" else RAISON_IMAGE}

    if task in TACHES_TEXTE:
        base = PROVIDER_MODELS_API.get(provider)
        url = base
        secours = PROVIDER_FALLBACK_MODELS.get(provider, [])
    else:
        base = PROVIDER_MODELS_API["openrouter"]
        url = base + _FILTRES_OPENROUTER[task]
        secours = VIDEO_FALLBACK if task == "transcription_video" else IMAGE_FALLBACK
    if not url:
        return {"models": list(secours), "source": "fallback"}

    cle = _cle_cache(provider, task, api_key)
    now = time.monotonic()
    with _verrou:
        hit = _cache.get(cle)
        if hit and now - hit[0] < CACHE_TTL:
            return _copie(hit[1])

    res = _interroger(url, api_key)
    if res is None:
        return {"models": list(secours), "source": "fallback"}
    ids, noms = res
    if task == "transcription_video":
        ids = [m for m in ids if m.startswith("google/")]
        noms = {k: v for k, v in noms.items() if k in ids}
    out = {"models": ids, "source": "api"}
    if noms:
        out["noms"] = noms
    with _verrou:
        _ranger(cle, now, _copie(out))
    return _copie(out)


def _copie(d: dict) -> dict:
    out = dict(d)
    out["models"] = list(d["models"])
    if "noms" in d:
        out["noms"] = dict(d["noms"])
    return out


def _valider(task: str, valeur) -> str:
    if valeur is None:
        return ""
    if not isinstance(valeur, str):
        raise ValueError(f"Modèle invalide pour la tâche « {task} » : texte attendu.")
    v = valeur.strip()
    if v and not MODEL_RE.fullmatch(v):
        raise ValueError(f"Modèle invalide pour la tâche « {task} » : 1 à 100 caractères parmi lettres, chiffres, . _ : / -.")
    return v


def resolve_models(req_modeles, default_model: str = "") -> dict:
    """Modèle effectif de chaque tâche. Vide -> défaut ; clé inconnue ignorée ; invalide -> ValueError."""
    if req_modeles is None:
        req_modeles = {}
    if not isinstance(req_modeles, dict):
        raise ValueError("Champ « modeles » invalide : objet {tâche: modèle} attendu.")
    defaut = _valider("resume", default_model) or os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
    if not MODEL_RE.fullmatch(defaut):
        raise ValueError("Modèle par défaut invalide.")
    v = {t: _valider(t, req_modeles.get(t)) for t in TACHES}
    resume = v["resume"] or defaut
    out = {
        "resume": resume,
        "fusion": v["fusion"] or resume,
        "qa": v["qa"] or resume,
        "transcription_video": valider_modele_video(v["transcription_video"] or DEFAULT_VIDEO_MODEL),
        "transcription_audio": v["transcription_audio"] or default_model_size(),
        "image": v["image"] or image_generator.DEFAULT_MODEL,
    }
    if v["transcription_audio"] and v["transcription_audio"] not in WHISPER_SIZES:
        raise ValueError("Modèle Whisper invalide : choisissez parmi " + ", ".join(WHISPER_SIZES) + ".")
    return out
