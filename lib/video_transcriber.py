"""Transcription de secours d'une vidéo YouTube sans sous-titres : OpenRouter + Gemini.

Gemini (via le fournisseur Google AI Studio d'OpenRouter) sait lire directement une URL
YouTube. On lui demande une transcription horodatée, qu'on convertit au même format que
les sous-titres (lib.extractor) : [{'text', 'start', 'duration'}].

Sécurité : l'URL envoyée est reconstruite à partir de l'identifiant (11 caractères) ;
la chaîne du client n'est jamais transmise. Les messages d'erreur sont purgés des secrets.
"""
import os
import re

import httpx

from lib.capabilities import is_light_mode
from lib.extractor import extract_youtube_id, get_video_title
from lib.security import scrub

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
PROMPT = ("Transcris fidèlement la parole de cette vidéo. Format strict : une ligne par phrase, "
          "préfixée de l'horodatage [MM:SS]. Aucun commentaire.")
PROVIDER_ROUTING = {"only": ["google-ai-studio"], "allow_fallbacks": False}
LAST_DURATION = 5.0
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,100}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# 60 s = limite de toute la requête en version en ligne (transcription + résumé) ; l'appel de
# transcription lui-même est coupé plus tôt (50 s au plus, moins si le budget restant est faible).
MSG_TIMEOUT_LIGHT = ("Vidéo trop longue pour être transcrite en ligne : la version en ligne doit transcrire et "
                     "résumer en moins de 60 s. Essayez une vidéo plus courte ou la version locale.")

def _ts(p: str) -> str:
    return rf"(?:(?P<{p}h>\d{{1,2}}):)?(?P<{p}m>\d{{1,3}}):(?P<{p}s>\d{{2}})(?:[.,]\d+)?"


_FIN = r"(?:\s*[-–—]\s*(?:\d{1,2}:)?\d{1,3}:\d{2}(?:[.,]\d+)?)?"
# Préfixes tolérés : puces markdown, citation, gras. Horodatage entre [ ] / ( ), ou nu en début de
# ligne s'il est suivi d'un espace, d'un séparateur, du gras ou de la fin de ligne ; plage
# « début - fin » ; séparateur ':' ou '-' avant le texte.
_LIGNE = re.compile(
    r"^[\s>*\-•]*\**\s*(?:[\[(]\s*" + _ts("b") + _FIN + r"\s*[\])]"
    r"|" + _ts("n") + _FIN + r"(?=[\s*:\-–—]|$))"
    r"\**\s*[:\-–—]?\s*\**\s*(?P<texte>.*)$"
)


def _horodatage(m):
    """(h, m, s) de la ligne, ou None si hors limites (minutes ou secondes > 59 : ligne parasite)."""
    p = "b" if m.group("bm") is not None else "n"
    h, mi, se = m.group(p + "h"), int(m.group(p + "m")), int(m.group(p + "s"))
    if se > 59 or mi > 59:
        return None
    return h, mi, se


def _secondes(h, m, s) -> float:
    return float(int(h or 0) * 3600 + int(m) * 60 + int(s))


def parse_timestamped_transcript(text) -> list[dict]:
    """Convertit « [MM:SS] phrase » (une par ligne) en entrées {'text', 'start', 'duration'}.

    Tolère [ MM:SS ], [H:MM:SS], [HH:MM:SS], les horodatages sans crochets en début de ligne
    (« 00:05 texte », « **00:05** texte », « 00:05 - texte »), les blocs de code, puces, lignes
    vides et lignes de continuation. Minutes ou secondes > 59 : ligne ignorée. Un horodatage qui recule (ou se répète) est fusionné avec
    l'entrée précédente pour garder une suite croissante. Lève ValueError si rien n'est exploitable.
    """
    entrees: list[dict] = []
    for brute in str(text or "").splitlines():
        ligne = brute.strip()
        if not ligne or ligne.startswith("```"):
            continue
        m = _LIGNE.match(ligne)
        if not m:
            # Continuation de la phrase précédente (le préambule avant le 1er horodatage est ignoré).
            if entrees:
                entrees[-1]["text"] = (entrees[-1]["text"] + " " + ligne).strip()
            continue
        hms = _horodatage(m)
        if hms is None:
            continue  # horodatage impossible (ex. 00:75) : ligne ignorée
        start = _secondes(*hms)
        texte = m.group("texte").strip().strip("*").strip()
        if entrees and start <= entrees[-1]["start"]:
            if texte:
                entrees[-1]["text"] = (entrees[-1]["text"] + " " + texte).strip()
            continue
        entrees.append({"text": texte, "start": start})
    entrees = [e for e in entrees if e["text"]]
    if not entrees:
        raise ValueError("La transcription générée par le modèle ne contient aucun horodatage exploitable "
                         "(format attendu : « [MM:SS] phrase »).")
    for i, e in enumerate(entrees):
        e["duration"] = (entrees[i + 1]["start"] - e["start"]) if i + 1 < len(entrees) else LAST_DURATION
    return [{"text": e["text"], "start": e["start"], "duration": e["duration"]} for e in entrees]


def valider_modele_video(model: str) -> str:
    model = (model or "").strip() or DEFAULT_MODEL
    if not MODEL_RE.fullmatch(model):
        raise ValueError("Modèle de transcription vidéo invalide (caractères autorisés : lettres, chiffres, . _ : / -).")
    if not model.startswith("google/"):
        raise ValueError("Modèle de transcription vidéo non pris en charge : seuls les modèles Gemini (google/…) "
                         "savent lire une URL YouTube via OpenRouter.")
    return model


def _video_id(video_id_or_url: str) -> str:
    s = str(video_id_or_url or "").strip()
    vid = s if _ID_RE.fullmatch(s) else extract_youtube_id(s)
    if not vid:
        raise ValueError("URL YouTube invalide. Format: youtube.com/watch?v=ID ou youtu.be/ID")
    return vid


def _contenu(data: dict) -> str:
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise ValueError("Transcription vidéo refusée par OpenRouter : " + scrub(msg, limite=200))
    try:
        c = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Réponse inattendue d'OpenRouter pendant la transcription vidéo.")
    if isinstance(c, list):
        c = "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return str(c or "")


def transcribe_youtube_via_llm(video_id_or_url: str, api_key: str, model: str = "", timeout: float = None,
                               provider: str = "openrouter") -> dict:
    """Transcrit une vidéo YouTube avec Gemini via OpenRouter. Même forme de retour que
    lib.extractor.get_youtube_transcript, plus `source: "llm-video"` et `model`."""
    if (provider or "openrouter") != "openrouter":
        raise ValueError("La transcription par modèle vidéo n'est disponible qu'avec le fournisseur OpenRouter.")
    vid = _video_id(video_id_or_url)
    model = valider_modele_video(model)
    if not (api_key or "").strip():
        raise ValueError("Clé API OpenRouter requise pour transcrire la vidéo avec un modèle vidéo.")
    light = is_light_mode()
    if timeout is None:
        try:
            timeout = float(os.getenv("LLM_VIDEO_TIMEOUT_LIGHT" if light else "LLM_VIDEO_TIMEOUT", "50" if light else "300"))
        except ValueError:
            timeout = 50.0 if light else 300.0
        timeout = int(timeout) if float(timeout).is_integer() else timeout

    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "video_url", "video_url": {"url": f"https://www.youtube.com/watch?v={vid}"}},
        ]}],
        "provider": dict(PROVIDER_ROUTING, only=list(PROVIDER_ROUTING["only"])),
    }
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json",
               "HTTP-Referer": "https://synopsis.local", "X-Title": "Synopsis"}
    try:
        r = httpx.post(OPENROUTER_URL, json=body, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        if light:
            raise ValueError(MSG_TIMEOUT_LIGHT)
        raise ValueError(f"Transcription vidéo interrompue : délai dépassé ({timeout} s). Essayez une vidéo plus courte.")
    except httpx.HTTPError as e:
        raise ValueError("Transcription vidéo impossible (réseau) : " + scrub(e, limite=200))

    code = r.status_code
    if code == 402:
        raise ValueError("Crédits OpenRouter insuffisants pour transcrire la vidéo (402) : rechargez votre compte OpenRouter.")
    if code in (401, 403):
        raise ValueError(f"Clé API refusée par OpenRouter ({code}) : vérifiez votre clé.")
    if code == 429:
        raise ValueError("Limite de requêtes atteinte chez OpenRouter (429) : réessayez dans une minute.")
    if code != 200:
        raise ValueError(f"Transcription vidéo impossible avec {model} (HTTP {code}) : " + scrub(getattr(r, "text", ""), limite=200))
    try:
        data = r.json()
    except ValueError:
        raise ValueError("Réponse illisible d'OpenRouter pendant la transcription vidéo.")

    entries = parse_timestamped_transcript(_contenu(data))
    last = entries[-1]
    return {
        "video_id": vid,
        "transcript": entries,
        "language": "unknown",
        "title": get_video_title(vid),
        "total_duration_minutes": (last["start"] + last["duration"]) / 60,
        "entries_count": len(entries),
        "source": "llm-video",
        "model": model,
    }
