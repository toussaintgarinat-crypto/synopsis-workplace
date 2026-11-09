"""YouTube Transcript Extractor — adapted for Synopsis."""

import logging
import re
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript,
)
import httpx

try:  # absents des anciennes versions de youtube-transcript-api
    from youtube_transcript_api._errors import RequestBlocked  # IpBlocked en hérite
except ImportError:  # pragma: no cover
    RequestBlocked = None
try:
    from youtube_transcript_api._errors import IpBlocked
except ImportError:  # pragma: no cover
    IpBlocked = None

from lib.capabilities import is_light_mode

logger = logging.getLogger(__name__)


class SansSousTitres(ValueError):
    """La vidéo est lisible mais YouTube ne fournit pas de sous-titres (désactivés, absents,
    vides, ou requête bloquée) : seul cas où une transcription par modèle vidéo a du sens."""


_BLOQUE = tuple(c for c in (RequestBlocked, IpBlocked) if c is not None)


def _classer(e: Exception) -> ValueError:
    if isinstance(e, TranscriptsDisabled):
        return SansSousTitres("Transcript désactivé sur cette vidéo YouTube")
    if isinstance(e, NoTranscriptFound):
        return SansSousTitres("Aucun transcript disponible pour cette vidéo")
    if _BLOQUE and isinstance(e, _BLOQUE):
        return SansSousTitres(f"Erreur transcript: {e}")
    return ValueError(f"Erreur transcript: {e}")


YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
}
_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def extract_youtube_id(url: str) -> Optional[str]:
    """ID de la vidéo, seulement si l'URL est réellement une URL YouTube (hôte vérifié)."""
    u = str(url or "").strip()
    if not u or u.startswith("-"):
        return None
    if "://" not in u:
        u = "https://" + u
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
    except ValueError:
        return None
    if p.scheme not in ("http", "https") or host not in YOUTUBE_HOSTS:
        return None
    if host == "youtu.be":
        cand = p.path.lstrip("/").split("/")[0]
    elif p.path.rstrip("/") == "/watch":
        cand = (parse_qs(p.query).get("v") or [""])[0]
    else:
        m = re.match(r"^/(?:embed|shorts|live|v)/([^/]+)", p.path)
        cand = m.group(1) if m else ""
    return cand if _ID_RE.fullmatch(cand) else None


def detect_channel(url: str) -> bool:
    return bool(re.search(r'youtube\.com/(@|channel/|c/)', url))


def get_video_title(video_id: str, timeout: float = None) -> str:
    """Titre via oEmbed. Délai court en mode léger (budget de 60 s par requête)."""
    if timeout is None:
        timeout = 3 if is_light_mode() else 10
    try:
        r = httpx.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json", timeout=timeout)
        if r.status_code == 200:
            return r.json().get("title", f"Video {video_id}")
    except Exception:
        logger.debug("Could not fetch title for %s", video_id)
    return f"Video {video_id}"


def get_youtube_transcript(url: str, languages: list = None) -> dict:
    if languages is None:
        languages = ['fr', 'en', 'es', 'de', 'it', 'pt']

    if detect_channel(url):
        raise ValueError("Cette URL est une chaîne YouTube. Fournissez l'URL d'une vidéo.")

    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError("URL YouTube invalide. Format: youtube.com/watch?v=ID ou youtu.be/ID")

    yt_api = YouTubeTranscriptApi()
    transcript_data = None
    original_lang = None

    # Un seul appel : la bibliothèque choisit la première langue disponible dans l'ordre de
    # priorité (comme l'ancienne boucle langue par langue, en 1 aller-retour au lieu de 6).
    try:
        transcript = yt_api.fetch(video_id, languages=list(languages))
        transcript_data = list(transcript)
        original_lang = getattr(transcript, 'language_code', None) or languages[0]
    except CouldNotRetrieveTranscript:
        pass  # classé par l'appel suivant (sans préférence de langue)
    except Exception:
        logger.debug("Sous-titres : premier essai échoué pour %s", video_id)

    if not transcript_data:
        try:
            transcript = yt_api.fetch(video_id)
            transcript_data = list(transcript)
            original_lang = getattr(transcript, 'language_code', 'en') or 'en'
        except CouldNotRetrieveTranscript as e:
            if "403" not in str(e) or isinstance(e, (TranscriptsDisabled, NoTranscriptFound)):
                raise _classer(e) from e
            time.sleep(3)
            try:
                transcript = yt_api.fetch(video_id, languages=languages)
                transcript_data = list(transcript)
                original_lang = 'en'
            except CouldNotRetrieveTranscript as e2:
                raise _classer(e2) from e2

    if not transcript_data:
        raise SansSousTitres("Transcript vide")

    formatted = [{'text': e.text, 'start': e.start, 'duration': e.duration} for e in transcript_data]
    last = formatted[-1]
    total_minutes = (last['start'] + last['duration']) / 60

    return {
        'video_id': video_id,
        'transcript': formatted,
        'language': original_lang or 'unknown',
        'title': get_video_title(video_id),
        'total_duration_minutes': total_minutes,
        'entries_count': len(formatted),
    }
