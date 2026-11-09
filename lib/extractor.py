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

logger = logging.getLogger(__name__)


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


def get_video_title(video_id: str) -> str:
    try:
        r = httpx.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json", timeout=10)
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

    for lang in languages:
        try:
            transcript = yt_api.fetch(video_id, languages=[lang])
            transcript_data = list(transcript)
            original_lang = lang
            break
        except Exception:
            continue

    if not transcript_data:
        try:
            transcript = yt_api.fetch(video_id)
            transcript_data = list(transcript)
            original_lang = getattr(transcript, 'language_code', 'en') or 'en'
        except TranscriptsDisabled:
            raise ValueError("Transcript désactivé sur cette vidéo YouTube")
        except NoTranscriptFound:
            raise ValueError("Aucun transcript disponible pour cette vidéo")
        except CouldNotRetrieveTranscript as e:
            if "403" in str(e):
                time.sleep(3)
                transcript = yt_api.fetch(video_id, languages=languages)
                transcript_data = list(transcript)
                original_lang = 'en'
            else:
                raise ValueError(f"Erreur transcript: {e}")

    if not transcript_data:
        raise ValueError("Transcript vide")

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
