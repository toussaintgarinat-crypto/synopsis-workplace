"""I1 : le repli Gemini ne se déclenche que pour les vidéos réellement sans sous-titres
(SansSousTitres), jamais pour une vidéo supprimée, privée, d'âge restreint ou injouable."""
import pytest
from youtube_transcript_api import _errors as yte

import main
from lib import extractor
from lib.extractor import SansSousTitres

VID = "jNQXAC9IVRw"
URL = f"https://www.youtube.com/watch?v={VID}"


def _fake_api(monkeypatch, exc_factory):
    calls = []

    class FakeApi:
        def fetch(self, video_id, languages=None):
            calls.append(languages)
            raise exc_factory()
    monkeypatch.setattr(extractor, "YouTubeTranscriptApi", FakeApi)
    monkeypatch.setattr(extractor, "get_video_title", lambda vid, timeout=None: "T")
    monkeypatch.setattr(extractor.time, "sleep", lambda s: None)
    return calls


SANS_SOUS_TITRES = {
    "disabled": lambda: yte.TranscriptsDisabled(VID),
    "not_found": lambda: yte.NoTranscriptFound(VID, ["fr"], "aucune"),
    "ip_blocked": lambda: yte.IpBlocked(VID),
    "request_blocked": lambda: yte.RequestBlocked(VID),
}
AUTRES = {
    "unavailable": lambda: yte.VideoUnavailable(VID),
    "invalid_id": lambda: yte.InvalidVideoId(VID),
    "age_restricted": lambda: yte.AgeRestricted(VID),
    "unplayable": lambda: yte.VideoUnplayable(VID, "privée", []),
}


@pytest.mark.parametrize("nom", sorted(SANS_SOUS_TITRES))
def test_extractor_raises_sans_sous_titres(monkeypatch, nom):
    _fake_api(monkeypatch, SANS_SOUS_TITRES[nom])
    with pytest.raises(SansSousTitres):
        extractor.get_youtube_transcript(URL)


@pytest.mark.parametrize("nom", sorted(AUTRES))
def test_extractor_other_errors_plain_value_error(monkeypatch, nom):
    _fake_api(monkeypatch, AUTRES[nom])
    with pytest.raises(ValueError) as e:
        extractor.get_youtube_transcript(URL)
    assert not isinstance(e.value, SansSousTitres)
    assert str(e.value).startswith("Erreur transcript: ")


def test_extractor_empty_transcript_is_sans_sous_titres(monkeypatch):
    class FakeApi:
        def fetch(self, video_id, languages=None):
            return []
    monkeypatch.setattr(extractor, "YouTubeTranscriptApi", FakeApi)
    with pytest.raises(SansSousTitres, match="Transcript vide"):
        extractor.get_youtube_transcript(URL)


def test_extractor_messages_unchanged(monkeypatch):
    _fake_api(monkeypatch, SANS_SOUS_TITRES["disabled"])
    with pytest.raises(SansSousTitres, match="^Transcript désactivé sur cette vidéo YouTube$"):
        extractor.get_youtube_transcript(URL)
    _fake_api(monkeypatch, SANS_SOUS_TITRES["not_found"])
    with pytest.raises(SansSousTitres, match="^Aucun transcript disponible pour cette vidéo$"):
        extractor.get_youtube_transcript(URL)


def test_extractor_stops_language_loop_on_non_language_error(monkeypatch):
    calls = _fake_api(monkeypatch, AUTRES["unavailable"])
    with pytest.raises(ValueError):
        extractor.get_youtube_transcript(URL)
    assert len(calls) <= 2  # pas 7 allers-retours pour une vidéo indisponible


# ---------- _summarize ----------

@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    calls = []

    def fake(src, api_key, model, timeout=None, provider="openrouter"):
        calls.append(src)
        return {"video_id": VID, "transcript": [{"text": "a", "start": 1.0, "duration": 1.0}], "language": "unknown",
                "title": "T", "total_duration_minutes": 0.1, "entries_count": 1, "source": "llm-video", "model": model}
    monkeypatch.setattr(main, "transcribe_youtube_via_llm", fake)
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: "## 📍 Chapitrage Temporel\n| [00:01] | *S* | D |")
    return calls


@pytest.mark.parametrize("nom", sorted(AUTRES))
def test_summarize_no_fallback_for_unavailable_videos(setup, monkeypatch, nom):
    _fake_api(monkeypatch, AUTRES[nom])
    with pytest.raises(ValueError) as e:
        main._summarize(URL, api_key="sk-or-client", provider="openrouter", transcription_llm=True)
    assert setup == []
    assert "Erreur transcript" in str(e.value) and "Whisper" in str(e.value)  # conseil toujours ajouté


@pytest.mark.parametrize("nom", sorted(SANS_SOUS_TITRES))
def test_summarize_fallback_for_missing_subtitles(setup, monkeypatch, nom):
    _fake_api(monkeypatch, SANS_SOUS_TITRES[nom])
    out = main._summarize(URL, api_key="sk-or-client", provider="openrouter", transcription_llm=True)
    assert setup == [URL] and out["transcription"] == "llm-video"


def test_summarize_plain_transcript_error_no_fallback(setup, monkeypatch):
    def bad(u):
        raise ValueError("Erreur transcript: vidéo privée")
    monkeypatch.setattr(main, "get_youtube_transcript", bad)
    with pytest.raises(ValueError, match="Whisper"):
        main._summarize(URL, api_key="sk-or-client", provider="openrouter", transcription_llm=True)
    assert setup == []
