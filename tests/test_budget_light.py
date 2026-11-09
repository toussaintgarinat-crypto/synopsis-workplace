"""I2 : budget de temps partagé en mode léger (une fonction Vercel dispose de 60 s)."""
import httpx
import pytest
from fastapi.testclient import TestClient

import main
from lib import extractor, llm_client
from lib import video_transcriber as vt
from lib.extractor import SansSousTitres
from lib.llm_client import LLMTimeoutError

URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
LLM_DATA = {"video_id": "jNQXAC9IVRw", "transcript": [{"text": "a", "start": 1.0, "duration": 4.0}],
            "language": "unknown", "title": "T", "total_duration_minutes": 0.1, "entries_count": 1,
            "source": "llm-video", "model": "google/gemini-2.5-flash"}
SUBS = {"video_id": "jNQXAC9IVRw", "transcript": [{"text": "a", "start": 0.0, "duration": 1.0}], "language": "en",
        "title": "T", "total_duration_minutes": 0.1, "entries_count": 1}
CHAP = "## 📍 Chapitrage Temporel\n| [00:01] | *S* | D |"


class Horloge:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def horloge(monkeypatch):
    h = Horloge()
    monkeypatch.setattr(main, "_maintenant", h)  # time.monotonic de main
    return h


@pytest.fixture
def light(monkeypatch, horloge):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.delenv("LIGHT_BUDGET_S", raising=False)
    monkeypatch.delenv("LIGHT_RESERVE_RESUME_S", raising=False)
    return horloge


@pytest.fixture
def transcriber(monkeypatch):
    calls = []

    def fake(src, api_key, model, timeout=None, provider="openrouter"):
        calls.append(timeout)
        return dict(LLM_DATA)
    monkeypatch.setattr(main, "transcribe_youtube_via_llm", fake)
    return calls


@pytest.fixture
def llm(monkeypatch):
    calls = []

    def fake(prompt, model="", api_key="", provider="", **k):
        calls.append(k.get("timeout"))
        return CHAP
    monkeypatch.setattr(main, "llm_complete", fake)
    return calls


def _sans_sous_titres_apres(horloge, secondes):
    def f(*a, **k):
        horloge.t += secondes
        raise SansSousTitres("Aucun transcript disponible pour cette vidéo")
    return f


# ---------- repli Gemini ----------

def test_gemini_timeout_is_capped_by_remaining_budget(light, transcriber, llm, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", _sans_sous_titres_apres(light, 10))
    main._summarize(URL, api_key="sk-or-c", provider="openrouter", transcription_llm=True)
    assert transcriber == [pytest.approx(55 - 10 - 20)]


def test_gemini_timeout_never_above_50(light, transcriber, llm, monkeypatch):
    monkeypatch.setenv("LIGHT_BUDGET_S", "200")
    monkeypatch.setattr(main, "get_youtube_transcript", _sans_sous_titres_apres(light, 0))
    main._summarize(URL, api_key="sk-or-c", provider="openrouter", transcription_llm=True)
    assert transcriber == [50]


def test_gemini_refused_before_paid_call_when_budget_low(light, transcriber, llm, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", _sans_sous_titres_apres(light, 26))  # reste 9 s < 10
    with pytest.raises(ValueError) as e:
        main._summarize(URL, api_key="sk-or-c", provider="openrouter", transcription_llm=True)
    assert str(e.value) == vt.MSG_TIMEOUT_LIGHT
    assert transcriber == [] and llm == []


def test_reserve_configurable(light, transcriber, llm, monkeypatch):
    monkeypatch.setenv("LIGHT_RESERVE_RESUME_S", "5")
    monkeypatch.setattr(main, "get_youtube_transcript", _sans_sous_titres_apres(light, 30))
    main._summarize(URL, api_key="sk-or-c", provider="openrouter", transcription_llm=True)
    assert transcriber == [pytest.approx(20)]


def test_full_mode_no_budget(monkeypatch, horloge, transcriber, llm):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.setattr(main, "get_youtube_transcript", _sans_sous_titres_apres(horloge, 500))
    main._summarize(URL, api_key="sk-or-c", provider="openrouter", transcription_llm=True)
    assert transcriber == [None] and llm == [None]


def test_light_message_agrees_with_limits():
    # Le message ne prétend plus que l'appel est coupé à 60 s : c'est la limite de toute la requête.
    assert "(60 s)" not in vt.MSG_TIMEOUT_LIGHT and "60 s" in vt.MSG_TIMEOUT_LIGHT


def test_video_title_short_timeout_in_light_mode(monkeypatch):
    seen = []

    def fake(url, timeout=None):
        seen.append(timeout)
        raise httpx.ConnectError("x")
    monkeypatch.setattr(extractor.httpx, "get", fake)
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    extractor.get_video_title("jNQXAC9IVRw")
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    extractor.get_video_title("jNQXAC9IVRw")
    assert seen == [3, 10]


# ---------- appels de résumé ----------

def test_summary_calls_get_remaining_budget(light, llm, monkeypatch):
    def subs(*a, **k):
        light.t += 12
        return dict(SUBS)
    monkeypatch.setattr(main, "get_youtube_transcript", subs)
    main._summarize(URL, api_key="sk-or-c", provider="openrouter")
    assert llm == [pytest.approx(55 - 12)]


def test_fusion_gets_what_is_left(light, monkeypatch):
    seen = []

    def fake(prompt, model="", api_key="", provider="", **k):
        seen.append(k.get("timeout"))
        light.t += 10
        return CHAP
    monkeypatch.setattr(main, "llm_complete", fake)
    monkeypatch.setattr(main, "chunk_transcript", lambda *a, **k: [{"text": "a", "tokens": 1}, {"text": "b", "tokens": 1}])
    main._summarize_data(dict(SUBS), api_key="k", provider="openrouter")
    assert seen == [pytest.approx(55), pytest.approx(45), pytest.approx(35)]


def test_summary_refused_when_budget_exhausted(light, monkeypatch):
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: pytest.fail("aucun appel attendu"))
    with pytest.raises(LLMTimeoutError, match="Délai dépassé"):
        main._summarize_data(dict(SUBS), api_key="k", provider="openrouter", deadline=light.t + 0.5)


def test_chunk_timeout_not_swallowed(light, monkeypatch):
    def boom(*a, **k):
        raise LLMTimeoutError(llm_client.MSG_DELAI)
    monkeypatch.setattr(main, "llm_complete", boom)
    with pytest.raises(LLMTimeoutError):
        main._summarize_data(dict(SUBS), api_key="k", provider="openrouter")


def test_resumer_route_timeout_is_clean_504(light, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", lambda *a, **k: dict(SUBS))

    def boom(*a, **k):
        raise LLMTimeoutError(llm_client.MSG_DELAI)
    monkeypatch.setattr(main, "llm_complete", boom)
    r = TestClient(main.app).post("/resumer", json={"url": URL, "api_key": "sk-or-c", "provider": "openrouter"})
    assert r.status_code == 504 and r.json()["detail"].startswith("Délai dépassé")


def test_qa_light_mode_has_budget(light, monkeypatch):
    seen = []
    monkeypatch.setattr(main, "llm_complete", lambda p, model="", **k: seen.append(k.get("timeout")) or "rep")
    r = TestClient(main.app).post("/qa", json={"titre": "t", "resume": "r", "question": "q", "api_key": "k"})
    assert r.status_code == 200 and seen == [pytest.approx(55)]

    def boom(*a, **k):
        raise LLMTimeoutError(llm_client.MSG_DELAI)
    monkeypatch.setattr(main, "llm_complete", boom)
    r = TestClient(main.app).post("/qa", json={"titre": "t", "resume": "r", "question": "q", "api_key": "k"})
    assert r.status_code == 504


# ---------- /resumer-lot : échéance partagée ----------

def test_lot_shares_one_deadline_in_light_mode(light, monkeypatch):
    seen = []
    monkeypatch.setattr(main, "_summarize", lambda u, *a, **k: seen.append(k.get("deadline")) or {"titre": "T"})
    r = TestClient(main.app).post("/resumer-lot", json={"urls": [URL, URL, URL], "api_key": "k"})
    assert r.status_code == 200
    assert len(seen) == 3 and seen[0] == pytest.approx(light.t + 55) and len(set(seen)) == 1


def test_lot_later_videos_fail_cleanly_after_deadline(light, monkeypatch):
    def subs(*a, **k):
        light.t += 30
        return dict(SUBS)
    monkeypatch.setattr(main, "get_youtube_transcript", subs)
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: CHAP)
    r = TestClient(main.app).post("/resumer-lot", json={"urls": [URL, URL], "api_key": "k"})
    j = r.json()
    assert j[0]["ok"] is True and j[1]["ok"] is False and "Délai dépassé" in j[1]["erreur"]


# ---------- llm_client : paramètre timeout ----------

@pytest.fixture
def post(monkeypatch):
    calls = []
    state = {}

    def fake(url, json=None, headers=None, timeout=None):
        calls.append(timeout)
        if state.get("exc"):
            raise state["exc"]
        return state["rsp"]
    monkeypatch.setattr(llm_client.httpx, "post", fake)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    return calls, state


class Rsp:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def test_default_timeout_unchanged(post):
    calls, state = post
    state["rsp"] = Rsp()
    assert llm_client._complete_openai_compatible("http://x", "p", "m") == "ok"
    assert calls == [180]


def test_default_mode_still_retries_on_timeout(post):
    calls, state = post
    state["exc"] = httpx.ReadTimeout("t")
    with pytest.raises(RuntimeError):
        llm_client._complete_openai_compatible("http://x", "p", "m")
    assert calls == [180, 180, 180]


def test_explicit_timeout_used_and_no_retry_on_timeout(post):
    calls, state = post
    state["exc"] = httpx.ReadTimeout("t")
    with pytest.raises(LLMTimeoutError) as e:
        llm_client._complete_openai_compatible("http://x", "p", "m", timeout=25)
    assert calls == [pytest.approx(25, abs=0.5)]
    assert str(e.value).startswith("Délai dépassé")


def test_explicit_timeout_capped_at_default(post):
    calls, state = post
    state["rsp"] = Rsp()
    llm_client._complete_openai_compatible("http://x", "p", "m", timeout=999)
    assert calls == [180]


def test_llm_complete_forwards_timeout(post, monkeypatch):
    calls, state = post
    state["rsp"] = Rsp()
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    llm_client.llm_complete("p", model="m", api_key="sk-x", provider="openrouter", timeout=12)
    assert calls == [pytest.approx(12, abs=0.5)]


def test_rate_limit_retry_does_not_outlive_budget(post):
    calls, state = post

    class R429(Rsp):
        status_code = 429
    state["rsp"] = R429()
    with pytest.raises(RuntimeError):
        llm_client._complete_openai_compatible("http://x", "p", "m", timeout=3)
    assert len(calls) == 1  # pas de pause de 5 s au-delà du budget


# ---------- sous-titres : moins d'allers-retours ----------

def test_subtitles_tried_in_one_call_with_language_priority(monkeypatch):
    from youtube_transcript_api import _errors as yte
    calls = []

    class FakeApi:
        def fetch(self, video_id, languages=("en",)):
            calls.append(list(languages))
            raise yte.NoTranscriptFound(video_id, languages, "aucune")
    monkeypatch.setattr(extractor, "YouTubeTranscriptApi", FakeApi)
    with pytest.raises(SansSousTitres):
        extractor.get_youtube_transcript(URL)
    assert calls[0] == ["fr", "en", "es", "de", "it", "pt"] and len(calls) <= 2


def test_subtitles_language_reported(monkeypatch):
    class Entry:
        text, start, duration = "a", 0.0, 1.0

    class Fetched(list):
        language_code = "en"

    class FakeApi:
        def fetch(self, video_id, languages=("en",)):
            return Fetched([Entry()])
    monkeypatch.setattr(extractor, "YouTubeTranscriptApi", FakeApi)
    monkeypatch.setattr(extractor, "get_video_title", lambda vid, timeout=None: "T")
    out = extractor.get_youtube_transcript(URL)
    assert out["language"] == "en" and out["entries_count"] == 1
