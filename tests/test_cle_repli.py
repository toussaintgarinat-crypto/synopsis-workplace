"""M1 / M7 : quelle clé paie le repli de transcription Gemini."""
import pytest
from fastapi.testclient import TestClient

import main
from lib.extractor import SansSousTitres

URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
LLM_DATA = {"video_id": "jNQXAC9IVRw", "transcript": [{"text": "a", "start": 1.0, "duration": 4.0}],
            "language": "unknown", "title": "T", "total_duration_minutes": 0.1, "entries_count": 1,
            "source": "llm-video", "model": "google/gemini-2.5-flash"}


def _no_subs(*a, **k):
    raise SansSousTitres("Aucun transcript disponible pour cette vidéo")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.delenv("ALLOW_SERVER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-server")
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: "## 📍 Chapitrage Temporel\n| [00:01] | *S* | D |")
    calls = []

    def fake(src, api_key, model, timeout=None, provider="openrouter"):
        calls.append(api_key)
        return dict(LLM_DATA)
    monkeypatch.setattr(main, "transcribe_youtube_via_llm", fake)
    return calls


def post(body):
    return TestClient(main.app).post("/resumer", json=body)


# ---------- M1 : passerelle et consentement explicite ----------

def test_gateway_without_client_key_does_not_spend_server_key(env, monkeypatch):
    monkeypatch.setenv("GATEWAY_URL", "http://gateway.invalid")
    r = post({"url": URL, "provider": "openrouter", "transcription_llm": True})
    assert r.status_code == 400 and env == []


def test_gateway_with_client_key_uses_client_key(env, monkeypatch):
    monkeypatch.setenv("GATEWAY_URL", "http://gateway.invalid")
    r = post({"url": URL, "provider": "openrouter", "api_key": "sk-or-client", "transcription_llm": True})
    assert r.status_code == 200 and env == ["sk-or-client"]


def test_server_key_requires_explicit_transcription_llm(env):
    r = post({"url": URL, "provider": "openrouter"})
    assert r.status_code == 400 and env == []
    r = post({"url": URL, "provider": "openrouter", "transcription_llm": True})
    assert r.status_code == 200 and env == ["sk-or-server"]


def test_client_key_used_without_explicit_flag(env):
    r = post({"url": URL, "provider": "openrouter", "api_key": "sk-or-client"})
    assert r.status_code == 200 and env == ["sk-or-client"]


def test_lot_server_key_requires_explicit_flag(env):
    c = TestClient(main.app)
    r = c.post("/resumer-lot", json={"urls": [URL], "provider": "openrouter"})
    assert r.status_code == 200 and r.json()[0]["ok"] is False and env == []
    r = c.post("/resumer-lot", json={"urls": [URL], "provider": "openrouter", "transcription_llm": True})
    assert r.json()[0]["ok"] is True and env == ["sk-or-server"]


# ---------- M7 : mode léger ----------

@pytest.fixture
def light(env, monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    return env


def test_light_client_key_used_exactly(light):
    r = post({"url": URL, "provider": "openrouter", "api_key": "sk-or-client", "transcription_llm": True})
    assert r.status_code == 200 and light == ["sk-or-client"]


def test_light_allow_server_key_uses_server_key(light, monkeypatch):
    monkeypatch.setenv("ALLOW_SERVER_KEY", "1")
    r = post({"url": URL, "provider": "openrouter", "transcription_llm": True})
    assert r.status_code == 200 and light == ["sk-or-server"]


def test_light_no_key_refused_before_any_paid_call(light, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", lambda *a, **k: pytest.fail("aucun appel attendu"))
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: pytest.fail("aucun appel attendu"))
    r = post({"url": URL, "provider": "openrouter", "transcription_llm": True})
    assert r.status_code == 403 and "clé" in r.json()["detail"] and light == []


def test_rate_limit_applies_to_fallback_path(light, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    body = {"url": URL, "provider": "openrouter", "api_key": "sk-or-client", "transcription_llm": True}
    codes = [post(body).status_code for _ in range(3)]
    assert codes == [200, 200, 429] and light == ["sk-or-client", "sk-or-client"]
