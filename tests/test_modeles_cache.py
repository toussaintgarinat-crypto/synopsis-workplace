"""I3 / M7 : cache de /modeles borné, clé partagée pour les listes publiques, limite de débit."""
import httpx
import pytest
from fastapi.testclient import TestClient

import main
from lib import models_catalog as mc
from lib import security


class Rsp:
    status_code = 200

    def json(self):
        return {"data": [{"id": "a/a"}, {"id": "google/gemini-2.5-flash"}]}


@pytest.fixture
def get(monkeypatch):
    calls = []

    def fake(url, headers=None, timeout=0):
        calls.append(url)
        return Rsp()
    monkeypatch.setattr(httpx, "get", fake)
    return calls


def test_public_openrouter_list_shared_between_keys(get):
    for i in range(5):
        mc.fetch_models("openrouter", "resume", f"sk-or-{i}")
        mc.fetch_models("openrouter", "image", f"sk-or-{i}")
    assert len(get) == 2 and len(mc._cache) == 2


def test_openai_list_stays_per_key(get):
    mc.fetch_models("openai", "resume", "sk-a")
    mc.fetch_models("openai", "resume", "sk-b")
    mc.fetch_models("openai", "resume", "sk-a")
    assert len(get) == 2


def test_cache_bounded_with_many_keys(get):
    for i in range(mc.CACHE_MAX + 50):
        mc.fetch_models("openai", "resume", f"sk-{i}")
    assert len(mc._cache) <= mc.CACHE_MAX


def test_cache_purges_expired_first_then_oldest(get, monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(mc.time, "monotonic", lambda: t["now"])
    monkeypatch.setattr(mc, "CACHE_MAX", 3)
    mc.fetch_models("openai", "resume", "sk-old")
    t["now"] += mc.CACHE_TTL + 1
    for k in ("sk-1", "sk-2", "sk-3"):
        t["now"] += 1
        mc.fetch_models("openai", "resume", k)
    cles = list(mc._cache)
    assert len(cles) == 3 and mc._cle_cache("openai", "resume", "sk-old") not in cles
    t["now"] += 1
    mc.fetch_models("openai", "resume", "sk-4")  # plein, rien d'expiré : la plus ancienne part
    assert mc._cle_cache("openai", "resume", "sk-1") not in mc._cache
    assert mc._cle_cache("openai", "resume", "sk-4") in mc._cache and len(mc._cache) == 3


def test_modeles_route_bounded(get, monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setattr(mc, "CACHE_MAX", 10)
    c = TestClient(main.app)
    for i in range(30):
        assert c.post("/modeles", json={"provider": "openai", "api_key": f"sk-{i}", "tache": "resume"}).status_code == 200
    assert len(mc._cache) <= 10


def test_modeles_post_rate_limited(get, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    c = TestClient(main.app)
    body = {"provider": "openrouter", "api_key": "sk-or-x", "tache": "resume"}
    limite = security.limite_modeles(2)
    codes = [c.post("/modeles", json=body).status_code for _ in range(limite + 1)]
    assert codes[:-1] == [200] * limite and codes[-1] == 429
    assert "/modeles" in security.RATE_LIMITED_PATHS


def test_modeles_bucket_separate_from_llm_routes(get, monkeypatch):
    """Le front charge 6 listes d'un coup : elles ne doivent pas consommer le quota de /resumer."""
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: {"titre": "T"})
    c = TestClient(main.app)
    for _ in range(security.limite_modeles(2) + 1):
        c.post("/modeles", json={"provider": "openrouter", "tache": "resume"})
    assert c.post("/resumer", json={"url": "https://youtu.be/jNQXAC9IVRw", "api_key": "k"}).status_code == 200


def test_modeles_not_limited_in_full_mode_by_default(get, monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("RATE_LIMIT_PER_MIN", raising=False)
    c = TestClient(main.app)
    assert all(c.post("/modeles", json={"provider": "openrouter"}).status_code == 200 for _ in range(30))
