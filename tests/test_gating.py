import httpx
import pytest
from fastapi.testclient import TestClient
import main
from lib import llm_client

R = {"titre": "T", "resume": "R", "chapitres": [], "insights": []}


def client():
    return TestClient(main.app)


def _fake_summary(*a, **k):
    return {"titre": "T", "resume": "R", "chapitres": [], "insights": [], "langue_source": "fr", "avertissements": []}


HEAVY_ROUTES = [
    ("post", "/export/obsidian", {"json": {"resultat": R, "vault_dir": "/tmp/zz"}}),
    ("post", "/export/excalidraw", {"json": R}),
    ("post", "/export/drive", {"json": {"resultat": R}}),
    ("post", "/image", {"json": {"resultat": R, "api_key": "k"}}),
    ("post", "/reel", {"json": {"url": "u", "api_key": "k"}}),
    ("post", "/reel", {"json": {"url": "u", "api_key": "k", "narration": True}}),
    ("post", "/resumer-fichier", {"files": {"fichier": ("a.mp3", b"x")}, "data": {"api_key": "k"}}),
    ("post", "/resumer", {"json": {"url": "u", "provider": "local", "api_key": "k"}}),
    ("post", "/resumer", {"json": {"url": "u", "provider": "custom", "api_key": "k"}}),
    ("post", "/qa", {"json": {"titre": "t", "resume": "r", "question": "q", "provider": "local", "api_key": "k"}}),
    ("post", "/resumer-lot", {"json": {"urls": ["u"], "provider": "local", "api_key": "k"}}),
    ("post", "/modeles", {"json": {"provider": "local"}}),
    ("get", "/modeles?provider=local", {}),
    ("get", "/modeles?provider=custom", {}),
]


@pytest.mark.parametrize("method,path,kw", HEAVY_ROUTES)
def test_light_mode_blocks_heavy_routes(monkeypatch, method, path, kw):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: pytest.fail("ne doit pas être appelé"))
    r = getattr(client(), method)(path, **kw)
    assert r.status_code == 403


def test_local_provider_allowed_in_full(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setattr(main, "_summarize", _fake_summary)
    r = client().post("/resumer", json={"url": "u", "provider": "local"})
    assert r.status_code == 200


def test_custom_provider_needs_env_in_full(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("CUSTOM_LLM_URL", raising=False)
    assert client().post("/resumer", json={"url": "u", "provider": "custom"}).status_code == 403
    assert client().get("/modeles?provider=custom").status_code == 403
    monkeypatch.setenv("CUSTOM_LLM_URL", "http://llm.example/v1")
    monkeypatch.setattr(main, "_summarize", _fake_summary)
    assert client().post("/resumer", json={"url": "u", "provider": "custom"}).status_code == 200


def test_custom_models_from_env_url(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setenv("CUSTOM_LLM_URL", "http://llm.example/v1/")
    seen = {}

    class Rsp:
        status_code = 200
        def json(self): return {"data": [{"id": "m2"}, {"id": "m1"}]}

    def get(url, **k):
        seen["url"] = url
        return Rsp()
    monkeypatch.setattr(httpx, "get", get)
    r = client().get("/modeles?provider=custom")
    assert r.json() == {"models": ["m1", "m2"], "source": "custom"}
    assert seen["url"] == "http://llm.example/v1/models"


def test_custom_models_unreachable_is_empty(monkeypatch):
    monkeypatch.setenv("CUSTOM_LLM_URL", "http://llm.example/v1")
    def boom(*a, **k): raise httpx.ConnectError("x")
    monkeypatch.setattr(httpx, "get", boom)
    assert llm_client.list_custom_models() == []


@pytest.mark.parametrize("method,path,body", [
    ("post", "/resumer", {"url": "u"}),
    ("post", "/resumer-lot", {"urls": ["u"]}),
    ("post", "/qa", {"titre": "t", "resume": "r", "question": "q"}),
])
def test_light_mode_refuses_server_key(monkeypatch, method, path, body):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.delenv("ALLOW_SERVER_KEY", raising=False)
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: pytest.fail("appelé"))
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: pytest.fail("appelé"))
    r = client().post(path, json=body)
    assert r.status_code == 403 and "clé" in r.json()["detail"]


def test_light_mode_with_own_key_or_allow_flag(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setattr(main, "_summarize", _fake_summary)
    monkeypatch.delenv("ALLOW_SERVER_KEY", raising=False)
    assert client().post("/resumer", json={"url": "u", "api_key": "sk"}).status_code == 200
    assert client().post("/resumer", json={"url": "u"}).status_code == 403
    monkeypatch.setenv("ALLOW_SERVER_KEY", "1")
    assert client().post("/resumer", json={"url": "u"}).status_code == 200


def test_full_mode_does_not_require_client_key(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setattr(main, "_summarize", _fake_summary)
    assert client().post("/resumer", json={"url": "u"}).status_code == 200


def test_batch_cap_light(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setattr(main, "_summarize", _fake_summary)
    ok = client().post("/resumer-lot", json={"urls": ["a", "b", "c"], "api_key": "k"})
    assert ok.status_code == 200 and len(ok.json()) == 3
    r = client().post("/resumer-lot", json={"urls": ["a", "b", "c", "d"], "api_key": "k"})
    assert r.status_code == 400 and "3" in r.json()["detail"]


def test_batch_cap_full_and_env(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setattr(main, "_summarize", _fake_summary)
    assert main._max_batch() == 20
    monkeypatch.setenv("MAX_BATCH_FULL", "2")
    assert client().post("/resumer-lot", json={"urls": ["a", "b", "c"]}).status_code == 400
    assert client().post("/resumer-lot", json={"urls": ["a", "b", " "]}).status_code == 200


def test_qa_route(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setattr(main, "llm_complete", lambda p, **k: "reponse")
    r = client().post("/qa", json={"titre": "t", "resume": "r", "question": "q"})
    assert r.status_code == 200 and r.json() == {"reponse": "reponse"}


def test_i18n_unknown_lang_404():
    assert client().get("/i18n/xx").status_code == 404
    assert client().get("/i18n/fr").status_code == 200


def test_modeles_fetches_live_list(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    seen = {}

    class Rsp:
        status_code = 200
        def json(self): return {"data": [{"id": "z/b"}, {"id": "a/a"}]}

    def get(url, headers=None, timeout=0):
        seen["h"] = headers
        return Rsp()
    monkeypatch.setattr(httpx, "get", get)
    r = client().get("/modeles?provider=openrouter")
    assert r.json() == {"models": ["a/a", "z/b"], "source": "api"}
    r = client().post("/modeles", json={"provider": "openrouter", "api_key": "sk-secret"})
    assert r.json()["source"] == "api" and seen["h"] == {"Authorization": "Bearer sk-secret"}


def test_modeles_get_ignores_query_key(monkeypatch):
    seen = {}

    class Rsp:
        status_code = 200
        def json(self): return {"data": [{"id": "m"}]}

    monkeypatch.setattr(httpx, "get", lambda url, headers=None, timeout=0: seen.update(h=headers) or Rsp())
    client().get("/modeles?provider=openrouter&api_key=sk-leak")
    assert seen["h"] == {}


def test_modeles_fallback_on_error(monkeypatch):
    def boom(*a, **k): raise httpx.ConnectError("x")
    monkeypatch.setattr(httpx, "get", boom)
    r = client().get("/modeles?provider=openai")
    assert r.json()["source"] == "fallback"


def test_resumer_fichier_upload_limit(monkeypatch, tmp_path):
    caps = {"whisper": {"ok": True, "raison": ""}, "llm_local": {"ok": True, "raison": ""}}
    monkeypatch.setattr(main, "get_capabilities", lambda: caps)
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")
    import tempfile, os
    created = []
    real = tempfile.mkstemp
    def spy(*a, **k):
        fd, p = real(*a, **k)
        created.append(p)
        return fd, p
    monkeypatch.setattr(tempfile, "mkstemp", spy)
    r = client().post("/resumer-fichier", files={"fichier": ("a.mp3", b"x" * 5000)}, data={"api_key": "k"})
    assert r.status_code == 413
    assert created and not os.path.exists(created[0])


def test_resumer_fichier_small_ok(monkeypatch):
    caps = {"whisper": {"ok": True, "raison": ""}}
    monkeypatch.setattr(main, "get_capabilities", lambda: caps)
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    import lib.whisper_transcriber as w
    monkeypatch.setattr(w, "transcribe_file", lambda p, l: [{"text": "x", "start": 0, "duration": 1}])
    monkeypatch.setattr(main, "_summarize_data", lambda *a, **k: {"titre": "a.mp3"})
    r = client().post("/resumer-fichier", files={"fichier": ("a.mp3", b"x" * 100)}, data={"api_key": "k"})
    assert r.status_code == 200


def test_resumer_fichier_calls_capabilities_once(monkeypatch):
    n = {"c": 0}
    def caps():
        n["c"] += 1
        return {"whisper": {"ok": False, "raison": "non"}}
    monkeypatch.setattr(main, "get_capabilities", caps)
    r = client().post("/resumer-fichier", files={"fichier": ("a.mp3", b"x")})
    assert r.status_code == 403 and n["c"] == 1


def test_provider_local_routing(monkeypatch):
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    seen = {}

    class Rsp:
        status_code = 200
        def json(self): return {"choices": [{"message": {"content": "ok"}}]}

    def post(url, **k):
        seen["url"] = url
        return Rsp()
    monkeypatch.setattr(llm_client.httpx, "post", post)
    assert llm_client.llm_complete("p", provider="local") == "ok"
    assert seen["url"] == f"{llm_client.LOCAL_LLM_URL}/v1/chat/completions"


def test_provider_custom_routing(monkeypatch):
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.setenv("CUSTOM_LLM_URL", "http://llm.example/v1/")
    seen = {}

    class Rsp:
        status_code = 200
        def json(self): return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(llm_client.httpx, "post", lambda url, **k: seen.update(url=url, h=k["headers"]) or Rsp())
    assert llm_client.llm_complete("p", provider="custom", api_key="sk") == "ok"
    assert seen["url"] == "http://llm.example/v1/chat/completions"
    monkeypatch.delenv("CUSTOM_LLM_URL")
    with pytest.raises(RuntimeError):
        llm_client.llm_complete("p", provider="custom", api_key="sk")


def test_summarize_chapterless_warns(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setattr(main, "llm_complete", lambda p, **k: "du texte sans chapitres")
    data = {"transcript": [{"text": "x", "start": 0, "duration": 1}], "title": "t", "language": "fr"}
    out = main._summarize_data(data)
    assert out["chapitres"] == [] and out["avertissements"] and "chapitre" in out["avertissements"][0]
