import pytest
from fastapi.testclient import TestClient
import main

EVIL = {"Origin": "http://evil.example"}
OK = {"Origin": "http://localhost:6090"}


def client():
    return TestClient(main.app)


@pytest.mark.parametrize("path,body", [
    ("/export/obsidian", {"resultat": {"titre": "t"}, "vault_dir": "/tmp/x"}),
    ("/qa", {"titre": "t", "resume": "r", "question": "q"}),
    ("/export/markdown", {"titre": "t"}),
])
def test_evil_origin_blocked(path, body):
    r = client().post(path, json=body, headers=EVIL)
    assert r.status_code == 403


def test_evil_referer_blocked_without_origin():
    r = client().post("/export/markdown", json={"titre": "t"}, headers={"Referer": "http://evil.example/page"})
    assert r.status_code == 403


def test_same_origin_passes():
    r = client().post("/export/markdown", json={"titre": "t"}, headers=OK)
    assert r.status_code != 403


def test_same_host_origin_passes():
    r = client().post("/export/markdown", json={"titre": "t"}, headers={"Origin": "http://testserver"})
    assert r.status_code != 403


def test_no_origin_passes():
    assert client().post("/export/markdown", json={"titre": "t"}).status_code == 200


def test_get_with_evil_origin_not_blocked_by_middleware():
    assert client().get("/sante", headers=EVIL).status_code == 200


def test_cors_not_wildcard():
    r = client().options("/qa", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert "access-control-allow-origin" not in r.headers


def test_allowed_origins_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example")
    monkeypatch.setenv("PORT", "7000")
    o = main._allowed_origins()
    assert "https://a.example" in o and "http://localhost:7000" in o


def test_untrusted_host_rejected():
    r = TestClient(main.app, base_url="http://evil.example").get("/sante")
    assert r.status_code == 400


def test_allowed_hosts(monkeypatch):
    monkeypatch.setenv("ALLOWED_HOSTS", "synopsis")
    monkeypatch.setenv("VERCEL", "1")
    h = main._allowed_hosts()
    assert {"localhost", "127.0.0.1", "host.docker.internal", "synopsis", "*.vercel.app"} <= set(h)
