import base64
import json
import pytest
import httpx
from lib.image_generator import build_image_prompt, generate_image
from lib.exporters import drive

D = {"titre": "Titre", "resume": "R", "chapitres": [{"subject": "A"}, {"subject": "B"}]}


class Resp:
    def __init__(self, status=200, data=None, content=b""):
        self.status_code, self._d, self.content, self.text = status, data or {}, content, json.dumps(data or {})
    def json(self):
        return self._d


def test_prompt_uses_title_and_chapters():
    p = build_image_prompt(D)
    assert "Titre" in p and "A" in p and "B" in p


def test_generate_image_decodes_base64(monkeypatch):
    png = b"\x89PNG-fake"
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    body = {"choices": [{"message": {"images": [{"type": "image_url", "image_url": {"url": url}}]}}]}
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp(200, body))
    assert generate_image("p", "k", "black-forest-labs/flux.2-klein-4b") == png


def test_generate_image_errors(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp(401))
    with pytest.raises(RuntimeError):
        generate_image("p", "bad", "")
    with pytest.raises(RuntimeError):
        generate_image("p", "", "")


def test_upload_markdown(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp(200, {"files": [{"id": "F1"}]}))
    def post(url, **k):
        calls.append((url, k))
        return Resp(200, {"id": "XYZ"})
    monkeypatch.setattr(httpx, "post", post)
    link = drive.upload_markdown(D, json.dumps({"access_token": "tok"}))
    assert link == "https://drive.google.com/file/d/XYZ/view"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer tok"


def test_upload_markdown_bad_credentials():
    with pytest.raises(ValueError):
        drive.upload_markdown(D, "pas du json")
    with pytest.raises(ValueError):
        drive.upload_markdown(D, "{}")


def test_load_credentials(tmp_path, monkeypatch):
    f = tmp_path / "c.json"
    f.write_text('{"access_token": "t"}')
    assert drive.load_credentials(str(f)) == '{"access_token": "t"}'
    assert drive.load_credentials('{"a": 1}') == '{"a": 1}'
    with pytest.raises(ValueError):
        drive.load_credentials("")


def test_routes_refuse_when_capability_off(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    c = TestClient(main.app)
    assert c.post("/export/drive", json={"resultat": D}).status_code == 403
    assert c.post("/image", json={"resultat": D}).status_code == 403


def test_image_route_needs_key(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = TestClient(main.app).post("/image", json={"resultat": D})
    assert r.status_code == 400


@pytest.mark.parametrize("model", ["p-image", "black-forest-labs/flux-schnell", "stable-diffusion-v3-5-large"])
def test_openrouter_key_only_sent_to_openrouter(monkeypatch, model):
    hosts = []
    def post(url, *a, **k):
        hosts.append(url)
        return Resp(401)
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("unexpected GET"))
    with pytest.raises(RuntimeError):
        generate_image("p", "sk-or-secret", model)
    assert hosts and all(h.startswith("https://openrouter.ai/") for h in hosts)
