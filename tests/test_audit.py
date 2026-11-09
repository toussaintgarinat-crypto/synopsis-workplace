"""Tests de l'audit de sécurité (voir .superpowers/sdd/security-audit-report.md)."""
import logging
import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from lib import image_generator as ig
from lib.exporters import drive
from lib.exporters.obsidian import save_note
from lib.extractor import extract_youtube_id

R = {"titre": "T", "resume": "R", "chapitres": [], "insights": []}


def client(**kw):
    return TestClient(main.app, **kw)


def _all_caps():
    return {k: {"ok": True, "raison": ""} for k in
            ("resume", "whisper", "reel", "llm_local", "llm_custom", "obsidian", "drive", "excalidraw", "images", "tts")}


# ---------- a. contrôle d'origine ----------

@pytest.mark.parametrize("headers", [
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
    {"Origin": "http://testserver/evil"},
    {"Origin": "http://localhost:6090/evil"},
    {"Origin": "null"},
    {"Origin": "http://localhost:6090.evil.com"},
    {"Origin": "http://evil.com#@localhost:6090"},
    {"Origin": "http://evil.com@localhost:6090"},
    {"Referer": "http://evil.com@localhost:6090/x"},
    {"Origin": "http://localhost:6090", "Sec-Fetch-Site": "cross-site"},
])
def test_origin_bypasses_blocked(headers):
    assert client().post("/export/markdown", json=R, headers=headers).status_code == 403


def test_sec_fetch_same_origin_passes():
    r = client().post("/export/markdown", json=R, headers={"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"})
    assert r.status_code == 200


def test_sec_fetch_none_passes():
    # Navigation initiée par l'utilisateur (barre d'adresse) : pas de CSRF.
    assert client().post("/export/markdown", json=R, headers={"Sec-Fetch-Site": "none"}).status_code == 200


def test_cross_site_multipart_upload_blocked():
    r = client().post("/resumer-fichier", files={"fichier": ("a.mp3", b"x")}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_same_host_shortcut_needs_trusted_host():
    r = TestClient(main.app, base_url="http://evil.example").post("/export/markdown", json=R, headers={"Origin": "http://evil.example"})
    assert r.status_code == 400


# ---------- b. Obsidian ----------

@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.delenv("OBSIDIAN_ALLOWED_ROOTS", raising=False)
    return h


def test_vault_inside_home_ok(home):
    p = save_note(R, str(home / "Vault"))
    assert Path(p).parent == (home / "Vault").resolve()


def test_vault_tilde_ok(home):
    p = save_note(R, "~/Vault")
    assert Path(p).parent == (home / "Vault").resolve()


@pytest.mark.parametrize("v", ["/etc", "/", "/tmp", "~/.ssh", "~/.claude", "~/Vault/.git", "~/Library/LaunchAgents", "relatif/vault", "~/../.."])
def test_vault_dangerous_refused(home, v):
    with pytest.raises(ValueError):
        save_note(R, v)


def test_vault_icloud_obsidian_ok(home):
    p = save_note(R, str(home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "V"))
    assert p.endswith("T.md")


def test_vault_symlink_escape_refused(home, tmp_path):
    dehors = tmp_path / "dehors"
    dehors.mkdir()
    (home / "lien").symlink_to(dehors)
    with pytest.raises(ValueError):
        save_note(R, str(home / "lien"))
    (home / "lien_cache").symlink_to(home / ".ssh", target_is_directory=True)
    with pytest.raises(ValueError):
        save_note(R, str(home / "lien_cache"))


def test_vault_never_overwrites_foreign_file(home):
    v = home / "Vault"
    v.mkdir()
    (v / "T.md").write_text("NOTE PERSONNELLE")
    with pytest.raises(ValueError):
        save_note(R, str(v))
    assert (v / "T.md").read_text() == "NOTE PERSONNELLE"


def test_vault_overwrites_own_note(home):
    v = home / "Vault"
    save_note(R, str(v))
    p = save_note(dict(R, resume="NOUVEAU"), str(v))
    assert "NOUVEAU" in Path(p).read_text()


def test_vault_allowed_roots_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_ALLOWED_ROOTS", str(tmp_path / "racine"))
    assert save_note(R, str(tmp_path / "racine" / "v")).endswith("T.md")
    with pytest.raises(ValueError):
        save_note(R, str(tmp_path / "ailleurs"))


def test_obsidian_route_refuses_etc(home):
    r = client().post("/export/obsidian", json={"resultat": R, "vault_dir": "/etc"})
    assert r.status_code == 400


# ---------- c / g. secrets et erreurs ----------

SECRET = "sk-or-v1-abcdef1234567890abcdef1234567890"


class Up:
    status_code = 500
    text = 'upstream {"error":"bad key ' + SECRET + '", "auth": "Bearer ya29.A0AfH6SMBxxxxxxxxxxxxxxx"}'
    headers = {}
    def json(self):
        return {}


def test_image_error_scrubbed(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Up())
    r = client().post("/image", json={"resultat": R, "api_key": SECRET})
    assert r.status_code == 502
    assert SECRET not in r.text and "ya29." not in r.text and "Bearer ya29" not in r.text


def test_llm_error_scrubbed(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Up())
    r = client().post("/qa", json={"titre": "t", "resume": "r", "question": "q", "api_key": SECRET, "provider": "openrouter"})
    assert r.status_code == 500
    assert SECRET not in r.text and "sk-or-v1-abc" not in r.text


def test_error_detail_truncated(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x" * 5000)
    monkeypatch.setattr(main, "_summarize", boom)
    r = client().post("/resumer", json={"url": "u", "api_key": "k"})
    assert r.status_code == 500 and len(r.json()["detail"]) <= 400


def test_server_env_secret_scrubbed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "zz-server-secret-value-123")
    def boom(*a, **k):
        raise RuntimeError("fuite zz-server-secret-value-123")
    monkeypatch.setattr(main, "_summarize", boom)
    r = client().post("/resumer", json={"url": "u", "api_key": "k"})
    assert "zz-server-secret-value-123" not in r.text


def test_drive_error_scrubbed(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS", '{"access_token": "ya29.SECRETTOKENxxxxxxxxxxxx"}')
    def boom(*a, **k):
        raise RuntimeError("Drive 500: token ya29.SECRETTOKENxxxxxxxxxxxx")
    monkeypatch.setattr(drive, "upload_to_drive", boom)
    r = client().post("/export/drive", json={"resultat": R})
    assert r.status_code == 502 and "SECRETTOKEN" not in r.text


def test_drive_ignores_request_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS", '{"access_token": "env-token"}')
    seen = []
    monkeypatch.setattr(drive, "upload_to_drive", lambda path, name, mime, token, *a, **k: seen.append(token) or {"success": True, "web_view_link": "https://drive/x"})
    r = client().post("/export/drive", json={"resultat": R, "credentials": '{"access_token": "attaquant"}', "access_token": "attaquant"})
    assert r.status_code == 200 and seen == ["env-token"]


def test_logs_scrubbed(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("synopsis").error("Chunk failed: HTTP 401 %s", SECRET)
    assert SECRET not in caplog.text


def test_dead_drive_oauth_helpers_removed():
    for f in ("get_google_auth_url", "exchange_code_for_token", "check_drive_configured"):
        assert not hasattr(drive, f)


# ---------- d. validation et taille ----------

@pytest.mark.parametrize("route", ["/export/markdown", "/export/excalidraw"])
@pytest.mark.parametrize("bad", [
    {"titre": "t", "chapitres": ["x"]},
    {"titre": "t", "insights": [1]},
    {"titre": "t", "chapitres": "abc"},
    {"titre": "t", "insights": {"a": 1}},
    {"titre": ["x"]},
    {"titre": "t", "chapitres": [{"subject": "a"}] * 2001},
])
def test_export_malformed_is_4xx(route, bad):
    r = client(raise_server_exceptions=False).post(route, json=bad)
    assert 400 <= r.status_code < 500


def test_obsidian_malformed_is_4xx(home):
    r = client(raise_server_exceptions=False).post("/export/obsidian", json={"resultat": {"titre": "t", "chapitres": [1]}, "vault_dir": str(home / "V")})
    assert r.status_code == 400


def test_export_valid_still_ok():
    d = {"titre": "T", "resume": "R", "langue_source": "fr", "avertissements": ["w"],
         "chapitres": [{"timestamp": "00:10", "ts_seconds": 10, "subject": "A", "description": "d"}],
         "insights": [{"timestamp": "00:10", "title": "I", "description": "d"}]}
    assert client().post("/export/markdown", json=d).status_code == 200
    assert client().post("/export/excalidraw", json=d).status_code == 200


def test_body_size_cap(monkeypatch):
    monkeypatch.setenv("MAX_BODY_MB", "1")
    r = client().post("/export/markdown", json={"titre": "t", "resume": "A" * (2 * 1024 * 1024)})
    assert r.status_code == 413


def test_body_size_cap_chunked(monkeypatch):
    monkeypatch.setenv("MAX_BODY_MB", "1")
    def gen():
        yield b'{"titre": "t", "resume": "'
        for _ in range(3):
            yield b"A" * (1024 * 1024)
        yield b'"}'
    r = client().post("/export/markdown", content=gen(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_upload_governed_by_upload_cap(monkeypatch):
    monkeypatch.setenv("MAX_BODY_MB", "1")
    monkeypatch.setenv("MAX_UPLOAD_MB", "3")
    monkeypatch.setattr(main, "get_capabilities", _all_caps)
    import lib.whisper_transcriber as wt
    monkeypatch.setattr(wt, "transcribe_file_with_model", lambda *a, **k: ([{"text": "x", "start": 0.0, "duration": 1.0}], "base"))
    monkeypatch.setattr(main, "_summarize_data", lambda *a, **k: R)
    ok = client().post("/resumer-fichier", files={"fichier": ("a.mp3", b"x" * (2 * 1024 * 1024))}, data={"api_key": "k"})
    assert ok.status_code == 200
    big = client().post("/resumer-fichier", files={"fichier": ("a.mp3", b"x" * (5 * 1024 * 1024))}, data={"api_key": "k"})
    assert big.status_code == 413


# ---------- e. export HTML autonome ----------

def _front():
    html = main.index()
    return html[html.index("<script>") + 8: html.index("</script>")]


def test_exported_document_has_csp():
    js = _front()
    fn = js[js.index("function documentExport"): js.index("function exporterHtml")]
    assert "Content-Security-Policy" in fn and "default-src 'none'" in fn
    assert "style-src 'unsafe-inline'" in fn and "img-src data:" in fn


# ---------- f. en-têtes ----------

def test_index_security_headers():
    r = client().get("/")
    assert r.status_code == 200
    h = r.headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] in ("no-referrer", "same-origin")
    csp = h["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp and "base-uri 'none'" in csp
    assert "'unsafe-eval'" not in csp


def test_index_csp_hashes_inline_script_and_handlers():
    import base64
    import hashlib
    import re
    html = main.index()
    csp = client().get("/").headers["content-security-policy"]
    script = html[html.index("<script>") + 8: html.index("</script>")]
    h = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"'sha256-{h}'" in csp
    script_src = [d for d in csp.split(";") if d.strip().startswith("script-src")][0]
    assert "'unsafe-inline'" not in script_src
    for v in set(re.findall(r'\son[a-z]+="([^"]*)"', html)):
        hv = base64.b64encode(hashlib.sha256(v.encode()).digest()).decode()
        assert f"'sha256-{hv}'" in script_src, v


def test_api_responses_have_headers():
    r = client().get("/sante")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


# ---------- i. limite de débit ----------

def test_rate_limit_light_mode(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: dict(R, langue_source="fr", avertissements=[]))
    codes = [client().post("/resumer", json={"url": "u", "api_key": "k"}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    # les routes non LLM ne sont pas limitées
    assert client().post("/export/markdown", json=R).status_code == 200


def test_rate_limit_default_20_light(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.delenv("RATE_LIMIT_PER_MIN", raising=False)
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: dict(R, langue_source="fr", avertissements=[]))
    codes = [client().post("/resumer", json={"url": "u", "api_key": "k"}).status_code for _ in range(21)]
    assert codes.count(200) == 20 and codes[-1] == 429


def test_rate_limit_off_in_full_without_env(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("RATE_LIMIT_PER_MIN", raising=False)
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: dict(R, langue_source="fr", avertissements=[]))
    codes = [client().post("/resumer", json={"url": "u"}).status_code for _ in range(25)]
    assert set(codes) == {200}


def test_rate_limit_env_in_full(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: dict(R, langue_source="fr", avertissements=[]))
    codes = [client().post("/resumer", json={"url": "u"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limit_ignores_spoofed_xff_outside_vercel(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: dict(R, langue_source="fr", avertissements=[]))
    c = client()
    assert c.post("/resumer", json={"url": "u", "api_key": "k"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert c.post("/resumer", json={"url": "u", "api_key": "k"}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


# ---------- j. fournisseurs d'image ----------

def test_image_only_openrouter_provider():
    assert set(ig.IMAGE_PROVIDERS) & {"pruna", "replicate", "stability-ai"} == set()
    for f in ("_call_pruna", "_call_replicate", "_call_stability_ai"):
        assert not hasattr(ig, f)
    assert ig.generate_image_result("p", provider="pruna", api_key="k")["success"] is False


def test_image_route_never_sends_foreign_key_to_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = client().post("/image", json={"resultat": R, "api_key": "sk-proj-openai", "provider": "openai"})
    assert r.status_code == 400


# ---------- reel : injection d'arguments yt-dlp / SSRF ----------

@pytest.mark.parametrize("url", [
    "--exec=touch${IFS}/tmp/pwned#youtube.com/watch?v=dQw4w9WgXcQ",
    "http://169.254.169.254/latest/?youtube.com/watch?v=dQw4w9WgXcQ",
    "https://evil.example/youtu.be/dQw4w9WgXcQ",
    "file:///etc/passwd?v=dQw4w9WgXcQ",
])
def test_extract_youtube_id_strict(url):
    assert extract_youtube_id(url) is None


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
    "https://m.youtube.com/watch?feature=x&v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?si=abc",
    "youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ",
    "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    "https://www.youtube.com/live/dQw4w9WgXcQ",
])
def test_extract_youtube_id_valid(url):
    assert extract_youtube_id(url) == "dQw4w9WgXcQ"


def test_reel_passes_canonical_url(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.setattr(main, "get_capabilities", _all_caps)
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: dict(R, langue_source="fr", avertissements=[]))
    seen = []
    monkeypatch.setattr(main, "_highlight_reel", lambda url, *a: seen.append(url) or {"reel_path": "x", "clip_count": 1})
    r = client().post("/reel", json={"url": "https://youtu.be/dQw4w9WgXcQ?si=1", "api_key": "k"})
    assert r.status_code == 200 and seen == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    seen.clear()
    r = client().post("/reel", json={"url": "--exec=id#youtube.com/watch?v=dQw4w9WgXcQ", "api_key": "k"})
    assert r.status_code == 400 and seen == []


def test_download_video_uses_end_of_options(monkeypatch, tmp_path):
    from lib import video_mounter as vm
    cmds = []
    class Res:
        returncode, stderr = 1, "x"
    monkeypatch.setattr(vm, "_find_yt_dlp", lambda: "yt-dlp")
    monkeypatch.setattr(vm.subprocess, "run", lambda cmd, **k: cmds.append(cmd) or Res())
    with pytest.raises(RuntimeError):
        vm.download_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", str(tmp_path))
    assert cmds[0][-2:] == ["--", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]


def test_reel_clip_duration_bounded():
    r = client().post("/reel", json={"url": "https://youtu.be/dQw4w9WgXcQ", "duree_clip": 1e9})
    assert r.status_code == 422
