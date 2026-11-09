import json
import re
from pathlib import Path
import pytest
from lib import image_generator as ig
from lib.exporters.obsidian import note_content
from lib.i18n import TEXTS
from fastapi.testclient import TestClient
import main


class Img:
    def __init__(self, status=200, ctype="image/png", content=b"PNG"):
        self.status_code, self.headers, self.content = status, {"content-type": ctype}, content


def _patch_url(monkeypatch, url, resp, calls):
    monkeypatch.setattr(ig, "generate_image_result", lambda *a, **k: {"success": True, "image_url": url})
    import httpx
    def get(u, **k):
        calls.append(k)
        return resp
    monkeypatch.setattr(httpx, "get", get)


def test_image_download_https_only(monkeypatch):
    calls = []
    _patch_url(monkeypatch, "http://evil/a.png", Img(), calls)
    with pytest.raises(RuntimeError, match="https"):
        ig.generate_image("p", "k")
    assert not calls


def test_image_download_ok_no_redirects(monkeypatch):
    calls = []
    _patch_url(monkeypatch, "https://cdn.example/a.png", Img(), calls)
    assert ig.generate_image("p", "k") == b"PNG"
    assert calls[0].get("follow_redirects") is False


def test_image_download_requires_image_content_type(monkeypatch):
    _patch_url(monkeypatch, "https://cdn.example/a", Img(ctype="text/html"), [])
    with pytest.raises(RuntimeError, match="image"):
        ig.generate_image("p", "k")


def test_image_download_redirect_refused(monkeypatch):
    _patch_url(monkeypatch, "https://cdn.example/a", Img(status=302), [])
    with pytest.raises(RuntimeError):
        ig.generate_image("p", "k")


def test_frontmatter_newlines_in_source_and_langue():
    n = note_content({"titre": "T", "langue_source": 'fr\nevil: 1', "resume": "R"}, 'http://x/\nmalicieux: "oui"')
    head = n.split("---")[1]
    assert len([l for l in head.strip().split("\n")]) == 4  # titre, langue, source, tags
    assert 'source: "http://x/ malicieux: \\"oui\\""' in head
    assert 'langue: "fr evil: 1"' in head


def _front():
    html = main.index()
    return html[html.index("<script>") + 8: html.index("</script>")]


def test_front_has_no_backslash_except_known_regex():
    js = _front()
    bad = [l for l in js.split("\n") if "\\" in l and "extractVideoId" not in l and "url.match" not in l]
    assert bad == []


def test_front_escape_html_quotes():
    js = _front()
    fn = js[js.index("function escapeHtml"): js.index("document.getElementById('url')")]
    assert "&quot;" in fn and "&#39;" in fn


def test_front_placeholders_escaped_and_info():
    js = _front()
    assert "escapeHtml(t('ph_question'))" in js and "escapeHtml(t('ph_vault'))" in js
    assert "showInfo(t('ok_obsidian')" in js
    assert 'id="narration" data-cap="tts"' in main.index()
    assert "!apiKey && !sansCle(provider)" in js
    assert "method: 'POST'" in js[js.index("async function rafraichirModeles"):js.index("function toggleKeyVisibility")]


def test_front_export_strips_fields():
    js = _front()
    fn = js[js.index("function documentExport"): js.index("function exporterHtml")]
    for sel in ("input", "textarea", "#qa-box", "#image-out"):
        assert sel in fn


def test_front_sequential_batch_uses_resumer():
    js = _front()
    fn = js[js.index("async function resumerLot"): js.index("async function resumerFichier")]
    assert "'/resumer'" in fn and "/resumer-lot" not in fn


def test_front_custom_provider_and_warnings():
    html = main.index()
    assert 'value="custom" data-cap="llm_custom"' in html
    assert "avertissements" in _front()
    assert "if (sel && sel.selectedIndex >= 0 && sel.options[sel.selectedIndex].disabled)" in _front()


def test_i18n_new_keys_both_languages():
    for k in ("prov_custom", "avertissements"):
        assert TEXTS["fr"][k] and TEXTS["en"][k]


def test_vercel_includes_prompts():
    cfg = json.loads((Path(main.__file__).parent / "vercel.json").read_text())
    f = cfg["functions"]["api/index.py"]
    assert f["includeFiles"] == "prompts/**" and f["maxDuration"] == 60 and cfg["rewrites"]
