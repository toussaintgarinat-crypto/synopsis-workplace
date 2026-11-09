from unittest.mock import patch
from lib.whisper_transcriber import segments_to_transcript


def test_segments():
    segs = [{"start": 0.0, "end": 2.5, "text": " Bonjour"}]
    assert segments_to_transcript(segs) == [{"text": "Bonjour", "start": 0.0, "duration": 2.5}]


def test_summarize_data_accepts_dict():
    import main
    data = {"transcript": [{"text": "Bonjour", "start": 0.0, "duration": 2.0}], "title": "t", "language": "fr"}
    with patch("main.llm_complete", return_value="## Resume\nok"):
        out = main._summarize_data(data, "Français", "m", api_key="k", provider="openai")
    assert out["titre"] == "t" and out["langue_source"] == "fr" and "ok" in out["resume"]


# ---------- M4 : modèle réellement utilisé ----------

def test_with_model_reports_local_size(monkeypatch):
    import sys
    import types
    from lib import whisper_transcriber as wt

    class M:
        def transcribe(self, path, language=None):
            return {"segments": [{"text": " a", "start": 0, "end": 1}]}
    monkeypatch.setitem(sys.modules, "whisper", types.SimpleNamespace(load_model=lambda name: M()))
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert wt.transcribe_file_with_model("/x", "", model_size="small")[1] == "small"
    assert wt.transcribe_file_with_model("/x", "")[1] == "base"


def test_with_model_reports_whisper_1_on_api_fallback(monkeypatch, tmp_path):
    import sys
    import httpx
    from lib import whisper_transcriber as wt
    monkeypatch.setitem(sys.modules, "whisper", None)  # import whisper -> ImportError
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    seen = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"segments": [{"text": "b", "start": 0, "end": 2}]}
    monkeypatch.setattr(httpx, "post", lambda url, **k: seen.update(url=url, data=k.get("data")) or R())
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    segs, modele = wt.transcribe_file_with_model(str(f), "", model_size="large")
    assert modele == "whisper-1" and seen["data"]["model"] == "whisper-1" and segs[0]["text"] == "b"
    assert wt.transcribe_file(str(f), "") == segs


def test_route_reports_model_actually_used(monkeypatch):
    from fastapi.testclient import TestClient
    import main
    import lib.whisper_transcriber as wt
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.setattr(main, "get_capabilities", lambda: {"whisper": {"ok": True, "raison": ""}})
    monkeypatch.setattr(wt, "transcribe_file_with_model",
                        lambda p, l="", model_size="": ([{"text": "x", "start": 0.0, "duration": 1.0}], "whisper-1"))
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: "## 📍 Chapitrage Temporel\n| [00:01] | *S* | D |")
    r = TestClient(main.app).post("/resumer-fichier", files={"fichier": ("a.mp3", b"x" * 10)},
                                  data={"api_key": "k", "modeles": '{"transcription_audio": "small"}'})
    assert r.status_code == 200
    assert r.json()["modeles_utilises"]["transcription_audio"] == "whisper-1"
