from lib import capabilities as c


def test_light_mode_disables_heavy(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    caps = c.get_capabilities()
    assert caps["resume"]["ok"] is True
    for k in ("whisper", "reel", "llm_local", "llm_custom", "obsidian", "drive", "excalidraw", "images", "tts"):
        assert caps[k]["ok"] is False
        assert "en ligne" in caps[k]["raison"]


def test_full_mode_reports_ffmpeg(monkeypatch):
    monkeypatch.delenv("SYNOPSIS_MODE", raising=False)
    monkeypatch.setattr(c.shutil, "which", lambda name: None)
    caps = c.get_capabilities()
    assert caps["reel"]["ok"] is False
    assert "ffmpeg" in caps["reel"]["raison"]


def test_vercel_env_forces_light(monkeypatch):
    monkeypatch.delenv("SYNOPSIS_MODE", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    assert c.is_light_mode() is True
    monkeypatch.delenv("VERCEL")
    assert c.is_light_mode() is False


def test_llm_custom_needs_env_in_full(monkeypatch):
    monkeypatch.delenv("SYNOPSIS_MODE", raising=False)
    monkeypatch.delenv("CUSTOM_LLM_URL", raising=False)
    assert c.get_capabilities()["llm_custom"]["ok"] is False
    monkeypatch.setenv("CUSTOM_LLM_URL", "http://x/v1")
    assert c.get_capabilities()["llm_custom"]["ok"] is True


def test_transcription_video_ok_in_both_modes(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    assert c.get_capabilities()["transcription_video"] == {"ok": True, "raison": ""}
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    assert c.get_capabilities()["transcription_video"]["ok"] is True
