import pytest
import main


def test_light_mode_rejects_long(monkeypatch):
    """Light mode should reject videos longer than 30 minutes."""
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setenv("MAX_MINUTES_LIGHT", "30")
    data = {
        "transcript": [{"text": "x", "start": 0, "duration": 1}],
        "title": "t",
        "language": "fr",
        "total_duration_minutes": 45
    }
    with pytest.raises(ValueError, match=r"trop longue.*max 30 min"):
        main._summarize_data(data, "Français", "", "", "")


def test_full_mode_allows_long(monkeypatch):
    """Full mode should allow videos longer than 30 minutes."""
    monkeypatch.delenv("SYNOPSIS_MODE", raising=False)
    # Mock llm_complete to avoid network calls
    def mock_llm(prompt, **kwargs):
        return "Résumé du vidéo: test"

    monkeypatch.setattr(main, "llm_complete", mock_llm)

    data = {
        "transcript": [{"text": "x", "start": 0, "duration": 1}],
        "title": "t",
        "language": "fr",
        "total_duration_minutes": 45
    }
    # Should not raise an error
    result = main._summarize_data(data, "Français", "", "", "")
    assert "titre" in result


def test_light_mode_limit_boundary_and_message(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setenv("MAX_MINUTES_LIGHT", "10")
    monkeypatch.setattr(main, "llm_complete", lambda prompt, **k: "ok")
    base = {"transcript": [{"text": "x", "start": 0, "duration": 1}], "title": "t", "language": "fr"}
    assert "titre" in main._summarize_data(dict(base, total_duration_minutes=10), "Français", "", "", "")
    with pytest.raises(ValueError, match="max 10 min"):
        main._summarize_data(dict(base, total_duration_minutes=10.5), "Français", "", "", "")
