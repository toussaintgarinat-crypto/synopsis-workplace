import pytest
import main


def _raise(msg):
    def f(url, *a, **k):
        raise ValueError(msg)
    return f


def test_light_mode_explique_les_limites(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setattr(main, "get_youtube_transcript", _raise("Aucun transcript disponible pour cette vidéo"))
    with pytest.raises(ValueError) as e:
        main._summarize("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    msg = str(e.value)
    assert "Aucun transcript disponible" in msg
    assert "sous-titres" in msg and "pas de transcription audio" in msg and "Whisper" in msg


def test_mode_complet_suggere_whisper(monkeypatch):
    monkeypatch.delenv("SYNOPSIS_MODE", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.setattr(main, "get_youtube_transcript", _raise("Transcript désactivé sur cette vidéo YouTube"))
    with pytest.raises(ValueError, match="Whisper"):
        main._summarize("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_autres_erreurs_inchangees(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setattr(main, "get_youtube_transcript", _raise("URL YouTube invalide. Format: youtube.com/watch?v=ID ou youtu.be/ID"))
    with pytest.raises(ValueError) as e:
        main._summarize("nimportequoi")
    assert "Whisper" not in str(e.value)
