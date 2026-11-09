import pytest
from lib.exporters.obsidian import note_content, save_note
from lib.exporters.excalidraw import to_excalidraw

@pytest.fixture(autouse=True)
def _racine_tmp(tmp_path, monkeypatch):
    # Politique du vault : racine autorisée = dossier temporaire du test (défaut réel : ~).
    monkeypatch.setenv("OBSIDIAN_ALLOWED_ROOTS", str(tmp_path))


D = {"titre": "Ma vidéo", "langue_source": "fr", "resume": "R",
     "chapitres": [{"timestamp": "00:10", "subject": "A", "description": "d"}], "insights": []}

def test_note_has_frontmatter():
    n = note_content(D, "https://youtu.be/x")
    assert n.startswith("---\n") and 'titre: "Ma vidéo"' in n and 'source: "https://youtu.be/x"' in n

def test_save_note(tmp_path):
    p = save_note(D, str(tmp_path))
    assert p.endswith("Ma vidéo.md") and open(p, encoding="utf-8").read().startswith("---")

def test_save_note_traversal(tmp_path):
    vault = tmp_path / "vault"
    d = dict(D, titre="../../evil")
    p = save_note(d, str(vault))
    assert (vault / "evil.md").exists() and str(vault) in p
    assert not (tmp_path / "evil.md").exists()

def test_save_note_dotdot_title(tmp_path):
    p = save_note(dict(D, titre=".."), str(tmp_path / "v"))
    assert p.startswith(str(tmp_path / "v"))

def test_frontmatter_quotes_escaped():
    n = note_content(dict(D, titre='A "b"'))
    assert 'titre: "A \\"b\\""' in n

def test_empty_vault_rejected():
    with pytest.raises(ValueError):
        save_note(D, "  ")

def test_excalidraw():
    e = to_excalidraw(D)
    assert e["type"] == "excalidraw"
    assert any(el["type"] == "text" and "A" in el["text"] for el in e["elements"])
