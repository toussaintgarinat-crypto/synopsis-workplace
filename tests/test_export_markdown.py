from lib.exporters.markdown import to_markdown

D = {"titre": "Vidéo", "langue_source": "fr", "resume": "Texte",
     "chapitres": [{"timestamp": "01:30", "subject": "Intro", "description": "d"}],
     "insights": [{"title": "T", "timestamp": "02:10", "description": "e"}]}

def test_markdown():
    md = to_markdown(D)
    assert md.startswith("# Vidéo")
    assert "- **01:30** Intro — d" in md
    assert "1. **T** [02:10] : e" in md
