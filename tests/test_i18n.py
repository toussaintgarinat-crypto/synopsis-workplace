from lib.i18n import TEXTS

def test_same_keys():
    assert set(TEXTS["fr"]) == set(TEXTS["en"])

def test_required_keys():
    for k in ["resumer", "chapitres", "insights", "resume", "export_html", "export_md",
              "export_pdf", "question", "plusieurs", "config", "langue_source"]:
        assert k in TEXTS["fr"]

def test_no_empty_values():
    assert all(TEXTS["fr"].values()) and all(TEXTS["en"].values())
