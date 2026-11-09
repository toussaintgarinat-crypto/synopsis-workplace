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
