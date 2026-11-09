from lib.fusion import _extract_chapters, _extract_insights

SAMPLE = """## 📍 Chapitrage Temporel
| Time | Sujet | Description |
| :--- | :--- | :--- |
| [01:30] | *Intro* | Présentation |
| [1:02:03] | *Fin* | Conclusion |

## 💡 Top 3 Moments Forts (Insights)
1. *Idée* [02:10] : Explication
"""

def test_chapters_parsed():
    ch = _extract_chapters(SAMPLE)
    assert ch[0] == {'timestamp': '01:30', 'ts_seconds': 90, 'subject': 'Intro', 'description': 'Présentation'}
    assert ch[1] == {'timestamp': '1:02:03', 'ts_seconds': 3723, 'subject': 'Fin', 'description': 'Conclusion'}
    assert len(ch) == 2

def test_insights_parsed():
    ins = _extract_insights(SAMPLE)
    assert ins == [{'title': 'Idée', 'timestamp': '02:10', 'description': 'Explication'}]

def test_empty():
    assert _extract_chapters("") == []
    assert _extract_insights("rien") == []
