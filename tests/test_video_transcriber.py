"""Transcription de secours d'une vidéo YouTube via OpenRouter + Gemini (lib/video_transcriber)."""
import httpx
import pytest

from lib import video_transcriber as vt

VID = "jNQXAC9IVRw"


# ---------- parse_timestamped_transcript ----------

def test_parse_simple_mm_ss():
    out = vt.parse_timestamped_transcript("[00:01] Bonjour\n[00:05] Deuxième phrase\n[00:12] Fin")
    assert out == [
        {"text": "Bonjour", "start": 1.0, "duration": 4.0},
        {"text": "Deuxième phrase", "start": 5.0, "duration": 7.0},
        {"text": "Fin", "start": 12.0, "duration": vt.LAST_DURATION},
    ]


def test_parse_spaces_inside_brackets():
    out = vt.parse_timestamped_transcript("[ 00:01 ] a\n[ 00:19 ] b")
    assert [e["start"] for e in out] == [1.0, 19.0]
    assert [e["text"] for e in out] == ["a", "b"]


def test_parse_hours():
    out = vt.parse_timestamped_transcript("[0:59:58] a\n[1:00:02] b\n[01:00:10] c")
    assert [e["start"] for e in out] == [3598.0, 3602.0, 3610.0]
    assert out[0]["duration"] == 4.0


def test_parse_code_fences_markdown_and_blank_lines():
    txt = "Voici la transcription :\n```text\n\n- **[00:00]** Salut\n\n* [00:03] : Ça va ?\n```\n"
    out = vt.parse_timestamped_transcript(txt)
    assert [(e["start"], e["text"]) for e in out] == [(0.0, "Salut"), (3.0, "Ça va ?")]


def test_parse_range_and_continuation_line():
    out = vt.parse_timestamped_transcript("[00:00 - 00:04] début\nsuite de la phrase\n[00:04] après")
    assert out[0] == {"text": "début suite de la phrase", "start": 0.0, "duration": 4.0}
    assert out[1]["start"] == 4.0


def test_parse_non_monotonic_merged_into_previous():
    out = vt.parse_timestamped_transcript("[00:10] a\n[00:05] b\n[00:20] c")
    assert [e["start"] for e in out] == [10.0, 20.0]
    assert out[0]["text"] == "a b"
    assert all(e["duration"] > 0 for e in out)


def test_parse_skips_empty_text_entries():
    out = vt.parse_timestamped_transcript("[00:01]\n[00:02] texte")
    assert [e["text"] for e in out] == ["texte"]


@pytest.mark.parametrize("txt", ["", "   ", "Pas d'horodatage ici.\nNi là.", "```\n```", None])
def test_parse_without_timestamps_raises_french(txt):
    with pytest.raises(ValueError, match="horodatage"):
        vt.parse_timestamped_transcript(txt)


@pytest.mark.parametrize("ligne", ["00:05 texte", "**00:05** texte", "00:05 - texte", "00:05: texte",
                                   "- 00:05 texte", "**00:05 -** texte", "0:00:05 texte"])
def test_parse_timestamp_without_brackets(ligne):
    out = vt.parse_timestamped_transcript(ligne + "\n[00:09] fin")
    assert out[0] == {"text": "texte", "start": 5.0, "duration": 4.0}


def test_parse_mixed_bracketed_and_bare():
    out = vt.parse_timestamped_transcript("[00:01] a\n00:03 b\n**00:07** c\n(00:09) d")
    assert [(e["start"], e["text"]) for e in out] == [(1.0, "a"), (3.0, "b"), (7.0, "c"), (9.0, "d")]


@pytest.mark.parametrize("junk", ["[00:75] faux", "00:61 faux", "[75:30] faux", "62:10 faux", "[1:60:00] faux"])
def test_parse_out_of_range_timestamps_are_junk(junk):
    out = vt.parse_timestamped_transcript(f"[00:01] a\n{junk}\n[00:04] b")
    assert [(e["start"], e["text"]) for e in out] == [(1.0, "a"), (4.0, "b")]


def test_parse_bare_timestamp_at_line_start_is_entry():
    out = vt.parse_timestamped_transcript("[00:01] il est\n10:30 passé le rendez-vous\n[00:20] b")
    # Un horodatage nu en début de ligne est une entrée ; 10:30 > 00:01 -> nouvelle entrée valide.
    assert out[1]["start"] == 630.0


def test_parse_text_starting_with_digits_not_timestamp():
    out = vt.parse_timestamped_transcript("[00:01] a\n2024 fut une année\n[00:04] b")
    assert out[0]["text"] == "a 2024 fut une année"


# ---------- transcribe_youtube_via_llm ----------

class Rsp:
    def __init__(self, status=200, content="[00:01] éléphants\n[00:05] trompes", text=""):
        self.status_code = status
        self._content = content
        self.text = text or "upstream"

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture
def post(monkeypatch):
    seen = {}

    def fake(url, json=None, headers=None, timeout=None):
        seen.update(url=url, json=json, headers=headers, timeout=timeout)
        return seen.get("rsp") or Rsp()
    monkeypatch.setattr(vt.httpx, "post", fake)
    monkeypatch.setattr(vt, "get_video_title", lambda vid: "Me at the zoo")
    monkeypatch.delenv("SYNOPSIS_MODE", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    return seen


def test_builds_exact_request(post):
    out = vt.transcribe_youtube_via_llm(VID, "sk-or-v1-test", "google/gemini-2.5-flash")
    assert post["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert post["json"] == {
        "model": "google/gemini-2.5-flash",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": vt.PROMPT},
            {"type": "video_url", "video_url": {"url": f"https://www.youtube.com/watch?v={VID}"}},
        ]}],
        "provider": {"only": ["google-ai-studio"], "allow_fallbacks": False},
    }
    assert post["headers"]["Authorization"] == "Bearer sk-or-v1-test"
    assert post["timeout"] == 300
    assert out["video_id"] == VID and out["title"] == "Me at the zoo"
    assert out["source"] == "llm-video" and out["language"] == "unknown"
    assert out["entries_count"] == 2 and out["transcript"][0] == {"text": "éléphants", "start": 1.0, "duration": 4.0}
    assert out["total_duration_minutes"] == pytest.approx((5 + vt.LAST_DURATION) / 60)


def test_prompt_is_the_verified_one():
    assert vt.PROMPT == ("Transcris fidèlement la parole de cette vidéo. Format strict : une ligne par phrase, "
                         "préfixée de l'horodatage [MM:SS]. Aucun commentaire.")


@pytest.mark.parametrize("src", [
    f"https://youtu.be/{VID}?si=x&evil=1",
    f"https://www.youtube.com/watch?v={VID}&list=PL1#t=3",
    f"https://www.youtube.com/shorts/{VID}",
])
def test_url_rebuilt_from_id(post, src):
    vt.transcribe_youtube_via_llm(src, "k", "google/gemini-2.5-flash")
    assert post["json"]["messages"][0]["content"][1]["video_url"]["url"] == f"https://www.youtube.com/watch?v={VID}"


@pytest.mark.parametrize("src", ["https://evil.com/watch?v=" + VID, "file:///etc/passwd", "-o /tmp/x", "abc"])
def test_rejects_non_youtube(post, src):
    with pytest.raises(ValueError, match="YouTube"):
        vt.transcribe_youtube_via_llm(src, "k", "google/gemini-2.5-flash")
    assert "url" not in post


@pytest.mark.parametrize("model", ["openai/gpt-5.2", "anthropic/claude-sonnet-4.5", "google/gem ini", "google/" + "x" * 100,
                                   "google/a;rm", ""])
def test_rejects_bad_models(post, model):
    if model == "":
        # Vide = modèle par défaut (accepté)
        vt.transcribe_youtube_via_llm(VID, "k", model)
        assert post["json"]["model"] == vt.DEFAULT_MODEL
        return
    with pytest.raises(ValueError, match="(?i)modèle"):
        vt.transcribe_youtube_via_llm(VID, "k", model)
    assert "url" not in post


@pytest.mark.parametrize("prov", ["openai", "opencode", "local", "custom"])
def test_rejects_non_openrouter_provider(post, prov):
    with pytest.raises(ValueError, match="OpenRouter"):
        vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash", provider=prov)
    assert "url" not in post


def test_requires_key(post):
    with pytest.raises(ValueError, match="(?i)clé"):
        vt.transcribe_youtube_via_llm(VID, "", "google/gemini-2.5-flash")


@pytest.mark.parametrize("status,needle", [(402, "rédit"), (401, "refusée"), (403, "refusée"), (429, "imite")])
def test_http_errors_mapped(post, status, needle):
    post["rsp"] = Rsp(status=status, text='{"error":"key sk-or-v1-abcdefabcdef"}')
    with pytest.raises(ValueError) as e:
        vt.transcribe_youtube_via_llm(VID, "sk-or-v1-abcdefabcdef", "google/gemini-2.5-flash")
    assert needle in str(e.value)
    assert "sk-or-v1-abcdefabcdef" not in str(e.value)


def test_other_http_error_scrubbed(post):
    post["rsp"] = Rsp(status=500, text="boom Bearer sk-or-v1-zzzzzzzzzz")
    with pytest.raises(ValueError) as e:
        vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash")
    assert "500" in str(e.value) and "sk-or-v1-zzzz" not in str(e.value)


def test_light_mode_timeout_message(post, monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")

    def slow(url, json=None, headers=None, timeout=None):
        post["timeout"] = timeout
        raise httpx.ReadTimeout("t")
    monkeypatch.setattr(vt.httpx, "post", slow)
    with pytest.raises(ValueError) as e:
        vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash")
    assert str(e.value) == ("Vidéo trop longue pour être transcrite en ligne : la version en ligne doit transcrire et "
                            "résumer en moins de 60 s. Essayez une vidéo plus courte ou la version locale.")
    assert post["timeout"] == 50


def test_full_mode_timeout_message(post, monkeypatch):
    monkeypatch.setattr(vt.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("t")))
    with pytest.raises(ValueError, match="(?i)délai"):
        vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash")


def test_network_error_scrubbed(post, monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("fail Bearer sk-or-v1-secretsecret")
    monkeypatch.setattr(vt.httpx, "post", boom)
    with pytest.raises(ValueError) as e:
        vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash")
    assert "sk-or-v1-secretsecret" not in str(e.value)


def test_unparsable_response_raises(post):
    post["rsp"] = Rsp(content="Je ne peux pas regarder cette vidéo.")
    with pytest.raises(ValueError, match="horodatage"):
        vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash")


def test_content_as_list_of_parts(post):
    class R2(Rsp):
        def json(self):
            return {"choices": [{"message": {"content": [{"type": "text", "text": "[00:02] a"}]}}]}
    post["rsp"] = R2()
    out = vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash")
    assert out["transcript"][0]["start"] == 2.0


def test_error_payload_in_200(post):
    class R3(Rsp):
        def json(self):
            return {"error": {"message": "video too long", "code": 400}}
    post["rsp"] = R3()
    with pytest.raises(ValueError, match="video too long"):
        vt.transcribe_youtube_via_llm(VID, "k", "google/gemini-2.5-flash")
