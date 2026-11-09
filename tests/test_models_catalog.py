"""Catalogue de modèles par tâche et résolveur des modèles d'une requête."""
import httpx
import pytest

from lib import models_catalog as mc
from lib import image_generator


class Rsp:
    def __init__(self, data, status=200):
        self.status_code = status
        self._data = data

    def json(self):
        return {"data": self._data}


@pytest.fixture
def get(monkeypatch):
    calls = []
    state = {"data": [{"id": "z/b", "name": "Z B"}, {"id": "a/a"}], "status": 200}

    def fake(url, headers=None, timeout=0):
        calls.append({"url": url, "headers": headers})
        if state.get("boom"):
            raise httpx.ConnectError("x")
        return Rsp(state["data"], state["status"])
    monkeypatch.setattr(httpx, "get", fake)
    return calls, state


# ---------- fetch_models ----------

def test_text_tasks_use_default_catalog(get):
    calls, _ = get
    out = mc.fetch_models("openrouter", "resume", "")
    assert out["models"] == ["a/a", "z/b"] and out["source"] == "api"
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/models"
    assert out["noms"] == {"z/b": "Z B"}


def test_video_task_filters_video_and_google(get):
    calls, state = get
    state["data"] = [{"id": "google/gemini-2.5-flash", "name": "Gemini Flash"}, {"id": "openai/gpt-4o"},
                     {"id": "google/gemini-2.5-pro"}]
    out = mc.fetch_models("openrouter", "transcription_video", "")
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/models?input_modalities=video"
    assert out["models"] == ["google/gemini-2.5-flash", "google/gemini-2.5-pro"]
    assert out["source"] == "api"


def test_image_task_uses_output_modality(get):
    calls, state = get
    state["data"] = [{"id": "black-forest-labs/flux.2-pro"}, {"id": "google/gemini-2.5-flash-image"}]
    out = mc.fetch_models("openrouter", "image", "")
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/models?output_modalities=image"
    assert out["models"] == ["black-forest-labs/flux.2-pro", "google/gemini-2.5-flash-image"]


def test_cache_ten_minutes(get, monkeypatch):
    calls, _ = get
    t = {"now": 1000.0}
    monkeypatch.setattr(mc.time, "monotonic", lambda: t["now"])
    mc.fetch_models("openrouter", "image", "")
    mc.fetch_models("openrouter", "image", "")
    assert len(calls) == 1
    t["now"] += 601
    mc.fetch_models("openrouter", "image", "")
    assert len(calls) == 2


def test_cache_per_task(get):
    calls, _ = get
    mc.fetch_models("openrouter", "image", "")
    mc.fetch_models("openrouter", "transcription_video", "")
    assert len(calls) == 2


def test_fallback_not_cached(get):
    calls, state = get
    state["boom"] = True
    out = mc.fetch_models("openrouter", "transcription_video", "")
    assert out["source"] == "fallback" and all(m.startswith("google/") for m in out["models"]) and out["models"]
    state["boom"] = False
    assert mc.fetch_models("openrouter", "transcription_video", "")["source"] == "api"


def test_image_fallback_lists_image_generator_models(get):
    _, state = get
    state["status"] = 500
    out = mc.fetch_models("openrouter", "image", "")
    assert out["source"] == "fallback" and image_generator.DEFAULT_MODEL in out["models"]


@pytest.mark.parametrize("prov", ["openai", "opencode"])
@pytest.mark.parametrize("task", ["transcription_video", "image"])
def test_other_providers_no_video_or_image(get, prov, task):
    calls, _ = get
    out = mc.fetch_models(prov, task, "k")
    assert out["models"] == [] and out["raison"] and not calls


def test_audio_task_fixed_whisper_sizes(get):
    calls, _ = get
    out = mc.fetch_models("openai", "transcription_audio", "")
    assert out["models"] == ["tiny", "base", "small", "medium", "large"] and not calls


def test_opencode_text_static(get):
    calls, _ = get
    out = mc.fetch_models("opencode", "qa", "")
    assert out["source"] == "fallback" and out["models"] and not calls


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        mc.fetch_models("openrouter", "pirate", "")


# ---------- resolve_models ----------

def test_resolver_defaults(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    out = mc.resolve_models(None, "deepseek/x")
    assert out == {
        "resume": "deepseek/x", "fusion": "deepseek/x", "qa": "deepseek/x",
        "transcription_video": "google/gemini-2.5-flash", "transcription_audio": "base",
        "image": image_generator.DEFAULT_MODEL,
    }


def test_resolver_fusion_qa_follow_resume():
    out = mc.resolve_models({"resume": "a/r", "fusion": "", "qa": "  "}, "d/d")
    assert out["resume"] == "a/r" and out["fusion"] == "a/r" and out["qa"] == "a/r"


def test_resolver_per_task_and_unknown_keys_ignored(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    out = mc.resolve_models({"resume": "a/r", "fusion": "b/f", "qa": "c/q", "image": "x/img",
                             "transcription_video": "google/gemini-2.5-pro", "pirate": "!!"}, "d/d")
    assert out["fusion"] == "b/f" and out["qa"] == "c/q" and out["image"] == "x/img"
    assert out["transcription_video"] == "google/gemini-2.5-pro" and out["transcription_audio"] == "small"
    assert "pirate" not in out


@pytest.mark.parametrize("bad", [{"resume": "a b"}, {"fusion": "x;y"}, {"qa": "x" * 101}, {"image": "<script>"},
                                 {"qa": 3}, {"transcription_video": "openai/gpt-4o"},
                                 {"transcription_audio": "huge"}, {"transcription_audio": "../base"}])
def test_resolver_rejects_invalid(bad):
    with pytest.raises(ValueError) as e:
        mc.resolve_models(bad, "d/d")
    assert "odèle" in str(e.value)


def test_resolver_rejects_invalid_default():
    with pytest.raises(ValueError):
        mc.resolve_models(None, "a b")


def test_resolver_rejects_non_dict():
    with pytest.raises(ValueError):
        mc.resolve_models(["resume"], "d/d")


def test_resolver_empty_default_uses_env_model():
    out = mc.resolve_models({}, "")
    assert out["resume"] and mc.MODEL_RE.fullmatch(out["resume"])


# ---------- whisper model_size ----------

def test_whisper_model_size_validated():
    from lib import whisper_transcriber as wt
    assert wt.WHISPER_SIZES == ["tiny", "base", "small", "medium", "large"]
    with pytest.raises(ValueError):
        wt.transcribe_file("/nonexistent", "", model_size="huge")


def test_whisper_model_size_passed_to_loader(monkeypatch):
    import sys
    import types
    from lib import whisper_transcriber as wt
    seen = {}

    class M:
        def transcribe(self, path, language=None):
            return {"segments": [{"text": " a", "start": 0, "end": 1}]}
    fake = types.SimpleNamespace(load_model=lambda name: seen.update(name=name) or M())
    monkeypatch.setitem(sys.modules, "whisper", fake)
    assert wt.transcribe_file("/x", "", model_size="small")[0]["text"] == "a"
    assert seen["name"] == "small"
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    wt.transcribe_file("/x", "")
    assert seen["name"] == "medium"
