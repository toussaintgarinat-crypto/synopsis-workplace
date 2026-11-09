"""Repli de transcription par modèle vidéo dans _summarize, modèles par tâche sur les routes,
paramètre `tache` de /modeles, sécurité des routes modifiées et section du front."""
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from lib import image_generator
from lib.extractor import SansSousTitres
from lib.i18n import TEXTS

URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
R = {"titre": "T", "resume": "R", "chapitres": [], "insights": []}
LLM_DATA = {"video_id": "jNQXAC9IVRw", "transcript": [{"text": "éléphants", "start": 1.0, "duration": 4.0}],
            "language": "unknown", "title": "Me at the zoo", "total_duration_minutes": 0.1, "entries_count": 1,
            "source": "llm-video", "model": "google/gemini-2.5-flash"}
SUBS = {"video_id": "jNQXAC9IVRw", "transcript": [{"text": "a", "start": 0.0, "duration": 1.0}], "language": "en",
        "title": "T", "total_duration_minutes": 0.1, "entries_count": 1}


def client():
    return TestClient(main.app)


def _no_subs(*a, **k):
    raise SansSousTitres("Aucun transcript disponible pour cette vidéo")


@pytest.fixture
def full(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "full")
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("GATEWAY_URL", raising=False)


@pytest.fixture
def llm(monkeypatch):
    """llm_complete factice qui enregistre le modèle de chaque appel."""
    calls = []

    def fake(prompt, model="", api_key="", provider="", **k):
        calls.append({"model": model, "prompt": prompt})
        return "## 📍 Chapitrage Temporel\n| [00:01] | *S* | D |"
    monkeypatch.setattr(main, "llm_complete", fake)
    return calls


@pytest.fixture
def transcriber(monkeypatch):
    calls = []

    def fake(src, api_key, model, timeout=None, provider="openrouter"):
        calls.append({"src": src, "key": api_key, "model": model, "provider": provider})
        return dict(LLM_DATA, model=model)
    monkeypatch.setattr(main, "transcribe_youtube_via_llm", fake)
    return calls


# ---------- _summarize : repli ----------

def test_fallback_used_when_no_subtitles(full, llm, transcriber, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    out = main._summarize(URL, api_key="sk-or-client", provider="openrouter", transcription_llm=True,
                          modeles=main.resolve_models({"transcription_video": "google/gemini-2.5-pro"}, "m/r"))
    assert transcriber == [{"src": URL, "key": "sk-or-client", "model": "google/gemini-2.5-pro", "provider": "openrouter"}]
    assert out["transcription"] == "llm-video" and out["modele_transcription"] == "google/gemini-2.5-pro"
    assert out["langue_source"] == "unknown" and out["titre"] == "Me at the zoo"
    assert ("Transcription générée par google/gemini-2.5-pro (pas de sous-titres) : horodatages approximatifs."
            in out["avertissements"])
    assert out["modeles_utilises"]["transcription_video"] == "google/gemini-2.5-pro"


def test_subtitles_path_unchanged(full, llm, transcriber, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", lambda u: dict(SUBS))
    out = main._summarize(URL, api_key="k", provider="openrouter", transcription_llm=True)
    assert transcriber == [] and out["transcription"] == "sous-titres"
    assert "modele_transcription" not in out
    assert not any("Transcription générée" in a for a in out["avertissements"])


def test_no_fallback_on_other_errors(full, llm, transcriber, monkeypatch):
    def bad(u):
        raise ValueError("URL YouTube invalide. Format: youtube.com/watch?v=ID ou youtu.be/ID")
    monkeypatch.setattr(main, "get_youtube_transcript", bad)
    with pytest.raises(ValueError, match="URL YouTube invalide"):
        main._summarize("x", api_key="k", provider="openrouter", transcription_llm=True)
    assert transcriber == []


def test_no_fallback_when_disabled(full, llm, transcriber, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    with pytest.raises(ValueError) as e:
        main._summarize(URL, api_key="k", provider="openrouter", transcription_llm=False)
    assert transcriber == [] and "Whisper" in str(e.value) and "Aucun transcript" in str(e.value)


@pytest.mark.parametrize("prov", ["openai", "opencode", "local", "custom"])
def test_no_fallback_for_other_providers(full, llm, transcriber, monkeypatch, prov):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-server")
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    with pytest.raises(ValueError):
        main._summarize(URL, api_key="k", provider=prov, transcription_llm=True)
    assert transcriber == []


def test_empty_provider_uses_env_openrouter_key(full, llm, transcriber, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-server")
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    main._summarize(URL, provider="", transcription_llm=True)
    assert transcriber[0]["key"] == "sk-or-server"


def test_empty_provider_without_any_key_gives_hint(full, llm, transcriber, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    with pytest.raises(ValueError, match="Whisper"):
        main._summarize(URL, provider="", transcription_llm=True)
    assert transcriber == []


def test_light_mode_server_key_not_used(llm, transcriber, monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.delenv("ALLOW_SERVER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-server")
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    with pytest.raises(ValueError):
        main._summarize(URL, provider="openrouter", transcription_llm=True)
    assert transcriber == []


def test_light_hint_mentions_option_for_other_provider(llm, transcriber, monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    with pytest.raises(ValueError) as e:
        main._summarize(URL, api_key="k", provider="openai", transcription_llm=True)
    msg = str(e.value)
    assert "Transcrire avec un modèle vidéo" in msg and "OpenRouter" in msg
    assert "sous-titres" in msg and "pas de transcription audio" in msg
    # Le conseil tient dans la limite de 300 caractères du gestionnaire d'erreurs.
    assert msg.index("Détail") < 290


def test_light_duration_limit_after_llm_transcription(llm, transcriber, monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)
    monkeypatch.setattr(main, "transcribe_youtube_via_llm", lambda *a, **k: dict(LLM_DATA, total_duration_minutes=45))
    with pytest.raises(ValueError, match="trop longue"):
        main._summarize(URL, api_key="k", provider="openrouter", transcription_llm=True)
    assert llm == []


def test_transcriber_error_propagates(full, llm, monkeypatch):
    monkeypatch.setattr(main, "get_youtube_transcript", _no_subs)

    def boom(*a, **k):
        raise ValueError("Crédits OpenRouter insuffisants pour transcrire la vidéo (402) : rechargez votre compte OpenRouter.")
    monkeypatch.setattr(main, "transcribe_youtube_via_llm", boom)
    with pytest.raises(ValueError, match="402"):
        main._summarize(URL, api_key="k", provider="openrouter", transcription_llm=True)


# ---------- modèle par tâche réellement utilisé ----------

def test_chunks_use_resume_model_and_fusion_uses_fusion_model(full, llm, monkeypatch):
    monkeypatch.setattr(main, "chunk_transcript", lambda *a, **k: [{"text": "a", "tokens": 1}, {"text": "b", "tokens": 1}])
    m = main.resolve_models({"resume": "x/resume", "fusion": "y/fusion"}, "d/d")
    out = main._summarize_data(dict(SUBS), "Français", "d/d", "k", "openrouter", modeles=m)
    assert [c["model"] for c in llm] == ["x/resume", "x/resume", "y/fusion"]
    assert out["modeles_utilises"] == {"resume": "x/resume", "fusion": "y/fusion"}


def test_single_chunk_no_fusion_reported(full, llm):
    out = main._summarize_data(dict(SUBS), "Français", "d/d", "k", "openrouter")
    assert [c["model"] for c in llm] == ["d/d"]
    assert out["modeles_utilises"] == {"resume": "d/d"} and out["transcription"] == "sous-titres"


def test_resumer_route_passes_modeles_and_flag(full, monkeypatch):
    seen = {}

    def fake(url, langue, modele, api_key="", provider="", modeles=None, transcription_llm=False, **k):
        seen.update(modeles=modeles, flag=transcription_llm)
        return dict(R, langue_source="fr", avertissements=[])
    monkeypatch.setattr(main, "_summarize", fake)
    r = client().post("/resumer", json={"url": URL, "api_key": "k", "modele": "d/d",
                                        "modeles": {"fusion": "f/f", "pirate": "x"}})
    assert r.status_code == 200
    assert seen["flag"] is True and seen["modeles"]["fusion"] == "f/f" and seen["modeles"]["resume"] == "d/d"
    client().post("/resumer", json={"url": URL, "api_key": "k", "transcription_llm": False})
    assert seen["flag"] is False


@pytest.mark.parametrize("route,body", [
    ("/resumer", {"url": URL, "api_key": "k"}),
    ("/resumer-lot", {"urls": [URL], "api_key": "k"}),
    ("/qa", {"titre": "t", "resume": "r", "question": "q", "api_key": "k"}),
    ("/image", {"resultat": R, "api_key": "k"}),
])
@pytest.mark.parametrize("bad", [{"resume": "a b"}, {"qa": "<x>"}, {"image": "x" * 101}, {"transcription_video": "openai/gpt"}])
def test_invalid_modeles_400(full, monkeypatch, route, body, bad):
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: pytest.fail("ne doit pas être appelé"))
    monkeypatch.setattr(main, "llm_complete", lambda *a, **k: pytest.fail("ne doit pas être appelé"))
    r = client().post(route, json=dict(body, modeles=bad))
    assert r.status_code == 400 and "odèle" in r.json()["detail"]


def test_invalid_modele_field_400(full, monkeypatch):
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: pytest.fail("ne doit pas être appelé"))
    assert client().post("/resumer", json={"url": URL, "api_key": "k", "modele": "a;b"}).status_code == 400


def test_resumer_lot_passes_modeles(full, monkeypatch):
    seen = []
    monkeypatch.setattr(main, "_summarize", lambda u, l, m, api_key="", provider="", modeles=None, transcription_llm=False, **k:
                        seen.append((modeles["resume"], transcription_llm)) or dict(R))
    r = client().post("/resumer-lot", json={"urls": [URL, URL], "api_key": "k", "modeles": {"resume": "r/r"},
                                            "transcription_llm": False})
    assert r.status_code == 200 and seen == [("r/r", False), ("r/r", False)]


def test_qa_uses_qa_model(full, monkeypatch):
    seen = []
    monkeypatch.setattr(main, "llm_complete", lambda p, model="", **k: seen.append(model) or "rep")
    r = client().post("/qa", json={"titre": "t", "resume": "r", "question": "q", "api_key": "k", "modele": "d/d",
                                   "modeles": {"resume": "x/r", "qa": "q/q"}})
    assert r.status_code == 200 and seen == ["q/q"]
    client().post("/qa", json={"titre": "t", "resume": "r", "question": "q", "api_key": "k", "modele": "d/d",
                               "modeles": {"resume": "x/r"}})
    assert seen[-1] == "x/r"


def test_image_uses_image_model(full, monkeypatch):
    seen = []
    monkeypatch.setattr(main, "get_capabilities", lambda: {"images": {"ok": True, "raison": ""}})
    monkeypatch.setattr(image_generator, "generate_image", lambda prompt, key, model="": seen.append(model) or b"png")
    r = client().post("/image", json={"resultat": R, "api_key": "k", "modeles": {"image": "black-forest-labs/flux.2-pro"}})
    assert r.status_code == 200 and seen == ["black-forest-labs/flux.2-pro"]
    client().post("/image", json={"resultat": R, "api_key": "k", "modele": "recraft/recraft-v4.1"})
    assert seen[-1] == "recraft/recraft-v4.1"
    client().post("/image", json={"resultat": R, "api_key": "k"})
    assert seen[-1] == image_generator.DEFAULT_MODEL


# ---------- /resumer-fichier : champ modeles (JSON) ----------

@pytest.fixture
def fichier(full, monkeypatch):
    monkeypatch.setattr(main, "get_capabilities", lambda: {k: {"ok": True, "raison": ""} for k in ("whisper", "resume")})
    import lib.whisper_transcriber as wt
    seen = {}

    def tf(path, language="", model_size=""):
        seen["size"] = model_size
        return [{"text": "x", "start": 0.0, "duration": 1.0}], model_size or "base"
    monkeypatch.setattr(wt, "transcribe_file_with_model", tf)

    def sd(data, langue, modele, api_key="", provider="", modeles=None):
        seen["modeles"] = modeles
        seen["data"] = data
        return dict(R)
    monkeypatch.setattr(main, "_summarize_data", sd)
    return seen


def _upload(data):
    return client().post("/resumer-fichier", files={"fichier": ("a.mp3", b"x" * 10)}, data=dict({"api_key": "k"}, **data))


def test_fichier_modeles_json(fichier):
    r = _upload({"modeles": json.dumps({"transcription_audio": "small", "resume": "r/r", "fusion": "f/f"})})
    assert r.status_code == 200
    assert fichier["size"] == "small" and fichier["modeles"]["fusion"] == "f/f"
    assert fichier["data"]["source"] == "whisper"


def test_fichier_without_modeles_keeps_default(fichier):
    assert _upload({}).status_code == 200
    assert fichier["size"] == "" and fichier["modeles"]["resume"]


@pytest.mark.parametrize("val", ["{pas du json", "[1,2]", json.dumps({"transcription_audio": "huge"}),
                                 json.dumps({"resume": "a b"}), json.dumps({"x": "y" * 5000})])
def test_fichier_modeles_invalid_400(fichier, val):
    r = _upload({"modeles": val})
    assert r.status_code == 400
    assert "size" not in fichier


# ---------- /modeles?tache= ----------

class Rsp:
    def __init__(self, data):
        self.status_code = 200
        self._d = data

    def json(self):
        return {"data": self._d}


@pytest.fixture
def get(monkeypatch):
    calls = []
    data = [{"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash"}, {"id": "openai/gpt-4o-video"},
            {"id": "google/gemini-2.5-pro"}]

    def fake(url, headers=None, timeout=0):
        calls.append({"url": url, "headers": headers})
        return Rsp(data)
    monkeypatch.setattr(httpx, "get", fake)
    return calls


def test_modeles_get_tache_video(full, get):
    r = client().get("/modeles?provider=openrouter&tache=transcription_video")
    j = r.json()
    assert r.status_code == 200 and j["models"] == ["google/gemini-2.5-flash", "google/gemini-2.5-pro"]
    assert j["source"] == "api" and j["tache"] == "transcription_video" and j["noms"]["google/gemini-2.5-flash"]
    assert get[0]["url"].endswith("/models?input_modalities=video")
    client().get("/modeles?provider=openrouter&tache=transcription_video")
    assert len(get) == 1  # cache


def test_modeles_post_tache_image_key_in_header_only(full, get):
    r = client().post("/modeles", json={"provider": "openrouter", "api_key": "sk-or-x", "tache": "image"})
    assert r.status_code == 200 and r.json()["tache"] == "image"
    assert get[0]["url"] == "https://openrouter.ai/api/v1/models?output_modalities=image"
    assert "sk-or-x" not in get[0]["url"] and get[0]["headers"] == {"Authorization": "Bearer sk-or-x"}


@pytest.mark.parametrize("tache", ["resume", "fusion", "qa"])
def test_modeles_text_tasks(full, get, tache):
    j = client().get(f"/modeles?provider=openrouter&tache={tache}").json()
    assert get[0]["url"] == "https://openrouter.ai/api/v1/models" and len(j["models"]) == 3


def test_modeles_audio_task(full, get):
    j = client().get("/modeles?provider=openrouter&tache=transcription_audio").json()
    assert j["models"] == ["tiny", "base", "small", "medium", "large"] and not get


def test_modeles_unknown_task_400(full, get):
    assert client().get("/modeles?tache=pirate").status_code == 400
    assert client().post("/modeles", json={"tache": "pirate"}).status_code == 400


def test_modeles_without_tache_unchanged(full, get):
    assert client().get("/modeles?provider=openrouter").json() == {
        "models": ["google/gemini-2.5-flash", "google/gemini-2.5-pro", "openai/gpt-4o-video"], "source": "api"}


def test_modeles_other_provider_video_empty_with_reason(full, get):
    j = client().post("/modeles", json={"provider": "openai", "api_key": "k", "tache": "transcription_video"}).json()
    assert j["models"] == [] and "OpenRouter" in j["raison"] and not get


def test_modeles_local_still_gated_in_light(monkeypatch, get):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    assert client().get("/modeles?provider=local&tache=transcription_audio").status_code == 403


# ---------- sécurité des routes nouvelles / modifiées ----------

EVIL = {"Origin": "http://evil.com"}


@pytest.mark.parametrize("path,kw", [
    ("/modeles", {"json": {"provider": "openrouter", "tache": "image", "api_key": "k"}}),
    ("/resumer", {"json": {"url": URL, "api_key": "k", "modeles": {"resume": "a/b"}}}),
    ("/resumer-lot", {"json": {"urls": [URL], "api_key": "k", "modeles": {"resume": "a/b"}}}),
    ("/qa", {"json": {"titre": "t", "resume": "r", "question": "q", "modeles": {"qa": "a/b"}}}),
    ("/image", {"json": {"resultat": R, "modeles": {"image": "a/b"}}}),
    ("/resumer-fichier", {"files": {"fichier": ("a.mp3", b"x")}, "data": {"modeles": "{}"}}),
])
def test_evil_origin_blocked(full, monkeypatch, path, kw):
    monkeypatch.setattr(main, "_summarize", lambda *a, **k: pytest.fail("appelé"))
    monkeypatch.setattr(main, "transcribe_youtube_via_llm", lambda *a, **k: pytest.fail("appelé"))
    r = client().post(path, headers=EVIL, **kw)
    assert r.status_code == 403


def test_capabilities_expose_transcription_video(full):
    assert client().get("/sante").json()["capacites"]["transcription_video"]["ok"] is True
