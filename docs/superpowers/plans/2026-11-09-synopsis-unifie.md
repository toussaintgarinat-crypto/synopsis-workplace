# Synopsis unifié — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Une seule application FastAPI (front orange de `localhost:6090`) qui regroupe les fonctions de `synopsis-workplace`, `youtube-summarizer` et `synopsis-jet.vercel.app`, avec un mode complet (local/Docker) et un mode léger (Vercel).

**Architecture:** `main.py` garde les routes et la page unique. Chaque fonction vit dans un module de `lib/`. `lib/capabilities.py` détecte ce qui est disponible ; `/sante` l'expose et le front grise les boutons indisponibles.

**Tech Stack:** Python 3.14, FastAPI, httpx, pytest, fpdf2 (non utilisé pour le PDF : impression navigateur), yt-dlp, openai-whisper (optionnel), ffmpeg (optionnel).

## Global Constraints

- Front conservé : la page intégrée à `main.py` (thème orange). Pas de nouveau framework front.
- Aucune clé API dans le code ni dans git : `.env` reste dans `.gitignore`.
- Mode léger (variable `SYNOPSIS_MODE=light`) : sous-titres YouTube uniquement, vidéos ≤ 30 min (`MAX_MINUTES_LIGHT=30`).
- Les JS inclus dans `main.py` (chaîne Python non raw) ne doivent pas contenir de `\` sans le doubler.
- Pas de push vers GitHub sans accord explicite de l'utilisateur.
- Applications natives (PyInstaller, `.app`) et PWA : hors périmètre.

## File Structure

| Fichier | Responsabilité |
|---|---|
| `lib/capabilities.py` (créer) | `get_capabilities() -> dict[str, dict]` |
| `lib/llm_client.py` (modifier) | fournisseur `local`, `list_local_models()` |
| `lib/exporters/__init__.py`, `markdown.py`, `obsidian.py`, `excalidraw.py`, `drive.py` (créer) | exports |
| `lib/qa.py` (créer) | `answer_question(...)` |
| `lib/batch.py` (créer) | `summarize_many(...)` |
| `lib/whisper_transcriber.py` (créer) | transcription audio |
| `lib/image_generator.py` (créer) | images à partir du résumé |
| `lib/i18n.py` (créer) | textes FR/EN servis au front |
| `main.py` (modifier) | routes + front |
| `tests/` (créer) | un fichier de tests par module |
| `vercel.json`, `api/index.py` (créer) | déploiement mode léger |

---

### Task 1: Base de tests et extraction des chapitres

**Files:**
- Create: `tests/__init__.py`, `tests/test_fusion.py`
- Modify: `requirements.txt` (ajouter `pytest`)

**Interfaces:**
- Produces: `lib.fusion._extract_chapters(str) -> list[{'timestamp','ts_seconds','subject','description'}]`, `_extract_insights(str) -> list[{'title','timestamp','description'}]` (déjà existants).

- [ ] **Step 1: Écrire les tests**

```python
# tests/test_fusion.py
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
```

- [ ] **Step 2: Lancer**
Run: `cd /Users/garinat_t/Desktop/synopsis-workplace && pip install pytest && python3 -m pytest tests/test_fusion.py -v`
Expected: FAIL sur `1:02:03` (format HH:MM:SS non géré). Corriger `row_pattern` et `item_pattern` dans `lib/fusion.py` pour accepter `(\d{1,3}(?::\d{2}){1,2})` et calculer `ts_seconds` avec les heures (3 parties → h*3600+m*60+s), puis relancer : PASS (3 tests).

- [ ] **Step 3: Commit**
```bash
git init 2>/dev/null; git add tests requirements.txt lib && git commit -m "test: couverture extraction chapitres/insights"
```

---

### Task 2: Détection des capacités + mode léger

**Files:**
- Create: `lib/capabilities.py`, `tests/test_capabilities.py`
- Modify: `main.py` (route `/sante`)

**Interfaces:**
- Produces: `get_capabilities() -> dict[str, dict]` ; chaque entrée = `{"ok": bool, "raison": str}`. Clés : `resume`, `whisper`, `reel`, `llm_local`, `obsidian`, `drive`, `excalidraw`, `images`, `tts`.
- `is_light_mode() -> bool`

- [ ] **Step 1: Test**
```python
# tests/test_capabilities.py
from lib import capabilities as c

def test_light_mode_disables_heavy(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    caps = c.get_capabilities()
    assert caps["resume"]["ok"] is True
    for k in ("whisper", "reel", "llm_local", "obsidian", "drive", "images", "tts"):
        assert caps[k]["ok"] is False
        assert "en ligne" in caps[k]["raison"]

def test_full_mode_reports_ffmpeg(monkeypatch):
    monkeypatch.delenv("SYNOPSIS_MODE", raising=False)
    monkeypatch.setattr(c.shutil, "which", lambda name: None)
    caps = c.get_capabilities()
    assert caps["reel"]["ok"] is False
    assert "ffmpeg" in caps["reel"]["raison"]
```
- [ ] **Step 2:** Run `python3 -m pytest tests/test_capabilities.py -v` → FAIL (module absent).
- [ ] **Step 3: Implémenter**
```python
# lib/capabilities.py
"""Détecte les fonctions disponibles selon le mode et l'environnement."""
import importlib.util
import os
import shutil

LIGHT_REASON = "indisponible en ligne (mode léger)"
HEAVY = ("whisper", "reel", "llm_local", "obsidian", "drive", "excalidraw", "images", "tts")


def is_light_mode() -> bool:
    return os.getenv("SYNOPSIS_MODE", "full").lower() == "light"


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def get_capabilities() -> dict:
    caps = {"resume": {"ok": True, "raison": ""}}
    if is_light_mode():
        for k in HEAVY:
            caps[k] = {"ok": False, "raison": LIGHT_REASON}
        return caps
    ffmpeg = shutil.which("ffmpeg") is not None
    caps["reel"] = {"ok": ffmpeg, "raison": "" if ffmpeg else "nécessite ffmpeg"}
    caps["whisper"] = {"ok": ffmpeg and (_has("whisper") or bool(os.getenv("OPENAI_API_KEY"))),
                       "raison": "nécessite ffmpeg et openai-whisper ou OPENAI_API_KEY"}
    caps["llm_local"] = {"ok": True, "raison": ""}
    caps["obsidian"] = {"ok": True, "raison": ""}
    caps["excalidraw"] = {"ok": True, "raison": ""}
    caps["drive"] = {"ok": bool(os.getenv("GOOGLE_DRIVE_CREDENTIALS")), "raison": "nécessite GOOGLE_DRIVE_CREDENTIALS"}
    caps["images"] = {"ok": True, "raison": ""}
    caps["tts"] = {"ok": _has("gtts"), "raison": "nécessite gtts"}
    for v in caps.values():
        if v["ok"]:
            v["raison"] = ""
    return caps
```
- [ ] **Step 4:** Dans `main.py`, remplacer le début de `sante()` : ajouter `from lib.capabilities import get_capabilities, is_light_mode` en haut, et `checks = {"api": "ok", "version": "1.0.0", "mode": "light" if is_light_mode() else "full", "capacites": get_capabilities()}`.
- [ ] **Step 5:** Run `python3 -m pytest tests -v` → PASS ; relancer le serveur et `curl -s localhost:6090/sante` doit contenir `"capacites"`.
- [ ] **Step 6: Commit** `git add lib/capabilities.py tests main.py && git commit -m "feat: détection des capacités et mode léger"`

---

### Task 3: Front — grisage selon les capacités

**Files:** Modify: `main.py` (JS du front)

**Interfaces:** Consumes: `GET /sante` → `capacites`. Produces: JS `CAPS` global et `capOk(nom) -> bool`.

- [ ] **Step 1:** Dans le `<script>` de `main.py`, avant `init()`, ajouter (sans backslash) :
```javascript
var CAPS = {};
function capOk(nom) { return !!(CAPS[nom] && CAPS[nom].ok); }
async function chargerCapacites() {
  try {
    var r = await fetch('/sante');
    var d = await r.json();
    CAPS = d.capacites || {};
  } catch (e) { CAPS = {}; }
  var els = document.querySelectorAll('[data-cap]');
  for (var i = 0; i < els.length; i++) {
    var nom = els[i].getAttribute('data-cap');
    if (!capOk(nom)) {
      els[i].disabled = true;
      els[i].title = (CAPS[nom] && CAPS[nom].raison) || 'Indisponible';
      els[i].classList.add('indispo');
    }
  }
}
chargerCapacites();
```
Ajouter dans le CSS : `.indispo { opacity: .45; cursor: not-allowed; }`. Ajouter `data-cap="reel"` au bouton `#btn-reel`.
- [ ] **Step 2:** Lancer le serveur avec `SYNOPSIS_MODE=light` ; `curl -s localhost:6090/ | grep -c 'data-cap="reel"'` → `1`. Ouvrir la page : le bouton reel est grisé avec l'infobulle.
- [ ] **Step 3: Commit** `git commit -am "feat(front): boutons grisés selon capacités"`

---

### Task 4: Fournisseur LLM local (Ollama / LM Studio)

**Files:** Modify: `lib/llm_client.py`, `main.py` (`/modeles`, front) ; Create: `tests/test_llm_local.py`

**Interfaces:**
- Produces: `llm_complete(..., provider="local")` ; `list_local_models(base_url: str = "") -> list[str]` ; variable `LOCAL_LLM_URL` (défaut `http://localhost:11434`).

- [ ] **Step 1: Test**
```python
# tests/test_llm_local.py
import httpx
from lib import llm_client as m

def test_list_local_models(monkeypatch):
    class R:
        status_code = 200
        def json(self): return {"models": [{"name": "llama3.2"}, {"name": "qwen3"}]}
    monkeypatch.setattr(m.httpx, "get", lambda url, timeout=5: R())
    assert m.list_local_models("http://x:11434") == ["llama3.2", "qwen3"]

def test_list_local_models_down(monkeypatch):
    def boom(url, timeout=5): raise httpx.ConnectError("no")
    monkeypatch.setattr(m.httpx, "get", boom)
    assert m.list_local_models() == []
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implémenter** dans `lib/llm_client.py` :
```python
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434").rstrip("/")


def list_local_models(base_url: str = "") -> list[str]:
    base = (base_url or LOCAL_LLM_URL).rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []
```
Dans `llm_complete`, avant `if provider and api_key:` ajouter :
```python
    if provider == "local":
        return _complete_openai_compatible(
            f"{LOCAL_LLM_URL}/v1/chat/completions",
            prompt, model, system, temperature, api_key="ollama",
        )
```
- [ ] **Step 4:** Dans `main.py`, route `/modeles` : au début, `if provider == "local": return {"models": list_local_models(), "source": "local"}` (importer `list_local_models`). Dans le front : ajouter l'option `local` dans `PROVIDERS` avec `keyHint: 'Aucune clé nécessaire'` et dans `resumer()` ne pas exiger la clé quand `provider === 'local'` ; ajouter `data-cap="llm_local"` à l'`<option value="local">`.
- [ ] **Step 5:** `python3 -m pytest tests -v` → PASS.
- [ ] **Step 6: Commit** `git commit -am "feat: fournisseur LLM local (Ollama)"`

---

### Task 5: Export Markdown

**Files:** Create: `lib/exporters/__init__.py` (vide), `lib/exporters/markdown.py`, `tests/test_export_markdown.py` ; Modify: `main.py`

**Interfaces:** Produces: `to_markdown(d: dict) -> str` (`d` = réponse de `/resumer`) ; JS `exporterMarkdown()`.

- [ ] **Step 1: Test**
```python
# tests/test_export_markdown.py
from lib.exporters.markdown import to_markdown

D = {"titre": "Vidéo", "langue_source": "fr", "resume": "Texte",
     "chapitres": [{"timestamp": "01:30", "subject": "Intro", "description": "d"}],
     "insights": [{"title": "T", "timestamp": "02:10", "description": "e"}]}

def test_markdown():
    md = to_markdown(D)
    assert md.startswith("# Vidéo")
    assert "- **01:30** Intro — d" in md
    assert "1. **T** [02:10] : e" in md
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:**
```python
# lib/exporters/markdown.py
def to_markdown(d: dict) -> str:
    lines = [f"# {d.get('titre', 'Synopsis')}", f"*Langue source : {d.get('langue_source', '?')}*", "", "## Résumé", d.get("resume", ""), ""]
    if d.get("chapitres"):
        lines.append("## Chapitres")
        lines += [f"- **{c.get('timestamp','')}** {c.get('subject','')} — {c.get('description','')}" for c in d["chapitres"]]
        lines.append("")
    if d.get("insights"):
        lines.append("## Insights")
        lines += [f"{i}. **{x.get('title','')}** [{x.get('timestamp','')}] : {x.get('description','')}" for i, x in enumerate(d["insights"], 1)]
    return "\n".join(lines) + "\n"
```
- [ ] **Step 4:** Route `POST /export/markdown` (corps = dict du résumé, retourne `PlainTextResponse(to_markdown(d), media_type="text/markdown")`). Front : bouton « Enregistrer en Markdown » à côté des deux autres dans `afficherResume`, qui poste `dernierResume` à `/export/markdown` et télécharge le blob en `.md` (même méthode que `exporterHtml`).
- [ ] **Step 5:** tests PASS ; commit `git commit -am "feat: export Markdown"`

---

### Task 6: Plusieurs vidéos

**Files:** Create: `lib/batch.py`, `tests/test_batch.py` ; Modify: `main.py`

**Interfaces:** Produces: `summarize_many(urls: list[str], summarize: Callable[[str], dict]) -> list[dict]` ; chaque élément = `{"url", "ok": True, "resultat": {...}}` ou `{"url", "ok": False, "erreur": str}`. Route `POST /resumer-lot` `{urls: [str], langue, modele, api_key, provider}`.

- [ ] **Step 1: Test**
```python
# tests/test_batch.py
from lib.batch import summarize_many

def test_continues_after_error():
    def fake(url):
        if "bad" in url: raise ValueError("boom")
        return {"titre": url}
    out = summarize_many(["a", "bad", "c"], fake)
    assert [o["ok"] for o in out] == [True, False, True]
    assert out[1]["erreur"] == "boom"
```
- [ ] **Step 2:** FAIL. **Step 3:**
```python
# lib/batch.py
def summarize_many(urls, summarize):
    out = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            out.append({"url": url, "ok": True, "resultat": summarize(url)})
        except Exception as e:
            out.append({"url": url, "ok": False, "erreur": str(e)})
    return out
```
- [ ] **Step 4:** Route dans `main.py` : modèle `LotRequest(BaseModel)` avec `urls: list[str]` + mêmes champs que `ResumerRequest`, puis `return summarize_many(req.urls, lambda u: _summarize(u, req.langue, req.modele, api_key=req.api_key, provider=req.provider))`. Front : `<details>` « Résumer plusieurs vidéos » avec `<textarea id="urls-multi">` et bouton ; chaque résultat est rendu par `afficherResume` dans son propre conteneur, les erreurs en bloc `.error`.
- [ ] **Step 5:** tests PASS ; commit `git commit -am "feat: résumé de plusieurs vidéos"`

---

### Task 7: Questions-réponses sur la vidéo

**Files:** Create: `lib/qa.py`, `tests/test_qa.py` ; Modify: `main.py`

**Interfaces:** Produces: `build_qa_prompt(titre: str, resume: str, question: str) -> str`, `answer_question(titre, resume, question, llm) -> str` où `llm: Callable[[str], str]`. Route `POST /qa` `{titre, resume, question, modele, api_key, provider}` → `{"reponse": str}`.

- [ ] **Step 1: Test**
```python
# tests/test_qa.py
from lib.qa import build_qa_prompt, answer_question

def test_prompt_contains_parts():
    p = build_qa_prompt("T", "R", "Q?")
    assert "T" in p and "R" in p and "Q?" in p

def test_answer_uses_llm():
    assert answer_question("T", "R", "Q?", lambda p: "ok") == "ok"
```
- [ ] **Step 2:** FAIL. **Step 3:**
```python
# lib/qa.py
def build_qa_prompt(titre, resume, question):
    return (
        "Réponds uniquement à partir du résumé ci-dessous. Si l'information n'y est pas, dis-le.\n\n"
        f"Vidéo : {titre}\n\nRésumé :\n{resume}\n\nQuestion : {question}\n"
    )


def answer_question(titre, resume, question, llm):
    return llm(build_qa_prompt(titre, resume, question)).strip()
```
- [ ] **Step 4:** Route `/qa` : `answer_question(req.titre, req.resume, req.question, lambda p: llm_complete(p, model=req.modele or MODEL, api_key=req.api_key, provider=req.provider))`. Front : champ de question sous le résumé, appel `fetch('/qa')`, réponse affichée via `escapeHtml`.
- [ ] **Step 5:** PASS ; commit `git commit -am "feat: questions-réponses"`

---

### Task 8: Interface FR / EN

**Files:** Create: `lib/i18n.py`, `tests/test_i18n.py` ; Modify: `main.py`

**Interfaces:** Produces: `TEXTS: dict[str, dict[str, str]]` (clés `fr`, `en`), route `GET /i18n/{lang}` → dict ; JS `setLangue(l)` qui remplace le texte des éléments `[data-t]`.

- [ ] **Step 1: Test**
```python
# tests/test_i18n.py
from lib.i18n import TEXTS

def test_same_keys():
    assert set(TEXTS["fr"]) == set(TEXTS["en"])
```
- [ ] **Step 2:** FAIL. **Step 3:** créer `lib/i18n.py` avec `TEXTS = {"fr": {...}, "en": {...}}` contenant au minimum les clés : `resumer`, `chapitres`, `insights`, `resume`, `export_html`, `export_md`, `export_pdf`, `question`, `plusieurs`, `config`, `langue_source` (valeurs FR et EN correspondantes, ex. `"resumer": "Résumer"` / `"Summarize"`).
- [ ] **Step 4:** Route `/i18n/{lang}` (404 si langue inconnue). Front : boutons FR/EN dans l'en-tête, ajouter `data-t="<clé>"` aux libellés, `setLangue` charge `/i18n/<lang>`, remplace `textContent`, mémorise le choix dans `localStorage` (dans un `try/catch`).
- [ ] **Step 5:** PASS ; commit `git commit -am "feat: interface FR/EN"`

---

### Task 9: Transcription Whisper et upload audio/vidéo

**Files:** Create: `lib/whisper_transcriber.py`, `tests/test_whisper.py` ; Modify: `main.py`, `requirements-full.txt` (créer : `openai-whisper`, `python-multipart`)

**Interfaces:** Produces: `transcribe_file(path: str, language: str = "") -> list[dict]` (même format que l'extracteur : `{'text','start','duration'}`) ; `segments_to_transcript(segments: list[dict]) -> list[dict]`. Route `POST /resumer-fichier` (multipart : `fichier`, `langue`, `modele`, `api_key`, `provider`), réutilise `_summarize` via `data = {"transcript": ..., "title": nom_fichier, "language": ...}`.

- [ ] **Step 1: Refactor** : dans `main.py`, scinder `_summarize` en `_summarize(url, ...)` (extrait puis appelle) et `_summarize_data(data, output_language, model, api_key, provider)` (tout le reste). Ajouter un test qui vérifie que `_summarize_data` accepte un dict `{"transcript": [...], "title": "t", "language": "fr"}` avec `llm_complete` patché.
- [ ] **Step 2: Test**
```python
# tests/test_whisper.py
from lib.whisper_transcriber import segments_to_transcript

def test_segments():
    segs = [{"start": 0.0, "end": 2.5, "text": " Bonjour"}]
    assert segments_to_transcript(segs) == [{"text": "Bonjour", "start": 0.0, "duration": 2.5}]
```
- [ ] **Step 3:**
```python
# lib/whisper_transcriber.py
"""Transcription locale (openai-whisper) avec repli sur l'API OpenAI."""
import os


def segments_to_transcript(segments):
    return [{"text": s["text"].strip(), "start": float(s["start"]), "duration": float(s["end"]) - float(s["start"])}
            for s in segments if s["text"].strip()]


def transcribe_file(path, language=""):
    try:
        import whisper
        model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
        result = model.transcribe(path, language=language or None)
        return segments_to_transcript(result["segments"])
    except ImportError:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("Whisper indisponible : installez openai-whisper ou définissez OPENAI_API_KEY")
        import httpx
        with open(path, "rb") as f:
            r = httpx.post("https://api.openai.com/v1/audio/transcriptions",
                           headers={"Authorization": f"Bearer {key}"},
                           files={"file": f}, data={"model": "whisper-1", "response_format": "verbose_json"}, timeout=600)
        r.raise_for_status()
        return segments_to_transcript(r.json()["segments"])
```
- [ ] **Step 4:** Route multipart (`UploadFile`), écrit dans le dossier temporaire, appelle `transcribe_file`, supprime le fichier, puis `_summarize_data`. Retourne 403 si `not get_capabilities()["whisper"]["ok"]`. Front : champ fichier dans un `<details>`, `data-cap="whisper"`.
- [ ] **Step 5:** PASS ; commit `git commit -am "feat: Whisper et upload fichier"`

---

### Task 10: Exports Obsidian et Excalidraw

**Files:** Create: `lib/exporters/obsidian.py`, `lib/exporters/excalidraw.py`, `tests/test_exporters.py` ; Modify: `main.py`

**Interfaces:** Produces: `obsidian.note_content(d: dict, url: str = "") -> str` (frontmatter YAML + markdown), `obsidian.save_note(d, vault_dir, url="") -> str` (retourne le chemin), `excalidraw.to_excalidraw(d: dict) -> dict` (JSON `.excalidraw` : un rectangle par chapitre avec son texte). Routes `POST /export/obsidian` `{resultat, vault_dir, url}`, `POST /export/excalidraw`.

- [ ] **Step 1: Test**
```python
# tests/test_exporters.py
from lib.exporters.obsidian import note_content, save_note
from lib.exporters.excalidraw import to_excalidraw

D = {"titre": "Ma vidéo", "langue_source": "fr", "resume": "R",
     "chapitres": [{"timestamp": "00:10", "subject": "A", "description": "d"}], "insights": []}

def test_note_has_frontmatter():
    n = note_content(D, "https://youtu.be/x")
    assert n.startswith("---\n") and 'titre: "Ma vidéo"' in n and "source: https://youtu.be/x" in n

def test_save_note(tmp_path):
    p = save_note(D, str(tmp_path))
    assert p.endswith("Ma vidéo.md") and open(p, encoding="utf-8").read().startswith("---")

def test_excalidraw():
    e = to_excalidraw(D)
    assert e["type"] == "excalidraw"
    assert any(el["type"] == "text" and "A" in el["text"] for el in e["elements"])
```
- [ ] **Step 2:** FAIL. **Step 3:**
```python
# lib/exporters/obsidian.py
import re
from pathlib import Path
from .markdown import to_markdown


def note_content(d, url=""):
    fm = ["---", f'titre: "{d.get("titre", "")}"', f'langue: {d.get("langue_source", "")}']
    if url:
        fm.append(f"source: {url}")
    fm += ["tags: [synopsis, youtube]", "---", ""]
    return "\n".join(fm) + to_markdown(d)


def save_note(d, vault_dir, url=""):
    name = re.sub(r'[\\/:*?"<>|]', "", d.get("titre", "Synopsis")).strip() or "Synopsis"
    path = Path(vault_dir).expanduser() / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note_content(d, url), encoding="utf-8")
    return str(path)
```
```python
# lib/exporters/excalidraw.py
def to_excalidraw(d):
    elements, y = [], 0
    for i, c in enumerate(d.get("chapitres", [])):
        label = f'{c.get("timestamp", "")} {c.get("subject", "")}'
        elements.append({"id": f"r{i}", "type": "rectangle", "x": 0, "y": y, "width": 420, "height": 60})
        elements.append({"id": f"t{i}", "type": "text", "x": 12, "y": y + 18, "text": label, "fontSize": 20})
        y += 90
    return {"type": "excalidraw", "version": 2, "source": "synopsis", "elements": elements}
```
- [ ] **Step 4:** Routes ; l'export Excalidraw renvoie le JSON (le front le télécharge en `.excalidraw`). L'export Obsidian est refusé (403) si la capacité `obsidian` n'est pas ok. Front : boutons `data-cap="obsidian"` / `data-cap="excalidraw"`, champ « dossier du vault » mémorisé dans `localStorage`.
- [ ] **Step 5:** PASS ; commit `git commit -am "feat: exports Obsidian et Excalidraw"`

---

### Task 11: Export Google Drive et génération d'images

**Files:** Create: `lib/exporters/drive.py`, `lib/image_generator.py`, `tests/test_drive_images.py` ; Modify: `main.py`

**Interfaces:** Produces: `drive.upload_markdown(d: dict, credentials_json: str) -> str` (URL du fichier), `image_generator.build_image_prompt(d: dict) -> str`, `image_generator.generate_image(prompt: str, api_key: str, model: str) -> bytes`.

- [ ] **Step 1:** (Décision utilisateur : reprise fidèle, dépendances supplémentaires acceptées, ex. google-api-python-client dans `requirements-full.txt`.) Récupérer la logique d'origine : `gh api repos/toussaintgarinat-crypto/youtube-summarizer/contents/src/drive_exporter.py --jq .content | base64 -d` et `.../src/image_generator.py` ; l'implémentation reprend ces deux fichiers en remplaçant les imports `from src...` par `from lib...` et en exposant uniquement les signatures ci-dessus.
- [ ] **Step 2: Test** (sans réseau) :
```python
# tests/test_drive_images.py
from lib.image_generator import build_image_prompt

def test_prompt_uses_title_and_chapters():
    d = {"titre": "Titre", "chapitres": [{"subject": "A"}, {"subject": "B"}]}
    p = build_image_prompt(d)
    assert "Titre" in p and "A" in p and "B" in p
```
- [ ] **Step 3:** Implémenter `build_image_prompt` : `"Illustration pour la vidéo « {titre} » : " + ", ".join(sujets)`. `generate_image` appelle l'endpoint images OpenRouter/OpenAI avec `httpx`, décode le base64 retourné.
- [ ] **Step 4:** Routes `/export/drive` (403 si capacité `drive` non ok) et `/image` ; boutons front avec `data-cap="drive"` / `data-cap="images"`.
- [ ] **Step 5:** PASS ; commit `git commit -am "feat: export Drive et images"`

---

### Task 12: Mode léger — limite de durée

**Files:** Modify: `main.py` (`_summarize_data`), `tests/test_light_limit.py`

**Interfaces:** Consumes: `is_light_mode()`, `data["total_duration_minutes"]`. Produces: `ValueError("Vidéo trop longue pour la version en ligne (max 30 min). Utilisez la version locale.")`.

- [ ] **Step 1: Test**
```python
# tests/test_light_limit.py
import pytest
import main

def test_light_mode_rejects_long(monkeypatch):
    monkeypatch.setenv("SYNOPSIS_MODE", "light")
    data = {"transcript": [{"text": "x", "start": 0, "duration": 1}], "title": "t", "language": "fr", "total_duration_minutes": 45}
    with pytest.raises(ValueError, match="trop longue"):
        main._summarize_data(data, "Français", "", "", "")
```
- [ ] **Step 2:** FAIL. **Step 3:** au début de `_summarize_data` :
```python
    if is_light_mode() and data.get("total_duration_minutes", 0) > int(os.getenv("MAX_MINUTES_LIGHT", "30")):
        raise ValueError("Vidéo trop longue pour la version en ligne (max 30 min). Utilisez la version locale.")
```
- [ ] **Step 4:** PASS ; commit `git commit -am "feat: limite de durée en mode léger"`

---

### Task 13: Déploiement Vercel (mode léger) et Docker (mode complet)

**Files:** Create: `vercel.json`, `api/index.py`, `requirements-light.txt` ; Modify: `Dockerfile`, `docker-compose.yml`, `README.md`

**Interfaces:** Produces: déploiement `SYNOPSIS_MODE=light` sur Vercel ; image Docker avec ffmpeg en mode `full`.

- [ ] **Step 1:** `api/index.py` : `from main import app` (Vercel détecte l'objet ASGI `app`). `vercel.json` :
```json
{"rewrites": [{"source": "/(.*)", "destination": "/api/index"}], "functions": {"api/index.py": {"maxDuration": 60}}, "env": {"SYNOPSIS_MODE": "light"}}
```
`requirements-light.txt` : `youtube-transcript-api`, `tiktoken`, `httpx`, `fastapi`, `uvicorn` (sans yt-dlp, gtts, whisper).
- [ ] **Step 2:** `Dockerfile` : ajouter `apt-get install -y ffmpeg` et installer `requirements.txt` + `requirements-full.txt`. `docker-compose.yml` : ajouter `LOCAL_LLM_URL=http://host.docker.internal:11434` et `env_file: .env`.
- [ ] **Step 3:** `README.md` : documenter les deux modes, les variables (`SYNOPSIS_MODE`, `LOCAL_LLM_URL`, `WHISPER_MODEL`, `GOOGLE_DRIVE_CREDENTIALS`) et le lancement `set -a; . ./.env; set +a; python3 main.py`.
- [ ] **Step 4: Vérification** : `SYNOPSIS_MODE=light python3 main.py` puis `curl -s localhost:6090/sante` → `"mode":"light"` ; `python3 -m pytest tests -v` → tout PASS.
- [ ] **Step 5: Commit** `git add -A -- . ':!.env' && git commit -m "feat: déploiement Vercel (léger) et Docker (complet)"`

---

### Task 14: Mise en ligne sur GitHub (sur accord explicite uniquement)

**Files:** aucun

- [ ] **Step 1:** Demander à l'utilisateur la confirmation : push sur une branche de `toussaintgarinat-crypto/youtube-summarizer` (jamais directement sur `main`), en remplaçant l'application Streamlit/GUI.
- [ ] **Step 2:** Après accord : `git remote add origin https://github.com/toussaintgarinat-crypto/youtube-summarizer.git && git push -u origin HEAD:refs/heads/synopsis-unifie`, puis ouvrir une PR avec `gh pr create`.
- [ ] **Step 3:** Vérifier que `.env` n'est pas dans l'index : `git ls-files | grep -c '^.env$'` → `0`.

---

## Self-Review

- Couverture du design : capacités (T2-T3), exports HTML/PDF/Markdown (T5 + déjà fait), multi-vidéos (T6), Q&R (T7), FR/EN (T8), LLM local (T4), Whisper/upload (T9), Obsidian/Excalidraw (T10), Drive/images (T11), mode léger (T2, T12), Vercel/Docker (T13), push (T14). Reel, TTS et Gateway existent déjà dans `lib/` et sont grisés via T2.
- Les chapitres à zéro et les boutons HTML/PDF sont déjà corrigés dans `main.py` (hors plan).
- Cohérence des noms : `get_capabilities`, `is_light_mode`, `to_markdown`, `summarize_many`, `answer_question`, `_summarize_data` sont utilisés avec la même signature dans toutes les tâches.
