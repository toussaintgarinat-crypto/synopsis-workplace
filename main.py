"""Synopsis — YouTube Video Summarizer. Gateway-aware, standalone-capable."""

import functools
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from lib.extractor import get_youtube_transcript
from lib.chunker import chunk_transcript, estimate_tokens
from lib.fusion import (
    _extract_chapters, _extract_insights,
    merge_chapter_tables, select_top_insights,
    _build_chapter_table, _build_insights_list,
)
from lib.qa import answer_question
from lib.llm_client import llm_complete, list_local_models, list_custom_models
from lib.exporters.markdown import to_markdown
from lib.exporters.obsidian import save_note
from lib.exporters.excalidraw import to_excalidraw
from lib.batch import summarize_many
from lib.capabilities import get_capabilities, is_light_mode
from lib.i18n import TEXTS
from lib.exporters import drive as drive_exporter
from lib import image_generator
from lib import security
from lib.security import installer_masquage_journaux
from lib.extractor import extract_youtube_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("synopsis")

app = FastAPI(title="Synopsis API", version="1.0.0", description="YouTube video summarizer — Gateway-ready.")


def _allowed_origins() -> list[str]:
    """Origines autorisées (CORS + contrôle d'origine des requêtes qui modifient)."""
    port = os.getenv("PORT", "6090")
    brut = os.getenv("ALLOWED_ORIGINS") or "http://localhost:6090,http://127.0.0.1:6090"
    origines = [o.strip().rstrip("/") for o in brut.split(",") if o.strip()]
    for h in ("localhost", "127.0.0.1"):
        o = f"http://{h}:{port}"
        if o not in origines:
            origines.append(o)
    return origines


def _allowed_hosts() -> list[str]:
    hotes = ["localhost", "127.0.0.1", "host.docker.internal"]
    hotes += [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
    if os.getenv("VERCEL"):
        hotes.append("*.vercel.app")
    return hotes


installer_masquage_journaux()

# Ordre : chaque add_middleware enveloppe les précédents ; le dernier ajouté s'exécute en premier.
# Exécution : TrustedHost -> en-têtes -> contrôle d'origine -> taille du corps -> débit -> CORS -> routes.
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins(), allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def limite_debit(request: Request, call_next):
    """Limite par IP des routes qui appellent un LLM (mode léger, ou RATE_LIMIT_PER_MIN défini)."""
    if request.method == "POST" and request.url.path in security.LLM_PATHS:
        limite = security.limite_par_minute(is_light_mode())
        if limite and not security.RATE_LIMITER.autoriser(security.ip_client(request), limite):
            return JSONResponse({"detail": "Trop de requêtes : réessayez dans une minute."}, status_code=429,
                                headers={"Retry-After": "60"})
    return await call_next(request)


app.add_middleware(security.BodySizeLimit)


def _origine_autorisee(origine: str, request: Request) -> bool:
    origine = origine.strip().rstrip("/")
    if origine in _allowed_origins():
        return True
    # Même origine que l'hôte demandé (Host déjà filtré par TrustedHostMiddleware) :
    # comparaison exacte scheme://host[:port], sans chemin ni userinfo.
    hote = request.headers.get("host", "")
    return bool(hote) and origine in (f"http://{hote}", f"https://{hote}")


@app.middleware("http")
async def controle_origine(request: Request, call_next):
    """Refuse (403) les requêtes qui modifient l'état et viennent d'une autre origine web.

    - Sec-Fetch-Site (non falsifiable par une page web) : 'cross-site' / 'same-site' refusés,
      sauf origine explicitement autorisée pour 'same-site' ; 'same-origin' et 'none' passent.
    - Sinon Origin, à défaut Referer : doit être une origine autorisée ou la même origine.
    - Sans aucun de ces en-têtes (curl, appel serveur à serveur) la requête passe.
    """
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        sfs = (request.headers.get("sec-fetch-site") or "").strip().lower()
        origine = request.headers.get("origin")
        if origine is None:
            ref = request.headers.get("referer")
            if ref:
                u = urlparse(ref)
                origine = f"{u.scheme}://{u.netloc}" if u.scheme and u.netloc else ref
        refus = JSONResponse({"detail": "Origine non autorisée."}, status_code=403)
        if sfs == "cross-site":
            return refus
        if sfs == "same-site" and not (origine and origine.strip().rstrip("/") in _allowed_origins()):
            return refus
        if origine is not None:
            # « Origin: null » + Sec-Fetch-Site: same-origin = requête de la page elle-même
            # sous une politique Referrer stricte : acceptée (Sec-Fetch-* est infalsifiable par une page).
            null_meme_origine = sfs == "same-origin" and origine.strip() == "null"
            if not null_meme_origine and not _origine_autorisee(origine, request):
                return refus
    return await call_next(request)


@app.middleware("http")
async def en_tetes_securite(request: Request, call_next):
    response = await call_next(request)
    for k, v in security.EN_TETES.items():
        response.headers.setdefault(k, v)
    if request.url.path == "/":
        response.headers["content-security-policy"] = _csp_page()
    elif request.url.path not in ("/docs", "/redoc") and not request.url.path.startswith("/docs/"):
        response.headers.setdefault("content-security-policy", security.CSP_API)
    else:
        response.headers.setdefault("content-security-policy", "frame-ancestors 'none'")
    return response


# Ajouté en dernier = middleware le plus externe : l'hôte est vérifié en premier.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts())


@app.exception_handler(StarletteHTTPException)
async def http_exception_masquee(request: Request, exc: StarletteHTTPException):
    """Les messages d'erreur (souvent issus d'exceptions amont) sont tronqués et purgés des secrets."""
    detail = exc.detail
    if isinstance(detail, str):
        detail = security.scrub(detail, limite=300)
    return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))

PROMPTS_DIR = Path(__file__).parent / "prompts"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE_TOKENS", "12000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_TOKENS", "1200"))
MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")


class ResumerRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    langue: str = Field("Français", description="Summary output language")
    modele: str = Field("", description="LLM model override")
    api_key: str = Field("", description="API key")
    provider: str = Field("", description="Provider: openrouter, openai, opencode, local, custom")


class LotRequest(BaseModel):
    urls: list[str] = Field(..., description="YouTube video URLs")
    langue: str = Field("Français", description="Summary output language")
    modele: str = Field("", description="LLM model override")
    api_key: str = Field("", description="API key")
    provider: str = Field("", description="Provider: openrouter, openai, opencode, local")


class QaRequest(BaseModel):
    titre: str = Field(..., description="Video title")
    resume: str = Field(..., description="Video summary")
    question: str = Field(..., description="User question")
    modele: str = Field("", description="LLM model override")
    api_key: str = Field("", description="API key")
    provider: str = Field("", description="Provider: openrouter, openai, opencode, local")


class ReelRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    duree_clip: float = Field(45, ge=5, le=300, description="Clip duration per chapter (seconds)")
    sous_titres: bool = Field(True, description="Burn chapter subtitles")
    narration: bool = Field(False, description="Add TTS narration")
    langue_narration: str = Field("fr", description="Narration language")
    export_vertical: str = Field("", description="Vertical export mode: blur, crop, pad")
    api_key: str = Field("", description="API key")
    provider: str = Field("", description="Provider: openrouter, openai, opencode, local")


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _exiger(cap: str, caps: dict = None) -> dict:
    """Lève 403 si la capacité est indisponible. Renvoie les capacités (un seul calcul par requête)."""
    caps = caps if caps is not None else get_capabilities()
    c = caps.get(cap) or {"ok": False, "raison": "capacité inconnue"}
    if not c["ok"]:
        raise HTTPException(status_code=403, detail=f"Fonction indisponible ({cap}) : {c['raison']}")
    return caps


def _verifier_llm(provider: str, api_key: str, caps: dict = None) -> dict:
    """Contrôles communs aux routes qui appellent un LLM."""
    caps = caps if caps is not None else get_capabilities()
    if provider == "local":
        _exiger("llm_local", caps)
    elif provider == "custom":
        _exiger("llm_custom", caps)
    if is_light_mode() and not (api_key or "").strip() and os.getenv("ALLOW_SERVER_KEY") != "1":
        raise HTTPException(status_code=403, detail="Saisissez votre propre clé API : la clé du serveur n'est pas utilisable en version en ligne.")
    return caps


def _max_batch() -> int:
    if is_light_mode():
        return int(os.getenv("MAX_BATCH_LIGHT", "3"))
    return int(os.getenv("MAX_BATCH_FULL", "20"))


def _transcript_hint() -> str:
    if is_light_mode():
        return ("Cette version en ligne lit uniquement les sous-titres YouTube (pas de transcription audio) : "
                "essayez une vidéo qui en a, ou utilisez la version locale (Whisper).")
    return "Astuce : envoyez le fichier audio/vidéo pour une transcription Whisper."


def _summarize(url: str, output_language: str = "Français", model: str = "", api_key: str = "", provider: str = "") -> dict:
    logger.info("Resumer %s (lang=%s)", url, output_language)
    try:
        data = get_youtube_transcript(url)
    except ValueError as e:
        if "ranscript" in str(e):
            cause = " ".join(str(e).split())[:140].rstrip(" .:")
            raise ValueError(f"{_transcript_hint()} (Détail : {cause})") from e
        raise
    return _summarize_data(data, output_language, model, api_key, provider)


def _summarize_data(data: dict, output_language: str = "Français", model: str = "", api_key: str = "", provider: str = "") -> dict:
    max_min = int(os.getenv("MAX_MINUTES_LIGHT", "30"))
    if is_light_mode() and data.get("total_duration_minutes", 0) > max_min:
        raise ValueError(f"Vidéo trop longue pour la version en ligne (max {max_min} min). Utilisez la version locale.")

    model = model or MODEL
    transcript = data["transcript"]
    title = data["title"]
    chunks = chunk_transcript(transcript, max_tokens=CHUNK_SIZE, overlap_tokens=CHUNK_OVERLAP)

    prompt_tpl = _load_prompt("analyzer.xml")
    analyses = []
    for i, ch in enumerate(chunks):
        prompt = prompt_tpl.replace("{video_title}", title).replace("{output_language}", output_language).replace("{transcript}", ch["text"])
        logger.info("Analyse chunk %d/%d (%d tokens)", i + 1, len(chunks), ch["tokens"])
        try:
            result = llm_complete(prompt, model=model, api_key=api_key, provider=provider)
            analyses.append(result)
        except Exception as e:
            logger.error("Chunk %d failed: %s", i + 1, e)
            analyses.append(f"[Chunk {i+1} error: {e}]")

    if len(analyses) > 1:
        fusion_tpl = _load_prompt("fusion.xml")
        fusion_prompt = fusion_tpl.replace("{video_title}", title).replace("{output_language}", output_language).replace("{analyses}", "\n\n---\n\n".join(analyses))
        logger.info("Fusion des %d analyses", len(analyses))
        final = llm_complete(fusion_prompt, model=model, api_key=api_key, provider=provider)
    else:
        final = analyses[0] if analyses else "Aucune analyse produite."

    chapters = _extract_chapters(final)
    insights = _extract_insights(final)

    avertissements = []
    if not chapters:
        avertissements.append("Aucun chapitre exploitable n'a été trouvé dans la réponse du modèle.")

    return {
        "titre": title,
        "resume": final,
        "chapitres": chapters,
        "insights": insights,
        "langue_source": data.get("language", "unknown"),
        "avertissements": avertissements,
    }


def _highlight_reel(url: str, resume: str, duree_clip: float, sous_titres: bool, narration: bool, langue_narration: str, export_vertical: str) -> dict:
    from lib.video_mounter import create_highlight_reel as _reel
    result = _reel(
        video_url=url,
        summary_text=resume,
        clip_duration=duree_clip,
        burn_subtitles=sous_titres,
        tts_narration=narration,
        tts_lang=langue_narration,
        export_vertical_mode=export_vertical,
    )
    return {"reel_path": result.get("reel_path"), "clip_count": result.get("clip_count"), "vertical_path": result.get("vertical_path")}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Synopsis — Résumé YouTube</title>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0a0b;
  --surface: #121214;
  --surface2: #18181c;
  --border: #262629;
  --text: #e4e4e7;
  --text2: #a1a1aa;
  --accent: #f5a623;
  --accent2: #e07b39;
  --red: #e5484d;
  --radius: 8px;
  --font-serif: 'Newsreader', Georgia, serif;
  --font-mono: 'DM Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-serif);
  min-height: 100vh;
  background-image:
    radial-gradient(ellipse at 50% 0%, rgba(245,166,35,0.06) 0%, transparent 60%),
    url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
}
.container { max-width: 800px; margin: 0 auto; padding: 48px 24px 120px; }

header { text-align: center; margin-bottom: 56px; }
header h1 {
  font-size: 3rem; font-weight: 500; letter-spacing: -0.03em; color: var(--text);
  display: flex; align-items: center; justify-content: center; gap: 14px;
}
header h1 span.accent { color: var(--accent); }
header p { color: var(--text2); font-size: 1.1rem; margin-top: 8px; font-style: italic; }
header .bar { width: 48px; height: 2px; background: var(--accent); margin: 20px auto 0; }

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
  margin-bottom: 20px;
}
.card h2 {
  font-size: 1.25rem; font-weight: 500; margin-bottom: 20px;
  display: flex; align-items: center; gap: 10px;
}
.card h2 .dot { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; }

.form-row { display: flex; gap: 12px; flex-wrap: wrap; }
.form-row > * { flex: 1; min-width: 140px; }

label { display: block; color: var(--text2); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; font-family: var(--font-mono); font-weight: 500; }
input, select, textarea {
  width: 100%; background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 12px 14px; border-radius: 6px;
  font-family: var(--font-mono); font-size: 0.85rem; outline: none;
  transition: border-color 0.2s;
}
input:focus, select:focus { border-color: var(--accent); }

.btn {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--accent); color: #0a0a0b; border: none;
  padding: 12px 24px; border-radius: 6px; font-family: var(--font-mono);
  font-size: 0.85rem; font-weight: 500; cursor: pointer;
  transition: background 0.2s, transform 0.15s;
  letter-spacing: 0.02em;
}
.btn:hover { background: var(--accent2); }
.btn:active { transform: scale(0.98); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
.btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }

.btn-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }

.spinner { display: none; width: 18px; height: 18px; border: 2px solid rgba(10,10,11,0.2); border-top-color: #0a0a0b; border-radius: 50%; animation: spin 0.6s linear infinite; margin-left: 4px; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading .spinner { display: inline-block; }

.error { background: rgba(229,72,77,0.1); border: 1px solid var(--red); color: var(--red); padding: 14px 18px; border-radius: 6px; margin-top: 16px; font-family: var(--font-mono); font-size: 0.8rem; display: none; }
.error.show { display: block; }
.error.info { background: rgba(48,164,108,0.1); border-color: var(--green, #30a46c); color: var(--green, #30a46c); }
#image-out img { max-width: 100%; border-radius: 6px; margin-top: 8px; }

.result { display: none; margin-top: 20px; }
.result.show { display: block; }

.result h3.titre-video {
  font-size: 1.5rem; font-weight: 600; margin-bottom: 20px; line-height: 1.3;
}
.result .meta { color: var(--text2); font-size: 0.85rem; font-family: var(--font-mono); margin-bottom: 24px; }

.section { margin-bottom: 24px; }
.section h4 {
  font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--accent); margin-bottom: 10px; font-family: var(--font-mono); font-weight: 500;
}
.section .resume-text {
  line-height: 1.7; color: var(--text2); font-size: 1rem; white-space: pre-wrap;
}

.chapter-list { list-style: none; }
.chapter-list li {
  display: flex; align-items: flex-start; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.15s;
  border-radius: 4px; padding: 10px 8px; margin: 0 -8px;
}
.chapter-list li:hover { background: var(--surface2); }
.chapter-list li:last-child { border-bottom: none; }
.chapter-list .time {
  font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent);
  min-width: 60px; padding-top: 2px;
}
.chapter-list .chap-title { font-size: 0.95rem; line-height: 1.4; }

.insight-list { list-style: none; }
.insight-list li {
  padding: 8px 0; padding-left: 18px; position: relative; color: var(--text2);
  font-size: 0.95rem; line-height: 1.5;
}
.insight-list li::before {
  content: ''; position: absolute; left: 0; top: 14px;
  width: 6px; height: 6px; background: var(--accent); border-radius: 50%;
}

.reel-config { display: none; }
.reel-config.show { display: block; }

.check-row { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
.check-row label { margin-bottom: 0; cursor: pointer; }
.check-row input[type=checkbox] { width: auto; accent-color: var(--accent); }

.reel-result { display: none; margin-top: 16px; }
.reel-result.show { display: block; }
.reel-result .path { font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent); word-break: break-all; }

footer { text-align: center; margin-top: 80px; color: var(--text2); font-size: 0.8rem; opacity: 0.5; }

.indispo { opacity: .45; cursor: not-allowed; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1><span class="accent">Synopsis</span></h1>
    <p data-t="tagline">Résumé vidéo YouTube par IA</p>
    <div class="lang-switch"><button type="button" class="btn btn-secondary" id="lang-fr" onclick="setLangue('fr')" style="padding:4px 10px">FR</button> <button type="button" class="btn btn-secondary" id="lang-en" onclick="setLangue('en')" style="padding:4px 10px">EN</button></div>
    <div class="bar"></div>
  </header>

  <div class="card" id="config-card">
    <h2><span class="dot"></span><span data-t="config">Configuration API</span></h2>
    <div class="form-row">
      <div style="flex:1">
        <label data-t="fournisseur">Fournisseur</label>
        <select id="provider-select">
          <option value="openrouter">OpenRouter</option>
          <option value="openai">OpenAI</option>
          <option value="opencode">OpenCode Go</option>
          <option value="local" data-cap="llm_local">Local (Ollama)</option>
          <option value="custom" data-cap="llm_custom" data-t="prov_custom">Personnalisé</option>
        </select>
      </div>
      <div style="flex:2">
        <label data-t="cle_api">Clé API</label>
        <div style="display:flex;gap:8px">
          <input type="password" id="api-key" placeholder="sk-..." autocomplete="off" style="flex:1">
          <button class="btn btn-secondary" onclick="toggleKeyVisibility()" style="padding:12px 14px;flex:0" title="Afficher/masquer">👁</button>
        </div>
      </div>
    </div>
    <div class="form-row" style="margin-top:12px">
      <div style="flex:1">
        <label data-t="modele">Modèle</label>
        <div style="display:flex;gap:8px">
          <select id="modele-select" style="flex:1"></select>
          <button class="btn btn-secondary" id="btn-refresh-models" onclick="rafraichirModeles()" style="padding:12px 14px;flex:0" title="Rafraîchir depuis l'API">↻</button>
        </div>
      </div>
    </div>
    <p id="modeles-source" style="color:var(--text2);font-size:0.75rem;margin-top:8px;font-family:var(--font-mono)"></p>
  </div>

  <div class="card">
    <h2><span class="dot"></span><span data-t="nouveau_resume">Nouveau résumé</span></h2>
    <div class="form-row">
      <div style="flex:2">
        <label data-t="url_youtube">URL YouTube</label>
        <input type="text" id="url" placeholder="https://www.youtube.com/watch?v=..." autofocus>
        <div id="limites" data-t="limites_light" style="display:none;margin-top:8px;font-size:0.85rem;color:var(--text2)"></div>
      </div>
      <div style="flex:1">
        <label data-t="langue">Langue</label>
        <select id="langue">
          <option>Français</option><option>English</option><option>Español</option>
          <option>Deutsch</option><option>Português</option><option>Italiano</option>
        </select>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn" id="btn-resumer" onclick="resumer()">
        <span data-t="resumer">Résumer</span> <span class="spinner"></span>
      </button>
      <button class="btn btn-secondary" id="btn-reel-toggle" onclick="toggleReel()" data-t="reel_toggle">+ Highlight Reel</button>
    </div>
    <div class="error" id="error"></div>
  </div>

  <div class="card reel-config" id="reel-config">
    <h2><span class="dot"></span><span data-t="reel_titre">Highlight Reel</span></h2>
    <div class="form-row">
      <div>
        <label data-t="duree_clip">Durée par clip (sec)</label>
        <input type="number" id="duree-clip" value="45" min="10" max="120">
      </div>
      <div>
        <label data-t="langue_narration">Langue narration</label>
        <select id="langue-narration">
          <option value="fr">Français</option><option value="en">English</option>
          <option value="es">Español</option><option value="de">Deutsch</option>
        </select>
      </div>
    </div>
    <div class="check-row">
      <input type="checkbox" id="sous-titres" checked>
      <label for="sous-titres" data-t="sous_titres">Sous-titres des chapitres</label>
    </div>
    <div class="check-row">
      <input type="checkbox" id="narration" data-cap="tts">
      <label for="narration" data-t="narration_tts">Narration TTS</label>
    </div>
    <div class="btn-row">
      <button class="btn" id="btn-reel" data-cap="reel" onclick="genererReel()" disabled>
        <span data-t="generer_reel">Générer le reel</span> <span class="spinner"></span>
      </button>
    </div>
    <div class="reel-result" id="reel-result"></div>
  </div>

  <details class="card" id="multi">
    <summary data-t="plusieurs">Résumer plusieurs vidéos</summary>
    <textarea id="urls-multi" rows="5" placeholder="Une URL YouTube par ligne" data-t-ph="ph_urls"></textarea>
    <div class="btn-row">
      <button class="btn" id="btn-multi" onclick="resumerLot()"><span data-t="resumer_lot">Résumer le lot</span> <span class="spinner"></span></button>
    </div>
    <div id="multi-result"></div>
  </details>

  <details class="card" id="fichier">
    <summary data-t="fichier_titre">Résumer un fichier audio/vidéo</summary>
    <input type="file" id="fichier-input" accept="audio/*,video/*" data-cap="whisper">
    <div class="btn-row">
      <button class="btn" id="btn-fichier" data-cap="whisper" onclick="resumerFichier()"><span data-t="resumer_fichier">Transcrire et résumer</span> <span class="spinner"></span></button>
    </div>
  </details>

  <div class="result" id="result"></div>

  <footer>Synopsis v1.0</footer>
</div>

<script>
const $ = (s) => document.querySelector(s);
let dernierResume = null;

var FR_SECOURS = {
  resume: 'Résumé', chapitres: 'Chapitres', insights: 'Insights clés',
  export_html: 'Enregistrer en HTML', export_md: 'Enregistrer en Markdown', export_pdf: 'Enregistrer en PDF',
  question: 'Poser une question', demander: 'Demander', ph_question: 'Une question sur cette vidéo...',
  langue_source: 'Langue source',
  avertissements: 'Avertissements', prov_custom: 'Personnalisé',
  limites_light: 'Version en ligne : fonctionne uniquement avec les vidéos qui ont des sous-titres YouTube (manuels ou automatiques) et jusqu’à environ 30 minutes. Pour une vidéo sans sous-titres, utilisez la version locale (transcription Whisper).',
  err_url: 'Colle une URL YouTube.', err_cle: 'Saisis ta clé API dans Configuration.', err_urls: 'Colle au moins une URL.',
  fichier_titre: 'Résumer un fichier audio/vidéo', resumer_fichier: 'Transcrire et résumer', err_fichier: 'Choisis un fichier audio ou vidéo.',
  export_obsidian: 'Envoyer vers Obsidian', export_excalidraw: 'Enregistrer en Excalidraw', ph_vault: 'Dossier du vault Obsidian', ok_obsidian: 'Note enregistrée : ', err_vault: 'Indique le dossier du vault.',
  export_drive: 'Envoyer vers Google Drive', ok_drive: 'Envoyé vers Drive : ', generer_image: 'Générer une image', err_image: 'Génération d’image impossible.'
};
var DICT = {};
var LANGUE = 'fr';
function t(cle) { return DICT[cle] || FR_SECOURS[cle] || cle; }
function appliquerTextes(racine) {
  var base = racine || document;
  var els = base.querySelectorAll('[data-t]');
  for (var i = 0; i < els.length; i++) {
    var v = DICT[els[i].getAttribute('data-t')];
    if (v) els[i].textContent = v;
  }
  var ph = base.querySelectorAll('[data-t-ph]');
  for (var j = 0; j < ph.length; j++) {
    var p = DICT[ph[j].getAttribute('data-t-ph')];
    if (p) ph[j].placeholder = p;
  }
}
async function setLangue(l) {
  try {
    var r = await fetch('/i18n/' + l);
    if (!r.ok) throw new Error('i18n');
    DICT = await r.json();
    LANGUE = l;
    document.documentElement.lang = l;
    appliquerTextes();
    $('#lang-fr').style.opacity = l === 'fr' ? '1' : '0.5';
    $('#lang-en').style.opacity = l === 'en' ? '1' : '0.5';
    try { localStorage.setItem('synopsis_lang', l); } catch (e) {}
  } catch (e) {}
}
(function initLangue() {
  var l = 'fr';
  try { l = localStorage.getItem('synopsis_lang') || 'fr'; } catch (e) {}
  if (l !== 'fr' && l !== 'en') l = 'fr';
  setLangue(l);
})();

const PROVIDERS = {
  openrouter: { name: 'OpenRouter', keyHint: 'sk-or-v1-...', keysUrl: 'https://openrouter.ai/keys' },
  openai: { name: 'OpenAI', keyHint: 'sk-proj-...', keysUrl: 'https://platform.openai.com/api-keys' },
  opencode: { name: 'OpenCode Go', keyHint: 'sk-...', keysUrl: 'https://opencode.ai' },
  local: { name: 'Local (Ollama)', keyHint: 'Aucune clé nécessaire', keysUrl: '' },
  custom: { name: 'Personnalisé', keyHint: 'Clé (si requise)', keysUrl: '' },
};

function sansCle(p) { return p === 'local' || p === 'custom'; }

const FALLBACK_MODELS = {
  openrouter: [
    "deepseek/deepseek-v4-flash","deepseek/deepseek-v4-pro",
    "google/gemini-2.5-pro","google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.5","anthropic/claude-opus-4.5",
    "openai/gpt-5.2","openai/gpt-5.2-mini",
    "qwen/qwen3-coder","mistral/pi-zero-2",
    "x-ai/grok-4","amazon/nova-pro-v2",
  ],
  openai: ["gpt-5.2","gpt-5.2-mini","gpt-4.1","gpt-4.1-mini","o4-mini","o3"],
  opencode: ["deepseek/deepseek-v4-flash","deepseek/deepseek-v4-pro"],
  local: [],
  custom: [],
};

function getProvider() { return $('#provider-select').value; }
function getApiKey() { return $('#api-key').value.trim(); }
function getModele() { return $('#modele-select').value; }

function saveConfig() {
  const p = getProvider();
  const k = getApiKey();
  const m = getModele();
  localStorage.setItem('synopsis_provider', p);
  localStorage.setItem('synopsis_api_key_' + p, k);
  localStorage.setItem('synopsis_modele_' + p, m);
  localStorage.setItem('synopsis_models_' + p, JSON.stringify(getModelOptions()));
}

function loadConfig() {
  const p = localStorage.getItem('synopsis_provider') || 'openrouter';
  $('#provider-select').value = p;
  const k = localStorage.getItem('synopsis_api_key_' + p) || '';
  $('#api-key').value = k;
  const cached = localStorage.getItem('synopsis_models_' + p);
  if (cached) {
    try {
      populateModels(JSON.parse(cached), 'cache');
    } catch(e) {
      populateModels(FALLBACK_MODELS[p], 'fallback');
    }
  } else {
    populateModels(FALLBACK_MODELS[p], 'fallback');
  }
  const m = localStorage.getItem('synopsis_modele_' + p);
  if (m) $('#modele-select').value = m;
  updateKeyPlaceholder();
}

function getModelOptions() {
  return Array.from($('#modele-select').options).map(function(o) { return o.value; });
}

function populateModels(models, source) {
  const sel = $('#modele-select');
  const current = sel.value;
  sel.innerHTML = '';
  models.forEach(function(m) {
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = m; sel.appendChild(opt);
  });
  if (current && models.indexOf(current) !== -1) sel.value = current;
  else sel.value = models[0] || '';
  var txt = models.length + ' modeles';
  if (source === 'api') txt += ' (via API)';
  else if (source === 'cache') txt += ' (en cache)';
  else txt += ' (liste par defaut)';
  $('#modeles-source').textContent = txt;
}

function onProviderChange() {
  const p = getProvider();
  const k = localStorage.getItem('synopsis_api_key_' + p) || '';
  $('#api-key').value = k;
  updateKeyPlaceholder();
  const cached = localStorage.getItem('synopsis_models_' + p);
  if (cached) {
    try { populateModels(JSON.parse(cached), 'cache'); return; } catch(e) {}
  }
  populateModels(FALLBACK_MODELS[p], 'fallback');
  const m = localStorage.getItem('synopsis_modele_' + p);
  if (m) $('#modele-select').value = m;
}

function updateKeyPlaceholder() {
  const p = getProvider();
  $('#api-key').placeholder = PROVIDERS[p].keyHint;
}

async function rafraichirModeles() {
  const p = getProvider();
  const k = getApiKey();
  const btn = $('#btn-refresh-models');
  btn.textContent = '...'; btn.disabled = true;
  try {
    const r = await fetch('/modeles', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({provider: p, api_key: k})
    });
    if (!r.ok) throw new Error('modeles');
    const data = await r.json();
    populateModels(data.models, data.source);
    saveConfig();
  } catch(e) {
    $('#modeles-source').textContent = 'Erreur reseau';
  }
  btn.textContent = '\u21BB'; btn.disabled = false;
}

function toggleKeyVisibility() {
  const inp = $('#api-key');
  inp.type = inp.type === 'password' ? 'text' : 'password';
}

$('#provider-select').addEventListener('change', onProviderChange);
$('#api-key').addEventListener('change', saveConfig);
$('#modele-select').addEventListener('change', saveConfig);

var CAPS = {};
function capOk(nom) { return !!(CAPS[nom] && CAPS[nom].ok); }
async function chargerCapacites() {
  try {
    var r = await fetch('/sante');
    var d = await r.json();
    CAPS = d.capacites || {};
    var lim = $('#limites');
    if (lim && d.mode === 'light') lim.style.display = '';
  } catch (e) { CAPS = {}; }
  appliquerCapacites();
}
function appliquerCapacites() {
  var els = document.querySelectorAll('[data-cap]');
  for (var i = 0; i < els.length; i++) {
    var nom = els[i].getAttribute('data-cap');
    if (!capOk(nom)) {
      els[i].disabled = true;
      els[i].title = (CAPS[nom] && CAPS[nom].raison) || 'Indisponible';
      els[i].classList.add('indispo');
    }
  }
  var sel = $('#provider-select');
  if (sel && sel.selectedIndex >= 0 && sel.options[sel.selectedIndex].disabled) {
    sel.value = 'openrouter';
    onProviderChange();
    try { saveConfig(); } catch (e) {}
  }
}
chargerCapacites();

(function init() {
  loadConfig();
})();

var errTimer = null;
function showError(msg) {
  const el = $('#error');
  el.textContent = msg;
  el.classList.remove('info');
  el.classList.add('show');
  clearTimeout(errTimer);
  errTimer = setTimeout(function() { el.classList.remove('show'); }, 10000);
}
function hideError() {
  clearTimeout(errTimer);
  $('#error').classList.remove('show', 'info');
}

function setLoading(btn, loading) {
  if (loading) { btn.classList.add('loading'); btn.disabled = true; }
  else { btn.classList.remove('loading'); btn.disabled = false; }
}

async function resumer() {
  const url = $('#url').value.trim();
  const apiKey = getApiKey();
  const provider = getProvider();
  if (!url) return showError(t('err_url'));
  if (!apiKey && !sansCle(provider)) return showError(t('err_cle'));
  hideError();
  const btn = $('#btn-resumer');
  setLoading(btn, true);
  $('#result').classList.remove('show');
  try {
    const r = await fetch('/resumer', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url, langue: $('#langue').value, api_key: apiKey, modele: getModele(), provider: provider})
    });
    if (!r.ok) { const d = await r.json(); throw new Error(d.detail || r.statusText); }
    const data = await r.json();
    dernierResume = data;
    afficherResume(data);
    if ($('#reel-config').classList.contains('show')) {
      $('#btn-reel').disabled = !capOk('reel');
    }
  } catch(e) { showError(e.message); }
  finally { setLoading(btn, false); }
}

function afficherResume(data, cible) {
  var html = '';
  html += '<h3 class="titre-video">' + escapeHtml(data.titre) + '</h3>';
  html += '<div class="meta"><span data-t="langue_source">' + t('langue_source') + '</span> : ' + escapeHtml(data.langue_source) + '</div>';
  html += '<div class="section"><h4 data-t="resume">' + t('resume') + '</h4><div class="resume-text">' + escapeHtml(data.resume) + '</div></div>';
  if (data.avertissements && data.avertissements.length) {
    html += '<div class="section"><h4 data-t="avertissements">' + t('avertissements') + '</h4><ul class="insight-list">';
    for (var w = 0; w < data.avertissements.length; w++) html += '<li>' + escapeHtml(data.avertissements[w]) + '</li>';
    html += '</ul></div>';
  }
  if (data.chapitres && data.chapitres.length) {
    html += '<div class="section"><h4 data-t="chapitres">' + t('chapitres') + '</h4><ul class="chapter-list">';
    for (var i = 0; i < data.chapitres.length; i++) {
      var c = data.chapitres[i];
      html += '<li onclick="naviguerChapitre(this)" data-time="' + (c.ts_seconds || 0) + '"><span class="time">' + escapeHtml(c.timestamp || formatTime(c.ts_seconds)) + '</span><span class="chap-title">' + escapeHtml(c.subject) + (c.description ? ' <span class="meta">- ' + escapeHtml(c.description) + '</span>' : '') + '</span></li>';
    }
    html += '</ul></div>';
  }
  if (data.insights && data.insights.length) {
    html += '<div class="section"><h4 data-t="insights">' + t('insights') + '</h4><ul class="insight-list">';
    for (var j = 0; j < data.insights.length; j++) {
      var ins = data.insights[j];
      html += '<li>' + (ins.timestamp ? '[' + escapeHtml(ins.timestamp) + '] ' : '') + '<b>' + escapeHtml(ins.title) + '</b> : ' + escapeHtml(ins.description) + '</li>';
    }
    html += '</ul></div>';
  }
  if (!cible) html += '<div class="section"><button type="button" onclick="exporterHtml()" data-t="export_html">' + t('export_html') + '</button> <button type="button" onclick="exporterPdf()" data-t="export_pdf">' + t('export_pdf') + '</button> <button type="button" onclick="exporterMarkdown()" data-t="export_md">' + t('export_md') + '</button> <button type="button" onclick="exporterExcalidraw()" data-cap="excalidraw" data-t="export_excalidraw">' + t('export_excalidraw') + '</button><div style="margin-top:8px"><input type="text" id="vault-dir" data-t-ph="ph_vault" placeholder="' + escapeHtml(t('ph_vault')) + '" value="' + escapeHtml(lireVault()) + '"> <button type="button" onclick="exporterObsidian()" data-cap="obsidian" data-t="export_obsidian">' + t('export_obsidian') + '</button></div><div style="margin-top:8px"><button type="button" onclick="exporterDrive()" data-cap="drive" data-t="export_drive">' + t('export_drive') + '</button> <button type="button" id="btn-image" onclick="genererImage()" data-cap="images" data-t="generer_image">' + t('generer_image') + '</button><div id="image-out"></div></div></div>';
  if (!cible) html += '<div class="section" id="qa-box"><h4 data-t="question">' + t('question') + '</h4><input type="text" id="qa-question" data-t-ph="ph_question" placeholder="' + escapeHtml(t('ph_question')) + '"> <button type="button" id="btn-qa" onclick="poserQuestion()" data-t="demander">' + t('demander') + '</button><div class="resume-text" id="qa-reponse"></div></div>';
  var el = cible || $('#result');
  el.innerHTML = html;
  appliquerCapacites();
  el.classList.add('show');
  if (!cible) el.scrollIntoView({behavior: 'smooth', block: 'start'});
}

async function poserQuestion() {
  var q = $('#qa-question').value.trim();
  if (!q || !dernierResume) return;
  var apiKey = getApiKey();
  var provider = getProvider();
  if (!apiKey && !sansCle(provider)) return showError(t('err_cle'));
  hideError();
  var btn = $('#btn-qa');
  setLoading(btn, true);
  try {
    var r = await fetch('/qa', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({titre: dernierResume.titre, resume: dernierResume.resume, question: q, api_key: apiKey, modele: getModele(), provider: provider})
    });
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    $('#qa-reponse').innerHTML = escapeHtml(d.reponse);
  } catch(e) { showError(e.message); }
  finally { setLoading(btn, false); }
}

async function resumerLot() {
  var apiKey = getApiKey();
  var provider = getProvider();
  var urls = $('#urls-multi').value.split(String.fromCharCode(10)).map(function(u) { return u.trim(); }).filter(function(u) { return u; });
  if (!urls.length) return showError(t('err_urls'));
  if (!apiKey && !sansCle(provider)) return showError(t('err_cle'));
  hideError();
  var btn = $('#btn-multi');
  setLoading(btn, true);
  var zone = $('#multi-result');
  zone.innerHTML = '';
  try {
    for (var i = 0; i < urls.length; i++) {
      var box = document.createElement('div');
      zone.appendChild(box);
      box.className = 'meta';
      box.textContent = urls[i] + ' ...';
      try {
        var r = await fetch('/resumer', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({url: urls[i], langue: $('#langue').value, api_key: apiKey, modele: getModele(), provider: provider})
        });
        var d = await r.json();
        if (!r.ok) throw new Error(d.detail || r.statusText);
        box.className = 'result';
        box.setAttribute('data-url', urls[i]);
        afficherResume(d, box);
      } catch (e) {
        box.className = 'error show';
        box.textContent = urls[i] + ' : ' + e.message;
      }
    }
  } finally { setLoading(btn, false); }
}

async function resumerFichier() {
  var apiKey = getApiKey();
  var provider = getProvider();
  var f = $('#fichier-input').files[0];
  if (!f) return showError(t('err_fichier'));
  if (!apiKey && !sansCle(provider)) return showError(t('err_cle'));
  hideError();
  var btn = $('#btn-fichier');
  setLoading(btn, true);
  $('#result').classList.remove('show');
  try {
    var fd = new FormData();
    fd.append('fichier', f);
    fd.append('langue', $('#langue').value);
    fd.append('modele', getModele());
    fd.append('api_key', apiKey);
    fd.append('provider', provider);
    var r = await fetch('/resumer-fichier', {method: 'POST', body: fd});
    var d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    dernierResume = d;
    afficherResume(d);
  } catch(e) { showError(e.message); }
  finally { setLoading(btn, false); }
}

function toggleReel() {
  var el = $('#reel-config');
  el.classList.toggle('show');
  if (dernierResume) $('#btn-reel').disabled = !capOk('reel');
}

async function genererReel() {
  var url = $('#url').value.trim();
  var apiKey = getApiKey();
  var provider = getProvider();
  if (!url || !dernierResume) return showError("Fais d'abord un resume.");
  if (!apiKey && !sansCle(provider)) return showError(t('err_cle'));
  hideError();
  var btn = $('#btn-reel');
  setLoading(btn, true);
  try {
    var r = await fetch('/reel', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        url: url,
        duree_clip: parseFloat($('#duree-clip').value) || 45,
        sous_titres: $('#sous-titres').checked,
        narration: $('#narration').checked,
        langue_narration: $('#langue-narration').value,
        export_vertical: '',
        api_key: apiKey,
        provider: provider,
      })
    });
    if (!r.ok) { var d = await r.json(); throw new Error(d.detail || r.statusText); }
    var data = await r.json();
    var html = '<div class="error" style="background:rgba(245,166,35,0.1);border-color:var(--accent);color:var(--accent);display:block;margin-top:0">';
    html += 'Reel genere ! ' + (data.clip_count || '?') + ' clips.<br>';
    if (data.reel_path) html += 'Fichier : <span class="path">' + escapeHtml(data.reel_path) + '</span>';
    if (data.vertical_path) html += '<br>Vertical : <span class="path">' + escapeHtml(data.vertical_path) + '</span>';
    html += '</div>';
    $('#reel-result').innerHTML = html;
    $('#reel-result').classList.add('show');
  } catch(e) { showError(e.message); }
  finally { setLoading(btn, false); }
}

function naviguerChapitre(el) {
  var time = el.dataset.time;
  if (!time) return;
  var holder = el.closest('[data-url]');
  var url = holder ? holder.getAttribute('data-url') : $('#url').value.trim();
  var videoId = extractVideoId(url);
  if (!videoId) return;
  var seconds = parseFloat(time) || 0;
  window.open('https://www.youtube.com/watch?v=' + videoId + '&t=' + Math.floor(seconds) + 's', '_blank');
}

function extractVideoId(url) {
  var m = url.match(/(?:v=|\\/)([\\w-]{11})(?:[&?]|$)/);
  return m ? m[1] : null;
}

function formatTime(seconds) {
  var s = parseFloat(seconds) || 0;
  var min = Math.floor(s / 60);
  var sec = Math.floor(s % 60);
  return min + ':' + String(sec).padStart(2, '0');
}

function documentExport() {
  var body = $('#result').cloneNode(true);
  var retirer = body.querySelectorAll('button, input, textarea, #qa-box, #image-out');
  for (var i = 0; i < retirer.length; i++) { if (retirer[i].parentNode) retirer[i].parentNode.removeChild(retirer[i]); }
  var css = '';
  var styles = document.querySelectorAll('style');
  for (var k = 0; k < styles.length; k++) css += styles[k].textContent;
  var titre = escapeHtml(dernierResume ? dernierResume.titre : 'Synopsis');
  // Document autonome : aucun script ne doit s'y exécuter (contenu issu du LLM).
  var csp = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">`;
  return '<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">' + csp + '<title>' + titre + '</title><style>' + css + ' .result{display:block !important;} body{display:block;}</style></head><body><div class="result show">' + body.innerHTML + '</div></body></html>';
}

function exporterHtml() {
  if (!dernierResume) return showError("Fais d'abord un resume.");
  var blob = new Blob([documentExport()], {type: 'text/html;charset=utf-8'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (dernierResume.titre || 'synopsis').replace(/[^a-zA-Z0-9]+/g, '_') + '.html';
  a.click();
  URL.revokeObjectURL(a.href);
}

function exporterMarkdown() {
  if (!dernierResume) return showError("Fais d'abord un resume.");
  fetch('/export/markdown', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(dernierResume)})
    .then(function(r) { if (!r.ok) throw new Error('Export impossible'); return r.blob(); })
    .then(function(b) {
      var a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = (dernierResume.titre || 'synopsis').replace(/[^a-zA-Z0-9]+/g, '_') + '.md';
      a.click();
      URL.revokeObjectURL(a.href);
    })
    .catch(function(e) { showError(e.message); });
}

function lireVault() {
  try { return localStorage.getItem('synopsis_vault') || ''; } catch (e) { return ''; }
}

function exporterExcalidraw() {
  if (!dernierResume) return showError("Fais d'abord un resume.");
  fetch('/export/excalidraw', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(dernierResume)})
    .then(function(r) { if (!r.ok) throw new Error('Export impossible'); return r.blob(); })
    .then(function(b) {
      var a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = (dernierResume.titre || 'synopsis').replace(/[^a-zA-Z0-9]+/g, '_') + '.excalidraw';
      a.click();
      URL.revokeObjectURL(a.href);
    })
    .catch(function(e) { showError(e.message); });
}

function exporterObsidian() {
  if (!dernierResume) return showError("Fais d'abord un resume.");
  var champ = $('#vault-dir');
  var vault = champ ? champ.value.trim() : '';
  if (!vault) return showError(t('err_vault'));
  try { localStorage.setItem('synopsis_vault', vault); } catch (e) {}
  fetch('/export/obsidian', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({resultat: dernierResume, vault_dir: vault, url: $('#url').value.trim()})})
    .then(function(r) { return r.json().then(function(j) { if (!r.ok) throw new Error(j.detail || 'Export impossible'); return j; }); })
    .then(function(j) { showInfo(t('ok_obsidian') + j.path); })
    .catch(function(e) { showError(e.message); });
}

function showInfo(msg) {
  var el = $('#error');
  el.textContent = msg;
  el.classList.add('show', 'info');
  clearTimeout(errTimer);
  errTimer = setTimeout(function() { el.classList.remove('show', 'info'); }, 10000);
}

function exporterDrive() {
  if (!dernierResume) return showError("Fais d'abord un resume.");
  hideError();
  fetch('/export/drive', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({resultat: dernierResume})})
    .then(function(r) { return r.json().then(function(j) { if (!r.ok) throw new Error(j.detail || 'Export impossible'); return j; }); })
    .then(function(j) { showInfo(t('ok_drive') + j.url); })
    .catch(function(e) { showError(e.message); });
}

function genererImage() {
  if (!dernierResume) return showError("Fais d'abord un resume.");
  hideError();
  var btn = $('#btn-image');
  setLoading(btn, true);
  fetch('/image', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({resultat: dernierResume, api_key: getApiKey(), provider: getProvider()})})
    .then(function(r) { return r.json().then(function(j) { if (!r.ok) throw new Error(j.detail || t('err_image')); return j; }); })
    .then(function(j) {
      var img = document.createElement('img');
      img.src = 'data:' + j.mime + ';base64,' + j.image;
      img.alt = dernierResume.titre || '';
      var out = $('#image-out');
      out.textContent = '';
      out.appendChild(img);
    })
    .catch(function(e) { showError(e.message); })
    .finally(function() { setLoading(btn, false); });
}

function exporterPdf() {
  if (!dernierResume) return showError("Fais d'abord un resume.");
  var w = window.open('', '_blank');
  if (!w) return showError("Autorise les popups pour exporter en PDF.");
  w.document.write(documentExport());
  w.document.close();
  w.onload = function() { w.focus(); w.print(); };
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

document.getElementById('url').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') resumer();
});
</script>
</body>
</html>"""


@functools.lru_cache(maxsize=1)
def _csp_page() -> str:
    return security.csp_page(index())


@app.get("/i18n/{lang}")
def i18n(lang: str):
    if lang not in TEXTS:
        raise HTTPException(status_code=404, detail="Langue inconnue")
    return TEXTS[lang]


@app.get("/sante")
def sante():
    checks = {"api": "ok", "version": "1.0.0", "mode": "light" if is_light_mode() else "full", "capacites": get_capabilities()}
    try:
        gateway = os.getenv("GATEWAY_URL", "")
        if gateway:
            import httpx
            r = httpx.get(f"{gateway}/health/liveliness", timeout=5)
            checks["gateway"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        else:
            checks["gateway"] = "non configuré"
    except Exception as e:
        checks["gateway"] = "erreur: " + security.scrub(e, limite=200)
    return checks


PROVIDER_MODELS_API = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "opencode": None,
}

PROVIDER_FALLBACK_MODELS = {
    "openrouter": [
        "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro",
        "google/gemini-2.5-pro", "google/gemini-2.5-flash",
        "anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4.5",
        "openai/gpt-5.2", "openai/gpt-5.2-mini",
        "qwen/qwen3-coder", "mistral/pi-zero-2",
        "x-ai/grok-4", "amazon/nova-pro-v2",
    ],
    "openai": ["gpt-5.2", "gpt-5.2-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini", "o3"],
    "opencode": ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"],
}


def _liste_modeles(provider: str, api_key: str = "") -> dict:
    caps = get_capabilities()
    if provider == "local":
        _exiger("llm_local", caps)
        return {"models": list_local_models(), "source": "local"}
    if provider == "custom":
        _exiger("llm_custom", caps)
        return {"models": list_custom_models(api_key), "source": "custom"}
    url = PROVIDER_MODELS_API.get(provider)
    if not url:
        return {"models": PROVIDER_FALLBACK_MODELS.get(provider, []), "source": "fallback"}
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = httpx.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()["data"]
            models = sorted([m["id"] for m in data if m.get("id")])
            return {"models": models, "source": "api"}
    except Exception:
        pass
    return {"models": PROVIDER_FALLBACK_MODELS.get(provider, []), "source": "fallback"}


@app.get("/modeles")
def modeles(provider: str = "openrouter"):
    # Pas de clé en query string (journaux, historique) : utiliser POST pour une liste authentifiée.
    return _liste_modeles(provider)


class ModelesRequest(BaseModel):
    provider: str = Field("openrouter", description="Provider")
    api_key: str = Field("", description="API key")


@app.post("/modeles")
def modeles_post(req: ModelesRequest):
    return _liste_modeles(req.provider, req.api_key)


MAX_ITEMS_EXPORT = 2000


def _valider_resultat(d) -> dict:
    """Valide un résultat Synopsis venant du client (400 plutôt que 500 sur entrée malformée)."""
    if not isinstance(d, dict):
        raise HTTPException(status_code=400, detail="Résultat manquant.")
    for champ in ("titre", "resume", "langue_source"):
        v = d.get(champ)
        if v is not None and not isinstance(v, str):
            raise HTTPException(status_code=400, detail=f"Champ '{champ}' invalide (texte attendu).")
    for champ in ("chapitres", "insights", "avertissements"):
        v = d.get(champ)
        if v is None:
            continue
        if not isinstance(v, list):
            raise HTTPException(status_code=400, detail=f"Champ '{champ}' invalide (liste attendue).")
        if len(v) > MAX_ITEMS_EXPORT:
            raise HTTPException(status_code=400, detail=f"Champ '{champ}' : trop d'éléments (max {MAX_ITEMS_EXPORT}).")
        if champ != "avertissements":
            for item in v:
                if not isinstance(item, dict) or any(isinstance(x, (dict, list)) for x in item.values()):
                    raise HTTPException(status_code=400, detail=f"Champ '{champ}' invalide (objets simples attendus).")
    return d


@app.post("/export/markdown")
def export_markdown(d: dict):
    return PlainTextResponse(to_markdown(_valider_resultat(d)), media_type="text/markdown")


@app.post("/export/excalidraw")
def export_excalidraw(d: dict):
    _exiger("excalidraw")
    return JSONResponse(to_excalidraw(_valider_resultat(d)))


@app.post("/export/obsidian")
def export_obsidian(body: dict):
    _exiger("obsidian")
    vault_dir = str(body.get("vault_dir") or "").strip()
    if not vault_dir:
        raise HTTPException(status_code=400, detail="Dossier du vault manquant.")
    resultat = _valider_resultat(body.get("resultat"))
    try:
        path = save_note(resultat, vault_dir, str(body.get("url") or ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=400, detail="Écriture impossible : " + str(e))
    return {"path": path}


class ImageRequest(BaseModel):
    resultat: dict = Field(..., description="Synopsis result")
    api_key: str = Field("", description="API key")
    provider: str = Field("", description="Provider of the key; only openrouter (or empty) is used for images")
    modele: str = Field("", description="Image model override")


@app.post("/export/drive")
def export_drive(body: dict):
    _exiger("drive")
    # Identifiants Drive : uniquement l'environnement serveur, jamais la requête.
    resultat = _valider_resultat(body.get("resultat"))
    try:
        creds = drive_exporter.load_credentials(os.getenv("GOOGLE_DRIVE_CREDENTIALS", ""))
        url = drive_exporter.upload_markdown(resultat, creds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail="Export Drive impossible : " + str(e))
    return {"url": url}


@app.post("/image")
def generate_image_route(req: ImageRequest):
    _exiger("images")
    _valider_resultat(req.resultat)
    key = req.api_key if req.provider in ("", "openrouter") else ""
    key = key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400, detail="Clé API OpenRouter requise pour générer une image.")
    try:
        data = image_generator.generate_image(image_generator.build_image_prompt(req.resultat), key, req.modele)
    except Exception as e:
        raise HTTPException(status_code=502, detail="Génération d'image impossible : " + str(e))
    import base64
    return {"image": base64.b64encode(data).decode("ascii"), "mime": "image/png"}


@app.post("/resumer")
def resumer(req: ResumerRequest):
    _verifier_llm(req.provider, req.api_key)
    try:
        result = _summarize(req.url, req.langue, req.modele, api_key=req.api_key, provider=req.provider)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Erreur /resumer")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resumer-fichier")
def resumer_fichier(fichier: UploadFile = File(...), langue: str = Form("Français"), modele: str = Form(""),
                    api_key: str = Form(""), provider: str = Form("")):
    caps = _exiger("whisper")
    _verifier_llm(provider, api_key, caps)
    import tempfile
    from lib.whisper_transcriber import transcribe_file
    suffix = Path(fichier.filename or "").suffix[:10]
    if not suffix.replace(".", "").isalnum():
        suffix = ""
    nom = Path(fichier.filename or "fichier").name or "fichier"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        limite = int(float(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024)
        total = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                bloc = fichier.file.read(1024 * 1024)
                if not bloc:
                    break
                total += len(bloc)
                if total > limite:
                    raise HTTPException(status_code=413, detail="Fichier trop volumineux (max " + os.getenv("MAX_UPLOAD_MB", "200") + " Mo).")
                out.write(bloc)
        segments = transcribe_file(tmp, "")
        if not segments:
            raise ValueError("Aucune parole détectée dans le fichier.")
        data = {"transcript": segments, "title": nom, "language": "unknown"}
        return _summarize_data(data, langue, modele, api_key=api_key, provider=provider)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Erreur /resumer-fichier")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


@app.post("/qa")
def qa(req: QaRequest):
    _verifier_llm(req.provider, req.api_key)
    try:
        rep = answer_question(
            req.titre, req.resume, req.question,
            lambda p: llm_complete(p, model=req.modele or MODEL, api_key=req.api_key, provider=req.provider),
        )
        return {"reponse": rep}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Erreur /qa")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resumer-lot")
def resumer_lot(req: LotRequest):
    _verifier_llm(req.provider, req.api_key)
    n = len([u for u in req.urls if u.strip()])
    if n > _max_batch():
        raise HTTPException(status_code=400, detail=f"Trop de vidéos dans le lot ({n}) : maximum {_max_batch()}.")
    return summarize_many(
        req.urls,
        lambda u: _summarize(u, req.langue, req.modele, api_key=req.api_key, provider=req.provider),
    )


@app.post("/reel")
def reel(req: ReelRequest):
    caps = _exiger("reel")
    if req.narration:
        _exiger("tts", caps)
    _verifier_llm(req.provider, req.api_key, caps)
    vid = extract_youtube_id(req.url)
    if not vid:
        raise HTTPException(status_code=400, detail="URL YouTube invalide. Format: youtube.com/watch?v=ID ou youtu.be/ID")
    # URL canonique reconstruite : jamais la chaîne du client vers yt-dlp (injection d'options / SSRF).
    url = f"https://www.youtube.com/watch?v={vid}"
    try:
        summary = _summarize(url, "Français", api_key=req.api_key, provider=req.provider)
        result = _highlight_reel(url, summary["resume"], req.duree_clip, req.sous_titres, req.narration, req.langue_narration, req.export_vertical)
        result["titre"] = summary["titre"]
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Erreur /reel")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "6090")))
