"""Génération d'images via OpenRouter (reprise de youtube-summarizer/src/image_generator.py).

httpx est utilisé à la place de requests. Les clés API ne sont jamais journalisées.
"""
import base64
import re
import time
from typing import Optional

IMAGE_PROVIDERS = {
    "flux": {"name": "Flux", "default_model": "black-forest-labs/flux.2-klein-4b",
             "models": ["black-forest-labs/flux.2-klein-4b", "black-forest-labs/flux.2-max", "black-forest-labs/flux.2-pro", "black-forest-labs/flux.2-flex"]},
    "recraft": {"name": "Recraft", "default_model": "recraft/recraft-v4.1",
                "models": ["recraft/recraft-v4.1", "recraft/recraft-v4.1-pro", "recraft/recraft-v4.1-vector", "recraft/recraft-v4.1-pro-vector"]},
    "gemini-image": {"name": "Gemini Image", "default_model": "google/gemini-3.1-flash-image-preview",
                     "models": ["google/gemini-3.1-flash-image-preview", "google/gemini-2.5-flash-image", "google/gemini-3-pro-image-preview"]},
    "openai-image": {"name": "OpenAI Image", "default_model": "openai/gpt-5.4-image-2",
                     "models": ["openai/gpt-5.4-image-2", "openai/gpt-5-image", "openai/gpt-5-image-mini"]},
    "seedream": {"name": "Seedream", "default_model": "bytedance-seed/seedream-4.5", "models": ["bytedance-seed/seedream-4.5"]},
    "riverflow": {"name": "Riverflow", "default_model": "sourceful/riverflow-v2-pro",
                  "models": ["sourceful/riverflow-v2-pro", "sourceful/riverflow-v2-fast", "sourceful/riverflow-v2-standard-preview"]},
    "mai-image": {"name": "MAI Image", "default_model": "microsoft/mai-image-2.5", "models": ["microsoft/mai-image-2.5"]},
    "grok-imagine": {"name": "Grok Imagine", "default_model": "x-ai/grok-imagine-image-quality",
                     "models": ["x-ai/grok-imagine-image-quality"]},
}
DEFAULT_MODEL = IMAGE_PROVIDERS["flux"]["default_model"]

IMAGE_STYLES = {
    "realistic": "photorealistic, high detail, 4K, natural lighting, professional photography",
    "cartoon": "cartoon style, animated, colorful, fun, character illustration",
    "minimalist": "minimalist, clean design, flat illustration, simple, modern",
    "abstract": "abstract art, geometric shapes, artistic, creative composition",
    "infographic": "infographic style, diagram, data visualization, clean layout",
}


def build_image_prompt(d: dict) -> str:
    """Prompt à partir du synopsis : titre + sujets des chapitres."""
    sujets = [str(c.get("subject", "")).strip() for c in (d.get("chapitres") or []) if isinstance(c, dict)]
    sujets = [s for s in sujets if s]
    prompt = f"Illustration pour la vidéo « {d.get('titre', '')} » : " + ", ".join(sujets)
    return re.sub(r"\s+", " ", prompt).strip()


def build_insight_prompt(insight_text: str, video_title: str, style: str = "realistic") -> str:
    """Prompt optimisé à partir d'un insight (logique d'origine)."""
    title = "Moment fort"
    for line in insight_text.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and len(line) > 5:
            cand = re.sub(r"^\d+[.\)]\s*", "", line)
            cand = re.sub(r"\[?\d{1,2}:\d{2}\]?\s*:?\s*", "", cand)[:60].strip()
            if cand:
                title = cand
                break
    clean = re.sub(r"\[?\d{1,2}:\d{2}\]?\s*:?\s*", "", insight_text)
    clean = re.sub(r"[#*_`]", "", clean)[:200].strip()
    prompt = re.sub(r"\s+", " ", f"'{title}' - {clean}. {IMAGE_STYLES.get(style, IMAGE_STYLES['realistic'])}. Topic: {video_title}").strip()
    return prompt[:400] + "..." if len(prompt) > 400 else prompt


def _call_openrouter(prompt: str, model: str, api_key: str, size: str = "1024x1024") -> dict:
    import httpx
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://synopsis.local", "X-Title": "Synopsis"}
    payload = {"model": model, "modalities": ["image"], "messages": [{"role": "user", "content": prompt}], "max_tokens": 5000}
    for attempt in range(3):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                time.sleep((attempt + 1) * 15)
                continue
            if resp.status_code == 401:
                return {"error": "Clé API refusée par OpenRouter."}
            if resp.status_code == 402:
                return {"error": "Crédit API épuisé sur OpenRouter."}
            if resp.status_code != 200:
                return {"error": f"OpenRouter {resp.status_code}: {resp.text[:200]}"}
            data = resp.json()
            if data.get("error"):
                err = data["error"]
                return {"error": err.get("message", str(err)) if isinstance(err, dict) else str(err)}
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message", {})
            for img in msg.get("images", []) or []:
                if img.get("type") == "image_url":
                    u = img.get("image_url", {}).get("url", "")
                    if u:
                        return {"url": u, "revised": prompt}
            content = msg.get("content", "") or ""
            if content and ("data:image" in content or "http" in content):
                m = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", content) or re.search(r"(https?://[^\s)]+)", content)
                return {"url": m.group(1) if m else content.strip(), "revised": prompt}
            if choice.get("finish_reason") == "error":
                return {"error": "Le modèle n'a pas généré d'image"}
            return {"error": "Aucune image dans la réponse"}
        except httpx.TimeoutException:
            if attempt < 2:
                time.sleep(5)
                continue
            return {"error": "Timeout OpenRouter"}
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return {"error": str(e)}
    return {"error": "Échec après tentatives"}


def generate_image_result(prompt: str, provider: str = "flux", model: Optional[str] = None,
                          size: str = "1024x1024", api_key: Optional[str] = None) -> dict:
    """Logique d'origine : renvoie {success, image_url, ...} (URL ou data URL)."""
    if not api_key:
        return {"success": False, "error": "API key non configurée."}
    cfg = IMAGE_PROVIDERS.get(provider)
    if not cfg:
        return {"success": False, "error": f"Provider '{provider}' non supporté."}
    model = model or cfg["default_model"]
    # Tous les fournisseurs restants passent par OpenRouter : la clé n'est jamais envoyée ailleurs.
    result = _call_openrouter(prompt, model, api_key, size)
    if "error" in result:
        return {"success": False, "error": result["error"]}
    return {"success": True, "image_url": result["url"], "revised_prompt": result.get("revised", prompt),
            "provider": provider, "model": model}


def generate_image(prompt: str, api_key: str, model: str = "") -> bytes:
    """Génère une image et renvoie ses octets. Lève RuntimeError en cas d'échec."""
    model = model or DEFAULT_MODEL
    # Sécurité : la clé est une clé OpenRouter, donc toujours le chemin OpenRouter, quel que soit le modèle.
    res = generate_image_result(prompt, provider="flux", model=model, api_key=api_key)
    if not res.get("success"):
        raise RuntimeError(str(res.get("error") or "Échec de la génération d'image"))
    url = res["image_url"]
    if url.startswith("data:"):
        try:
            return base64.b64decode(url.split(",", 1)[1])
        except Exception:
            raise RuntimeError("Image illisible dans la réponse")
    import httpx
    if not url.startswith("https://"):
        raise RuntimeError("URL d'image refusée (https uniquement).")
    r = httpx.get(url, timeout=30, follow_redirects=False)
    if r.status_code != 200:
        raise RuntimeError(f"Téléchargement de l'image impossible ({r.status_code})")
    if not str(r.headers.get("content-type", "")).lower().startswith("image/"):
        raise RuntimeError("Le contenu téléchargé n'est pas une image.")
    return r.content
