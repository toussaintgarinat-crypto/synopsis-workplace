"""LLM Client — Gateway-aware with standalone fallback.

Priority:
  1. GATEWAY_URL (Workplace LiteLLM) — OpenAI-compatible /v1/chat/completions
  2. OPENROUTER_API_KEY — direct OpenRouter API
  3. OPENCODE_GO_API_KEY — direct OpenCode Go API
"""

import json
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
MAX_RETRIES = 2


OPENROUTER_MODELS = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "openai/gpt-5.2",
    "openai/gpt-5.2-mini",
    "qwen/qwen3-coder",
    "mistral/pi-zero-2",
    "x-ai/grok-4",
    "amazon/nova-pro-v2",
]

OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5.2-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o4-mini",
    "o3",
]

OPENCODE_MODELS = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
]

PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "opencode": "https://opencode.ai/zen/go/v1/chat/completions",
}

PROVIDER_EXTRA_HEADERS = {
    "openrouter": {"HTTP-Referer": "https://synopsis.local", "X-Title": "Synopsis"},
    "openai": {},
    "opencode": {},
}

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


def custom_base_url() -> str:
    """URL de base du fournisseur personnalisé : UNIQUEMENT via l'environnement serveur
    (jamais depuis la requête : évite SSRF et fuite de clé)."""
    return os.getenv("CUSTOM_LLM_URL", "").rstrip("/")


def list_custom_models(api_key: str = "") -> list[str]:
    base = custom_base_url()
    if not base:
        return []
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = httpx.get(f"{base}/models", headers=headers, timeout=10)
        if r.status_code == 200:
            return sorted(m["id"] for m in r.json().get("data", []) if m.get("id"))
    except Exception:
        pass
    return []


def llm_complete(prompt: str, model: str = "", system: str = "", temperature: float = 0.3, api_key: str = "", provider: str = "") -> str:
    model = model or DEFAULT_MODEL
    gateway_url = os.getenv("GATEWAY_URL", "").rstrip("/")
    gateway_key = os.getenv("GATEWAY_KEY", os.getenv("LITELLM_MASTER_KEY", ""))

    if gateway_url:
        return _complete_openai_compatible(
            f"{gateway_url}/v1/chat/completions",
            prompt, model, system, temperature,
            api_key=gateway_key,
        )

    if provider == "local":
        return _complete_openai_compatible(
            f"{LOCAL_LLM_URL}/v1/chat/completions",
            prompt, model, system, temperature, api_key="ollama",
        )

    if provider == "custom":
        base = custom_base_url()
        if not base:
            raise RuntimeError("Fournisseur personnalisé non configuré (CUSTOM_LLM_URL).")
        return _complete_openai_compatible(
            f"{base}/chat/completions", prompt, model, system, temperature, api_key=api_key,
        )

    if provider and api_key:
        base_url = PROVIDER_BASE_URLS.get(provider, PROVIDER_BASE_URLS["openrouter"])
        return _complete_openai_compatible(
            base_url, prompt, model, system, temperature,
            api_key=api_key,
            extra_headers=PROVIDER_EXTRA_HEADERS.get(provider, {}),
        )

    openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        return _complete_openai_compatible(
            "https://openrouter.ai/api/v1/chat/completions",
            prompt, model, system, temperature,
            api_key=openrouter_key,
            extra_headers={"HTTP-Referer": "https://synopsis.local", "X-Title": "Synopsis"},
        )

    opencode_key = os.getenv("OPENCODE_GO_API_KEY", "")
    if opencode_key:
        return _complete_openai_compatible(
            "https://opencode.ai/zen/go/v1/chat/completions",
            prompt, model, system, temperature,
            api_key=opencode_key,
        )

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        return _complete_openai_compatible(
            "https://api.openai.com/v1/chat/completions",
            prompt, model, system, temperature,
            api_key=openai_key,
        )

    raise RuntimeError(
        "Aucune cle API configuree. "
        "Definissez GATEWAY_URL, OPENROUTER_API_KEY, OPENAI_API_KEY, OPENCODE_GO_API_KEY, "
        "ou saisissez votre cle dans l'interface."
    )


def _complete_openai_compatible(
    base_url: str,
    prompt: str,
    model: str,
    system: str = "",
    temperature: float = 0.3,
    api_key: str = "",
    extra_headers: dict = None,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 32000,
    }

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = httpx.post(base_url, json=payload, headers=headers, timeout=180)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                last_err = f"Rate limited (attempt {attempt + 1})"
                continue
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
        except httpx.RequestError as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(2)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES + 1} attempts: {last_err}")
