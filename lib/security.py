"""Briques de sécurité transverses : masquage des secrets, plafond de taille des corps,
limite de débit en mémoire, politique CSP de la page."""
import base64
import collections
import hashlib
import json
import logging
import os
import re
import threading
import time
import traceback

# ---------- Masquage des secrets ----------

_MOTIFS = [
    re.compile(r"Bearer\s+[^\s\"',;}]+", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_\-*.]{6,}"),          # OpenAI / OpenRouter / OpenCode
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),            # clés Google
    re.compile(r"\bya29\.[0-9A-Za-z_\-.]+"),             # jetons d'accès Google
    re.compile(r"\b1//[0-9A-Za-z_\-]{20,}"),             # refresh tokens Google
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\br8_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)((?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password|token)"
               r"[\"']?\s*[:=]\s*[\"']?)[^\s\"',;}&]+"),
]
_NOMS_SECRETS = re.compile(r"KEY|TOKEN|SECRET|CREDENTIAL|PASSWORD", re.I)
MASQUE = "[masqué]"


def _secrets_env() -> list[str]:
    vals = []
    for k, v in os.environ.items():
        if _NOMS_SECRETS.search(k) and v and len(v) >= 8:
            vals.append(v)
            # Identifiants JSON (Drive) : masquer aussi chaque valeur interne.
            if v.lstrip().startswith("{"):
                try:
                    vals += [str(x) for x in json.loads(v).values() if isinstance(x, str) and len(x) >= 8]
                except (ValueError, AttributeError):
                    pass
    return sorted(set(vals), key=len, reverse=True)


def scrub(texte, limite: int = 0) -> str:
    """Retire tout ce qui ressemble à une clé/jeton ; tronque si `limite` > 0."""
    s = str(texte)
    for v in _secrets_env():
        s = s.replace(v, MASQUE)
    for m in _MOTIFS:
        s = m.sub(lambda x: (x.group(1) if m.groups else "") + MASQUE, s)
    if limite and len(s) > limite:
        s = s[:limite] + "…"
    return s


def installer_masquage_journaux() -> None:
    """Toute entrée de journal (tous loggers, y compris tracebacks) passe par scrub()."""
    ancienne = logging.getLogRecordFactory()
    if getattr(ancienne, "_synopsis_scrub", False):
        return

    def fabrique(*a, **k):
        rec = ancienne(*a, **k)
        try:
            rec.msg = scrub(rec.getMessage())
            rec.args = None
            if rec.exc_info and rec.exc_info[0] is not None:
                rec.exc_text = scrub("".join(traceback.format_exception(*rec.exc_info)).rstrip("\n"))
        except Exception:  # ne jamais casser la journalisation
            pass
        return rec

    fabrique._synopsis_scrub = True
    logging.setLogRecordFactory(fabrique)


# ---------- Plafond de taille du corps des requêtes ----------

def _mo(nom: str, defaut: str) -> int:
    try:
        return int(float(os.getenv(nom, defaut)) * 1024 * 1024)
    except ValueError:
        return int(float(defaut) * 1024 * 1024)


class _TropGros(Exception):
    pass


class BodySizeLimit:
    """ASGI : 413 si le corps dépasse MAX_BODY_MB (défaut 5). `/resumer-fichier` est
    plafonné à MAX_UPLOAD_MB (+1 Mo pour l'enveloppe multipart). Compte aussi les corps
    envoyés sans Content-Length (chunked)."""

    UPLOAD_PATHS = ("/resumer-fichier",)

    def __init__(self, app):
        self.app = app

    def _limite(self, path: str) -> int:
        if path in self.UPLOAD_PATHS:
            return _mo("MAX_UPLOAD_MB", "200") + 1024 * 1024
        return _mo("MAX_BODY_MB", "5")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limite = self._limite(scope.get("path", ""))
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > limite:
                        return await _json(send, 413, "Requête trop volumineuse.")
                except ValueError:
                    return await _json(send, 400, "Content-Length invalide.")
        recu = 0
        envoye = False
        depasse = False

        async def recevoir():
            nonlocal recu, depasse
            msg = await receive()
            if msg["type"] == "http.request":
                recu += len(msg.get("body", b""))
                if recu > limite:
                    depasse = True
                    raise _TropGros()
            return msg

        async def envoyer(msg):
            # FastAPI transforme une erreur de lecture du corps en 400 : on la remplace par 413.
            nonlocal envoye
            if depasse:
                if msg["type"] == "http.response.start" and not envoye:
                    envoye = True
                    await _json(send, 413, "Requête trop volumineuse.")
                return
            if msg["type"] == "http.response.start":
                envoye = True
            await send(msg)

        try:
            await self.app(scope, recevoir, envoyer)
        except _TropGros:
            if not envoye:
                await _json(send, 413, "Requête trop volumineuse.")
        except Exception as e:  # l'exception peut être enveloppée (ExceptionGroup / middleware)
            if _contient_trop_gros(e) and not envoye:
                await _json(send, 413, "Requête trop volumineuse.")
            else:
                raise


def _contient_trop_gros(e: BaseException) -> bool:
    vus = set()
    while e is not None and id(e) not in vus:
        vus.add(id(e))
        if isinstance(e, _TropGros):
            return True
        for sous in getattr(e, "exceptions", ()) or ():
            if _contient_trop_gros(sous):
                return True
        e = e.__cause__ or e.__context__
    return False


async def _json(send, status: int, detail: str, headers: list = None):
    corps = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(corps)).encode())] + (headers or [])})
    await send({"type": "http.response.body", "body": corps})


# ---------- Limite de débit ----------

class RateLimiter:
    """Fenêtre glissante d'une minute par IP, en mémoire du processus.
    Sur Vercel (serverless), chaque instance a sa propre mémoire : limite indicative."""

    def __init__(self):
        self._hits = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._hits.clear()

    def autoriser(self, cle: str, limite: int, fenetre: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[cle]
            while q and now - q[0] >= fenetre:
                q.popleft()
            if len(q) >= limite:
                return False
            q.append(now)
            if len(self._hits) > 10000:  # borne mémoire
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= fenetre]:
                    del self._hits[k]
            return True


RATE_LIMITER = RateLimiter()
LLM_PATHS = ("/resumer", "/resumer-lot", "/resumer-fichier", "/qa", "/reel", "/image")


def limite_par_minute(light: bool) -> int:
    """0 = pas de limite. Active en mode léger (défaut 20) ou si RATE_LIMIT_PER_MIN est défini."""
    brut = os.getenv("RATE_LIMIT_PER_MIN")
    if brut is None or not brut.strip():
        return 20 if light else 0
    try:
        return max(0, int(brut))
    except ValueError:
        return 20


def ip_client(request) -> str:
    # Sur Vercel, x-forwarded-for est réécrit par la plateforme ; ailleurs il est falsifiable.
    if os.getenv("VERCEL"):
        xff = request.headers.get("x-forwarded-for", "")
        if xff.strip():
            return xff.split(",")[0].strip()
        if request.headers.get("x-real-ip"):
            return request.headers["x-real-ip"].strip()
    return request.client.host if request.client else "inconnu"


# ---------- En-têtes / CSP ----------

def _sha(texte: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(texte.encode("utf-8")).digest()).decode() + "'"


def csp_page(html: str) -> str:
    """CSP de la page : script inline autorisé par empreinte (pas de 'unsafe-inline'),
    gestionnaires onclick autorisés par empreinte via 'unsafe-hashes'."""
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    handlers = sorted(set(re.findall(r'\son[a-z]+="([^"]*)"', html)))
    script_src = " ".join(["'self'"] + [_sha(s) for s in scripts] + (["'unsafe-hashes'"] + [_sha(h) for h in handlers] if handlers else []))
    return "; ".join([
        "default-src 'none'",
        f"script-src {script_src}",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data: blob:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
    ])


CSP_API = "default-src 'none'; frame-ancestors 'none'"
EN_TETES = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    # 'same-origin' plutôt que 'no-referrer' : avec 'no-referrer', les navigateurs envoient
    # « Origin: null » sur les POST de même origine, ce qui casserait le contrôle d'origine.
    "referrer-policy": "same-origin",
}
