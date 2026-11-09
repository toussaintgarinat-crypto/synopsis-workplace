# Synopsis Workplace

**Résumé vidéo YouTube par IA** — conçu pour fonctionner en standalone **ou** comme brique du projet [Workplace](https://github.com/anomalyco/youtube-summarizer).

## Quick Start

```bash
# Standalone
export OPENROUTER_API_KEY="sk-or-..."
pip install -r requirements.txt      # mode léger (jeu minimal, utilisé par Vercel)
# pip install -r requirements-full.txt  # mode complet (inclut requirements.txt)
python main.py

# Test
curl -X POST http://localhost:6090/resumer -H "Content-Type: application/json" -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "langue": "Français"}'
```

## Modes de fonctionnement

| Mode | Cible | Fonctions |
|------|-------|-----------|
| `full` (défaut) | local / Docker | Tout : Whisper, reel (ffmpeg), LLM local, Obsidian, Drive, images, TTS |
| `light` | Vercel (serverless) | Résumé, Q&R, export Markdown ; fonctions lourdes désactivées (Whisper, reel, LLM local, Obsidian, Drive, Excalidraw, images, TTS) |

### Variables d'environnement

| Variable | Rôle |
|----------|------|
| `SYNOPSIS_MODE` | `full` (défaut) ou `light`. Automatiquement `light` si `VERCEL` est défini |
| `LOCAL_LLM_URL` | URL du LLM local (ex. Ollama : `http://localhost:11434`, depuis Docker : `http://host.docker.internal:11434`) |
| `WHISPER_MODEL` | Modèle Whisper local (défaut `base`) |
| `GOOGLE_DRIVE_CREDENTIALS` | Identifiants Google Drive pour `/export/drive` |
| `OPENAI_API_KEY` | Transcription Whisper via API / fournisseur d'images |
| `OPENROUTER_API_KEY` | Clé LLM OpenRouter |
| `MAX_MINUTES_LIGHT` | Durée max (minutes) d'une vidéo en mode léger (défaut 30) |
| `HOST` | Adresse d'écoute (défaut `127.0.0.1`) |
| `ALLOWED_ORIGINS` / `ALLOWED_HOSTS` | Origines CORS et hôtes acceptés (voir Sécurité) |
| `ALLOW_SERVER_KEY` | `1` : en mode léger, autorise les requêtes LLM sans clé client (utilise la clé du serveur). Par défaut refusé, pour ne pas offrir votre clé à tout le monde |
| `MAX_BATCH_LIGHT` / `MAX_BATCH_FULL` | Nombre max d'URL par lot (défaut 3 / 20). L'interface résume les vidéos une par une (`/resumer`) |
| `MAX_UPLOAD_MB` | Taille max d'un fichier pour `/resumer-fichier` (défaut 200, 413 au-delà) |
| `MAX_BODY_MB` | Taille max du corps des requêtes JSON (défaut 5, 413 au-delà) ; `/resumer-fichier` est plafonné par `MAX_UPLOAD_MB` |
| `RATE_LIMIT_PER_MIN` | Requêtes LLM max par minute et par IP (`/resumer*`, `/qa`, `/reel`, `/image`). Actif par défaut (20) en mode léger, ou dès que la variable est définie ; `0` désactive |
| `OBSIDIAN_ALLOWED_ROOTS` | Racines autorisées pour le dossier du vault Obsidian (séparées par `:`), défaut : dossier personnel |
| `CUSTOM_LLM_URL` | URL de base OpenAI-compatible du fournisseur « Personnalisé » (ex. `http://hote:8000/v1`) ; lue uniquement côté serveur. Option désactivée si absente |

### Lancement local (mode complet)

```bash
pip install -r requirements-full.txt
set -a; . ./.env; set +a; python3 main.py
```

### Docker (mode complet, avec ffmpeg)

```bash
docker compose up --build
```

Le `.env` est lu au runtime (`env_file`) et exclu de l'image via `.dockerignore`. Le build installe `openai-whisper` (lourd : première construction longue).

### Dépendances

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `requirements.txt` | jeu léger | Vercel (lu automatiquement), mode `light` |
| `requirements-full.txt` | `-r requirements.txt` + gtts, yt-dlp, fpdf2, openai-whisper | mode `full`, Docker |
| `requirements-dev.txt` | `-r requirements-full.txt` + pytest | développement et tests |

Tests : `pip install -r requirements-dev.txt && python3 -m pytest tests -q`

### Vercel (mode léger)

Les fichiers `vercel.json`, `api/index.py` et `requirements.txt` (jeu léger, seul fichier lu par Vercel) sont fournis. Définir `SYNOPSIS_MODE=light` dans les variables d'environnement du projet Vercel (Settings > Environment Variables) ; le mode léger est aussi activé automatiquement quand `VERCEL` est présent. Durée max de fonction : 60 s. `vercel.json` inclut `prompts/**` dans la fonction (`includeFiles`) pour que les prompts XML soient déployés. En mode léger, l'utilisateur doit saisir sa propre clé API (sauf `ALLOW_SERVER_KEY=1`).

## Mode Gateway (Workplace)

Quand la variable `GATEWAY_URL` est définie, Synopsis détecte automatiquement le Gateway Workplace et route tous les appels LLM via celui-ci :

```bash
export GATEWAY_URL="http://host.docker.internal:4001"
export GATEWAY_KEY="sk-wp-..."
python main.py
```

## Endpoints API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/sante` | Health check (+ statut Gateway) |
| `POST` | `/resumer` | `{"url": "...", "langue": "Français"}` → résumé structuré |
| `POST` | `/reel` | `{"url": "...", "duree_clip": 45}` → highlight reel |
| `POST` | `/resumer-lot` | Résumé de plusieurs vidéos en lot |
| `POST` | `/resumer-fichier` | Résumé d'un fichier envoyé (multipart) |
| `POST` | `/qa` | Question/réponse sur un résumé / transcript |
| `POST` | `/export/markdown` | Export Markdown |
| `POST` | `/export/obsidian` | Export vers un coffre Obsidian (mode complet) |
| `POST` | `/export/excalidraw` | Export Excalidraw |
| `POST` | `/export/drive` | Export Google Drive (mode complet) |
| `POST` | `/image` | Génération d'image (mode complet) |
| `GET` | `/i18n/{lang}` | Textes de l'interface dans la langue demandée |

## Sécurité

- **Liaison réseau** : le serveur écoute sur `127.0.0.1` par défaut (variable `HOST`). L'image Docker fixe `HOST=0.0.0.0` dans le conteneur ; `docker-compose.yml` publie le port uniquement sur `127.0.0.1:6090`.
- **CORS restreint** : `ALLOWED_ORIGINS` (liste séparée par des virgules, défaut `http://localhost:6090,http://127.0.0.1:6090`, plus le `PORT` courant).
- **Contrôle d'origine** : toute requête POST/PUT/DELETE portant un en-tête `Origin` (ou `Referer`) d'une origine non autorisée reçoit un 403 (protège `/export/obsidian`, `/export/drive`, `/image`, `/qa`, `/resumer*`, `/reel` contre les pages web tierces). Sans `Origin`/`Referer` (curl, appels serveur à serveur comme la passerelle Workplace) la requête passe ; une origine identique à l'hôte demandé passe aussi.
- **Hôtes de confiance** : `localhost`, `127.0.0.1`, `host.docker.internal`, plus `ALLOWED_HOSTS` (liste séparée par des virgules, ex. le nom de service Docker utilisé par Workplace) et `*.vercel.app` quand `VERCEL` est défini.
- **Sec-Fetch-Site** : une requête POST marquée `cross-site` (ou `same-site` d'une origine non autorisée) par le navigateur est refusée (403), même sans `Origin`. L'origine identique à l'hôte n'est acceptée que sous la forme exacte `scheme://hôte[:port]`.
- **En-têtes** : `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin` (et non `no-referrer`, qui ferait envoyer `Origin: null` aux POST de la page), CSP sur toutes les réponses. La page `/` a une CSP stricte : `default-src 'none'`, script inline et gestionnaires `onclick` autorisés uniquement par empreinte SHA-256 (`'unsafe-hashes'`, pas de `'unsafe-inline'` pour les scripts), `connect-src 'self'`, `frame-ancestors 'none'`. Les exports HTML/PDF portent leur propre CSP (`default-src 'none'`) : aucun script ne s'y exécute.
- **Vault Obsidian** : chemin absolu (ou `~`), situé après résolution des liens symboliques dans le dossier personnel (ou `OBSIDIAN_ALLOWED_ROOTS`), sans dossier caché (`.ssh`, `.claude`, `.config`...) ni `~/Library` (sauf `Mobile Documents` / `CloudStorage`, pour les vaults iCloud). Une note existante n'est jamais écrasée sauf si elle a été créée par Synopsis (front-matter `tags: [synopsis, youtube]`).
- **Limite de débit** : en mémoire, par IP. Sur Vercel (serverless), chaque instance a sa propre mémoire : la limite est indicative (best-effort) ; pour une vraie limite, utiliser un pare-feu/WAF ou un stockage partagé.
- **Erreurs et journaux** : les messages d'erreur renvoyés au client sont tronqués et purgés de tout ce qui ressemble à un secret (`sk-...`, `Bearer ...`, jetons Google, valeurs des variables `*KEY*`/`*TOKEN*`/`*SECRET*`) ; même filtrage sur toutes les entrées de journal.
- **Clé API dans le navigateur** : la clé saisie dans l'interface est conservée dans le `localStorage` du navigateur (`synopsis_api_key_<fournisseur>`), en clair, pour l'origine de l'application uniquement. Elle est envoyée au serveur Synopsis à chaque appel LLM, jamais dans une URL. Sur un poste partagé, videz-la (champ vide puis changement de champ) ou les données du site.
- **Reel** : l'URL est réduite à son identifiant YouTube puis reconstruite (`https://www.youtube.com/watch?v=ID`) avant yt-dlp, appelé avec `--` (pas d'injection d'options ni de requête vers un hôte arbitraire).
- Ne pas exposer publiquement une instance contenant des clés (`.env`) sans authentification (reverse proxy).

## Intégration Workplace

1. Copier ce dossier dans `~/Desktop/Workplace/briques/synopsis/`
2. `cd ~/Desktop/Workplace/briques/synopsis && make up`
3. `curl -X POST http://localhost:5000/briques/reload`
4. L'assistant IA Workplace auto-découvre les capacités `youtube_resumer` et `youtube_reel`

## Licence

MIT
