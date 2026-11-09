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
| `light` | Vercel (serverless) | Résumé, Q&R, export Markdown, transcription par modèle vidéo (clé OpenRouter du client, vidéos courtes) ; fonctions lourdes désactivées (Whisper, reel, LLM local, Obsidian, Drive, Excalidraw, images, TTS) |

### Variables d'environnement

| Variable | Rôle |
|----------|------|
| `SYNOPSIS_MODE` | `full` (défaut) ou `light`. Automatiquement `light` si `VERCEL` est défini |
| `LOCAL_LLM_URL` | URL du LLM local (ex. Ollama : `http://localhost:11434`, depuis Docker : `http://host.docker.internal:11434`) |
| `WHISPER_MODEL` | Modèle Whisper local (défaut `base`) ; l'interface peut choisir `tiny`, `base`, `small`, `medium` ou `large` par requête. **`large` est lourd** sur une machine auto-hébergée (environ 10 Go de RAM/VRAM, transcription très lente sur CPU) : préférez `base` ou `small` sans GPU |
| `LLM_VIDEO_TIMEOUT` / `LLM_VIDEO_TIMEOUT_LIGHT` | Délai (s) de la transcription par modèle vidéo (défaut 300 en mode complet, 50 en mode léger) |
| `GOOGLE_DRIVE_CREDENTIALS` | Identifiants Google Drive pour `/export/drive` |
| `OPENAI_API_KEY` | Transcription Whisper via API / fournisseur d'images |
| `OPENROUTER_API_KEY` | Clé LLM OpenRouter |
| `MAX_MINUTES_LIGHT` | Durée max (minutes) d'une vidéo en mode léger (défaut 30) |
| `LIGHT_BUDGET_S` | Mode léger : budget total (s) d'une requête, transcription et résumé compris (défaut 55, sous la limite Vercel de 60 s) |
| `LIGHT_RESERVE_RESUME_S` | Mode léger : temps (s) gardé pour le résumé avant d'appeler Gemini (défaut 20) ; s'il reste moins de 10 s de transcription, la requête est refusée avant l'appel payant |
| `HOST` | Adresse d'écoute (défaut `127.0.0.1`) |
| `ALLOWED_ORIGINS` / `ALLOWED_HOSTS` | Origines CORS et hôtes acceptés (voir Sécurité) |
| `ALLOW_SERVER_KEY` | `1` : en mode léger, autorise les requêtes LLM sans clé client (utilise la clé du serveur). Par défaut refusé, pour ne pas offrir votre clé à tout le monde |
| `MAX_BATCH_LIGHT` / `MAX_BATCH_FULL` | Nombre max d'URL par lot (défaut 3 / 20). L'interface résume les vidéos une par une (`/resumer`) |
| `MAX_UPLOAD_MB` | Taille max d'un fichier pour `/resumer-fichier` (défaut 200, 413 au-delà) |
| `MAX_BODY_MB` | Taille max du corps des requêtes JSON (défaut 5, 413 au-delà) ; `/resumer-fichier` est plafonné par `MAX_UPLOAD_MB` |
| `RATE_LIMIT_PER_MIN` | Requêtes LLM max par minute et par IP (`/resumer*`, `/qa`, `/reel`, `/image`). Actif par défaut (20) en mode léger, ou dès que la variable est définie ; `0` désactive. `POST /modeles` est limité dans un compteur séparé, à 3 fois cette valeur |
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

## Vidéos sans sous-titres : transcription par modèle vidéo

Quand une vidéo YouTube n'a pas de sous-titres (manuels ou automatiques), Synopsis peut la faire transcrire par **Gemini via OpenRouter**, puis résumer cette transcription comme d'habitude.

- **Conditions** : fournisseur `openrouter` (ou vide avec une clé OpenRouter) et option `transcription_llm` activée (case « Si la vidéo n'a pas de sous-titres, la transcrire avec un modèle vidéo », cochée par défaut). En mode léger, la clé doit être celle du client (sauf `ALLOW_SERVER_KEY=1`). Avec un autre fournisseur, le message d'erreur indique l'option.
- **Quand** : uniquement si YouTube n'a pas de sous-titres pour une vidéo lisible (sous-titres désactivés ou absents, transcript vide, requête bloquée par YouTube). Une vidéo supprimée, privée, d'âge restreint ou injouable n'est jamais envoyée à Gemini.
- **Clé du serveur** : sans clé client, `OPENROUTER_API_KEY` n'est utilisée que si la requête envoie explicitement `transcription_llm`, et jamais quand `GATEWAY_URL` est défini.
- **Modèles** : seuls les modèles Gemini (`google/...`) lisent une URL YouTube, et uniquement chez le fournisseur Google AI Studio. Synopsis force donc `"provider": {"only": ["google-ai-studio"], "allow_fallbacks": false}`. Modèle par défaut : `google/gemini-2.5-flash`.
- **Requête envoyée** : l'URL est réduite à l'identifiant de la vidéo puis reconstruite (`https://www.youtube.com/watch?v=ID`). La chaîne saisie n'est jamais transmise telle quelle.
- **Coût** : environ 18 000 jetons d'entrée par minute de vidéo, soit environ 0,007 $ par minute avec `gemini-2.5-flash`. C'est payé sur le compte OpenRouter de la clé utilisée.
- **Limites** :
  - En mode léger (Vercel, 60 s par fonction), toute la requête partage un budget de `LIGHT_BUDGET_S` (55 s) : l'appel Gemini est coupé à 50 s au plus, moins s'il ne reste pas `LIGHT_RESERVE_RESUME_S` pour le résumé ; les appels de résumé reçoivent le temps restant et, une fois épuisé, la réponse est un 504 « Délai dépassé… ». Si le temps manque avant la transcription : « Vidéo trop longue pour être transcrite en ligne… ». Réservez-le aux vidéos courtes.
  - `/resumer-lot` en mode léger : une seule échéance pour tout le lot ; les vidéos traitées après elle échouent avec « Délai dépassé… », sans appel payant.
  - La limite de 30 minutes (`MAX_MINUTES_LIGHT`) ne peut pas être vérifiée avant : sans sous-titres, la durée n'est connue qu'**après** la transcription, à partir du dernier horodatage renvoyé par le modèle (valeur approximative). La transcription est donc déjà payée quand une vidéo trop longue est refusée.
  - Les horodatages sont approximatifs. Un avertissement le signale dans le résultat.
- **Réponse** de `/resumer` :
  - `transcription` : `sous-titres`, `llm-video` ou `whisper` ;
  - `modele_transcription` : présent si `llm-video` ;
  - `modeles_utilises` : par exemple `{"resume": "...", "fusion": "..."}` ; pour un fichier, `transcription_audio` donne le modèle réellement utilisé (taille locale, ou `whisper-1` si le repli sur l'API OpenAI a servi).

## Modèles par tâche

Dans Configuration, la section repliable « Modèles par tâche » permet de choisir un modèle par tâche. Les choix sont mémorisés par fournisseur dans `localStorage` (`synopsis_modeles_<fournisseur>`). Seuls des identifiants de modèles y sont stockés.

| Tâche (`tache`) | Rôle | Défaut | Liste proposée |
|---|---|---|---|
| `resume` | analyse de chaque segment de transcription | modèle principal (`modele`) | catalogue du fournisseur |
| `fusion` | fusion des analyses (vidéos longues) | identique à `resume` | catalogue du fournisseur |
| `qa` | questions-réponses | identique à `resume` | catalogue du fournisseur |
| `transcription_video` | transcription des vidéos sans sous-titres | `google/gemini-2.5-flash` | OpenRouter `?input_modalities=video`, ids `google/` uniquement |
| `transcription_audio` | taille Whisper (fichier envoyé, mode complet) | `WHISPER_MODEL` ou `base` | `tiny`, `base`, `small`, `medium`, `large` (`large` : lourd en CPU/RAM sur une machine auto-hébergée) |
| `image` | génération d'image (mode complet) | modèle par défaut de `lib/image_generator` | OpenRouter `?output_modalities=image` |

**Champs de requête** :
- `modeles` : objet `{tâche: id}`, accepté par `/resumer`, `/resumer-lot`, `/qa` et `/image`. Pour `/resumer-fichier`, c'est un champ multipart contenant une chaîne JSON d'au plus 2000 caractères.
- `transcription_llm` : booléen, par défaut `true`, accepté par `/resumer` et `/resumer-lot`.

**Validation de `modeles`** :
- Une valeur vide prend le défaut.
- Les tâches inconnues sont ignorées.
- Chaque id doit respecter `^[A-Za-z0-9._:/-]{1,100}$`, sinon la requête reçoit un 400.
- `transcription_video` doit commencer par `google/`.
- `transcription_audio` doit faire partie de la liste ci-dessus.
- Le champ historique `modele` reste le défaut de `resume`. Pour `/image`, il reste le défaut de `image`.

**Paramètre `tache` de `/modeles`** :
- `GET /modeles?provider=openrouter&tache=image` ou `POST /modeles {"provider", "api_key", "tache"}`. La clé passe uniquement par le POST, jamais dans l'URL.
- Valeurs acceptées : les six tâches du tableau. Une valeur inconnue donne un 400.
- Sans `tache`, la réponse est inchangée : `{"models", "source"}`.
- Cache de 10 minutes, plafonné à 256 entrées. Les listes publiques (OpenRouter) sont partagées entre toutes les clés ; seule la liste OpenAI est mise en cache par clé (empreinte, jamais la clé).
- Avec `tache`, la réponse ajoute :
  - `tache` ;
  - `noms` (id → nom lisible, si l'API le fournit) ;
  - `raison`, quand la tâche n'existe pas pour ce fournisseur (vidéo et image : OpenRouter uniquement).
- Les listes OpenRouter sont mises en cache 10 minutes en mémoire.

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
| `POST` | `/resumer` | `{"url": "...", "langue": "Français", "modeles": {...}, "transcription_llm": true}` → résumé structuré |
| `POST` | `/reel` | `{"url": "...", "duree_clip": 45}` → highlight reel |
| `POST` | `/resumer-lot` | Résumé de plusieurs vidéos en lot |
| `POST` | `/resumer-fichier` | Résumé d'un fichier envoyé (multipart) |
| `POST` | `/qa` | Question/réponse sur un résumé / transcript |
| `POST` | `/export/markdown` | Export Markdown |
| `POST` | `/export/obsidian` | Export vers un coffre Obsidian (mode complet) |
| `POST` | `/export/excalidraw` | Export Excalidraw |
| `POST` | `/export/drive` | Export Google Drive (mode complet) |
| `POST` | `/image` | Génération d'image (mode complet) |
| `GET`/`POST` | `/modeles` | Modèles disponibles (`provider`, `tache` ; clé en POST uniquement) |
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
