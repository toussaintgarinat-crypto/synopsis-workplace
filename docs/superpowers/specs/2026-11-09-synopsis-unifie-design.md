# Synopsis unifié — design

## Objectif
Une seule application (code unique) qui regroupe le maximum de fonctions des trois projets :
`synopsis-workplace` (base), `youtube-summarizer` (desktop/Streamlit) et la page `synopsis-jet.vercel.app`.
Le front conservé est celui de `localhost:6090` (thème orange, intégré à `main.py`).

## Base technique
- Backend : FastAPI (`main.py`), modules dans `lib/`.
- Dépôt cible : `toussaintgarinat-crypto/youtube-summarizer`, sur une branche (remplace Streamlit/GUI, à confirmer avant push).
- Un seul code, deux modes :
  - **Complet** (local / Docker) : toutes les fonctions.
  - **Léger** (Vercel) : sous-titres YouTube seulement, durée limitée (~30 min).
- `GET /sante` renvoie `capacites` (liste des fonctions disponibles). Le front affiche ou grise chaque bouton selon cette liste.

## Fonctions (cible : le maximum)
| Fonction | Origine | Mode |
|---|---|---|
| Résumé, chapitres, insights (corrigés) | workplace | léger + complet |
| Export HTML, PDF (impression), Markdown | workplace + Vercel | léger + complet |
| Plusieurs vidéos à la fois | Vercel | léger + complet |
| Questions-réponses sur la vidéo (`/qa`) | Vercel | léger + complet |
| Interface FR / EN | Vercel | léger + complet |
| Fournisseurs LLM : OpenRouter, OpenCode Go, OpenAI, personnalisé | workplace + Vercel | léger + complet |
| LLM local (Ollama / LM Studio, API compatible OpenAI) | youtube-summarizer | complet |
| Transcription Whisper, upload audio/vidéo | youtube-summarizer | complet |
| Reel vidéo, narration TTS, export vertical | workplace | complet |
| Export Obsidian, Drive, Excalidraw | youtube-summarizer | complet |
| Génération d'images | youtube-summarizer | complet |
| Mode Gateway Workplace | workplace | complet |

## Découpage
- `lib/capabilities.py` : détecte ce qui est disponible (ffmpeg, whisper, ollama, clés, mode).
- `lib/llm_client.py` : ajout du fournisseur local et de la liste de modèles locaux.
- `lib/qa.py`, `lib/batch.py` : questions-réponses et traitement multi-vidéos.
- `lib/exporters/` : `html`, `markdown`, `pdf`, `obsidian`, `drive`, `excalidraw`.
- `lib/whisper_transcriber.py`, `lib/image_generator.py` : repris de `youtube-summarizer/src/`.
- Front : `main.py` reste l'unique page ; lecture de `capacites` au chargement.

## Gestion des erreurs
- Fonction indisponible : bouton grisé avec la raison (« nécessite Whisper », « mode en ligne »).
- Échec d'un chunk ou d'une vidéo du lot : la suite continue, l'erreur est affichée sur l'élément concerné.
- Modèle local qui renvoie un tableau de chapitres mal formé : message clair, résumé brut conservé.

## Tests
- Tests unitaires de l'extraction des chapitres/insights (formats variés), du détecteur de capacités et des exporteurs.
- Lancement réel de l'application et vérification des routes à chaque étape.

## Étapes
1. Front actuel corrigé + exports HTML/PDF (fait), ajout Markdown.
2. Capacités, multi-vidéos, questions-réponses, FR/EN, LLM local.
3. Whisper/upload, Obsidian, Drive, Excalidraw, images.
4. Configuration Vercel (mode léger) et push sur `youtube-summarizer` après confirmation.

## Hors périmètre
Les applications GUI natives (PyInstaller, `.app`, Windows/Linux) et la PWA ne sont pas reprises : le front web les remplace.
