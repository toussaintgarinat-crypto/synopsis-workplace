"""Export Google Drive (reprise de youtube-summarizer/src/drive_exporter.py).

Les identifiants sont un JSON OAuth : soit {"access_token": "..."}, soit
{"client_id", "client_secret", "refresh_token"} (le jeton d'accès est alors rafraîchi).
httpx est importé à la demande.
"""
import json
import os
import tempfile
from typing import Optional

from lib.exporters.markdown import to_markdown

TOKEN_URL = "https://oauth2.googleapis.com/token"
FILES_URL = "https://www.googleapis.com/drive/v3/files"
UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
FOLDER_NAME = "Synopsis"

MIME_TYPES = {
    "txt": "text/plain", "md": "text/markdown", "pdf": "application/pdf", "json": "application/json",
    "excalidraw": "application/json", "png": "image/png", "jpg": "image/jpeg",
    "mp3": "audio/mpeg", "wav": "audio/wav",
}


def get_upload_mime_type(file_type: str) -> str:
    return MIME_TYPES.get(file_type, "application/octet-stream")


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> dict:
    import httpx
    resp = httpx.post(TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token"}, timeout=30)
    if resp.status_code != 200:
        return {"success": False, "error": f"Erreur refresh: {resp.text[:200]}"}
    return {"success": True, "tokens": resp.json()}


def _find_or_create_folder(headers: dict, folder_name: str) -> Optional[str]:
    import httpx
    safe = folder_name.replace("'", "")
    query = f"name='{safe}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    resp = httpx.get(FILES_URL, headers=headers, params={"q": query, "fields": "files(id,name)"}, timeout=15)
    if resp.status_code == 200:
        files = resp.json().get("files", [])
        if files:
            return files[0]["id"]
    resp = httpx.post(FILES_URL, headers=headers, json={
        "name": folder_name, "mimeType": "application/vnd.google-apps.folder"}, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("id")
    return None


def upload_to_drive(file_path: str, file_name: str, mime_type: str, access_token: str,
                    folder_name: str = FOLDER_NAME) -> dict:
    import httpx
    headers = {"Authorization": f"Bearer {access_token}"}
    folder_id = _find_or_create_folder(headers, folder_name)
    if not folder_id:
        return {"success": False, "error": "Impossible de créer le dossier Drive"}
    metadata = json.dumps({"name": file_name, "parents": [folder_id], "mimeType": mime_type})
    try:
        with open(file_path, "rb") as f:
            files = {"metadata": ("metadata", metadata, "application/json"),
                     "file": (file_name, f, mime_type)}
            resp = httpx.post(UPLOAD_URL, headers=headers, files=files, timeout=120)
    except Exception as e:
        return {"success": False, "error": f"Upload: {e}"}
    if resp.status_code == 200:
        file_id = resp.json().get("id", "")
        return {"success": True, "file_id": file_id, "file_name": file_name,
                "web_view_link": f"https://drive.google.com/file/d/{file_id}/view"}
    if resp.status_code == 401:
        return {"success": False, "error": "Token expiré — reconnectez-vous"}
    return {"success": False, "error": f"Drive {resp.status_code}: {resp.text[:200]}"}


def load_credentials(value: str) -> str:
    """Accepte le contenu JSON ou un chemin vers un fichier JSON ; renvoie le JSON."""
    value = (value or "").strip()
    if value.startswith("{"):
        return value
    if value and os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read()
    raise ValueError("GOOGLE_DRIVE_CREDENTIALS invalide (JSON ou chemin de fichier attendu).")


def _access_token(credentials_json: str) -> str:
    try:
        cred = json.loads(credentials_json)
    except (TypeError, ValueError):
        raise ValueError("Identifiants Drive illisibles (JSON attendu).")
    if not isinstance(cred, dict):
        raise ValueError("Identifiants Drive illisibles (objet JSON attendu).")
    if cred.get("access_token"):
        return cred["access_token"]
    if cred.get("refresh_token") and cred.get("client_id") and cred.get("client_secret"):
        r = refresh_access_token(cred["client_id"], cred["client_secret"], cred["refresh_token"])
        if r["success"] and r["tokens"].get("access_token"):
            return r["tokens"]["access_token"]
        raise RuntimeError(r.get("error") or "Rafraîchissement du jeton impossible")
    raise ValueError("Identifiants Drive incomplets (access_token ou refresh_token + client_id + client_secret).")


def upload_markdown(d: dict, credentials_json: str) -> str:
    """Dépose le synopsis en Markdown dans Drive ; renvoie l'URL du fichier."""
    token = _access_token(credentials_json)
    name = (str(d.get("titre") or "Synopsis").replace("/", "-").strip() or "Synopsis") + ".md"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
    try:
        tmp.write(to_markdown(d))
        tmp.close()
        res = upload_to_drive(tmp.name, name, get_upload_mime_type("md"), token)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if not res.get("success"):
        raise RuntimeError(res.get("error") or "Échec de l'export Drive")
    return res["web_view_link"]
