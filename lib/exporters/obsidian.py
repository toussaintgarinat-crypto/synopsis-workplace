import os
import re
from pathlib import Path
from .markdown import to_markdown

MARQUEUR = "tags: [synopsis, youtube]"
# Sous ~/Library (macOS), seuls les emplacements de synchronisation (iCloud, Dropbox, Google Drive...) sont admis.
LIBRARY_OK = (("Library", "Mobile Documents"), ("Library", "CloudStorage"))


def _q(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")


def note_content(d, url=""):
    fm = ["---", f'titre: "{_q(d.get("titre", ""))}"', f'langue: "{_q(d.get("langue_source", ""))}"']
    if url:
        fm.append(f'source: "{_q(url)}"')
    fm += [MARQUEUR, "---", ""]
    return "\n".join(fm) + to_markdown(d)


def _racines() -> list[Path]:
    """Racines autorisées : OBSIDIAN_ALLOWED_ROOTS (séparées par os.pathsep), sinon le dossier personnel."""
    brut = os.getenv("OBSIDIAN_ALLOWED_ROOTS", "")
    racines = [Path(r.strip()).expanduser().resolve() for r in brut.split(os.pathsep) if r.strip()]
    return racines or [Path.home().resolve()]


def verifier_vault(vault_dir) -> Path:
    """Renvoie le dossier résolu ou lève ValueError si l'emplacement n'est pas sûr.

    Politique : chemin absolu (ou ~), situé (après résolution des liens symboliques) dans
    une racine autorisée (défaut : dossier personnel), sans composant caché (.ssh, .claude,
    .config, .git...) et hors de ~/Library sauf dossiers de synchronisation.
    """
    if not vault_dir or not str(vault_dir).strip():
        raise ValueError("Dossier du vault manquant")
    brut = Path(str(vault_dir).strip()).expanduser()
    if not brut.is_absolute():
        raise ValueError("Le dossier du vault doit être un chemin absolu (ou commencer par ~).")
    base = brut.resolve()
    for racine in _racines():
        if base == racine or racine in base.parents:
            rel = base.relative_to(racine).parts
            break
    else:
        raise ValueError("Dossier du vault refusé : il doit se trouver dans votre dossier personnel.")
    if any(p.startswith(".") for p in rel):
        raise ValueError("Dossier du vault refusé : dossier caché ou système.")
    if rel[:1] == ("Library",) and tuple(rel[:2]) not in LIBRARY_OK:
        raise ValueError("Dossier du vault refusé : emplacement système (~/Library).")
    return base


def _est_note_synopsis(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            tete = f.read(4096)
    except OSError:
        return False
    if not tete.startswith("---\n"):
        return False
    fin = tete.find("\n---", 4)
    return fin != -1 and MARQUEUR in tete[4:fin].split("\n")


def save_note(d, vault_dir, url=""):
    base = verifier_vault(vault_dir)
    name = re.sub(r'[\\/:*?"<>|]', "", str(d.get("titre", "Synopsis") or "")).strip().strip(".").strip() or "Synopsis"
    path = (base / f"{name}.md").resolve()
    if path.parent != base:
        raise ValueError("Nom de note invalide")
    if os.path.lexists(path) and not (path.is_file() and _est_note_synopsis(path)):
        raise ValueError("Une note du même nom, non créée par Synopsis, existe déjà : elle n'est pas écrasée.")
    base.mkdir(parents=True, exist_ok=True)
    path.write_text(note_content(d, url), encoding="utf-8")
    return str(path)
