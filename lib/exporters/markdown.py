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
