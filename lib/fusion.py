"""Fusion — chapter/insight extraction from LLM markdown output."""

import re


def _extract_chapters(analysis: str) -> list[dict]:
    chapters = []
    if not analysis:
        return chapters
    pattern = r'##\s*📍\s*Chapitrage\s*Temporel\s*\n(.*?)(?=\n##\s|\Z)'
    match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
    if not match:
        return chapters
    section = match.group(1)
    row_pattern = r'\|\s*\[?(\d{1,3}(?::\d{2}){1,2})\]?\s*\|\s*\*?(.+?)\*?\s*\|\s*(.+?)\s*\|'
    for m in re.finditer(row_pattern, section):
        ts_raw = m.group(1)
        subject = m.group(2).strip().rstrip('*').lstrip('*')
        desc = m.group(3).strip()
        if subject.startswith(':') and desc.startswith(':'):
            continue
        parts = ts_raw.split(':')
        if len(parts) == 2:  # MM:SS
            ts_seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            ts_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            ts_seconds = 0
        chapters.append({'timestamp': ts_raw, 'ts_seconds': ts_seconds, 'subject': subject, 'description': desc})
    return chapters


def _extract_insights(analysis: str) -> list[dict]:
    insights = []
    if not analysis:
        return insights
    pattern = r'##\s*💡\s*Top\s*\d*\s*Moments\s*Forts.*?\n(.*?)(?=\n##\s|\Z)'
    match = re.search(pattern, analysis, re.DOTALL | re.IGNORECASE)
    if not match:
        return insights
    section = match.group(1)
    item_pattern = r'\d+\.\s*\*?(.+?)\*?\s*\[?(\d{1,3}(?::\d{2}){1,2})\]?\s*:\s*(.+)'
    for m in re.finditer(item_pattern, section):
        title = m.group(1).strip().rstrip('*').lstrip('*')
        ts = m.group(2)
        desc = m.group(3).strip()
        insights.append({'title': title, 'timestamp': ts, 'description': desc})
    return insights


def merge_chapter_tables(chapters_lists: list[list[dict]]) -> list[dict]:
    all_chapters = []
    for cl in chapters_lists:
        all_chapters.extend(cl)
    all_chapters.sort(key=lambda x: x.get('ts_seconds', 0))
    seen_ts = set()
    unique = []
    for c in all_chapters:
        ts = c.get('ts_seconds', 0)
        if ts not in seen_ts:
            seen_ts.add(ts)
            unique.append(c)
    return unique


def select_top_insights(insights_lists: list[list[dict]], max_insights: int = 3) -> list[dict]:
    all_insights = []
    seen = set()
    for il in insights_lists:
        for ins in il:
            key = ins.get('title', '').lower()
            if key and key not in seen:
                seen.add(key)
                all_insights.append(ins)
    return all_insights[:max_insights]


def _build_chapter_table(chapters: list[dict]) -> str:
    if not chapters:
        return ""
    lines = ["## 📍 Chapitrage Temporel", "| Time | Sujet | Description |", "| :--- | :--- | :--- |"]
    for c in chapters:
        lines.append(f"| {c.get('timestamp', '')} | *{c.get('subject', '')}* | {c.get('description', '')} |")
    return "\n".join(lines)


def _build_insights_list(insights: list[dict]) -> str:
    if not insights:
        return ""
    lines = ["## 💡 Top 3 Moments Forts (Insights)"]
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. *{ins.get('title', '')}* [{ins.get('timestamp', '')}] : {ins.get('description', '')}")
    return "\n".join(lines)
