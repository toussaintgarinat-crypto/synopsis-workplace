"""Transcript Chunker — token-aware splitting for LLM processing."""

import logging
import os
import tiktoken

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "12000"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "1200"))


def estimate_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def create_text_from_transcript(transcript: list[dict]) -> str:
    return "\n".join(f"[{format_timestamp(e['start'])}] {e['text']}" for e in transcript)


def chunk_transcript(
    transcript: list[dict],
    max_tokens: int = None,
    overlap_tokens: int = None,
) -> list[dict]:
    max_tokens = max_tokens or CHUNK_SIZE_TOKENS
    overlap_tokens = overlap_tokens or CHUNK_OVERLAP_TOKENS

    if not transcript:
        return []

    full_text = create_text_from_transcript(transcript)
    total_tokens = estimate_tokens(full_text)

    if total_tokens <= max_tokens:
        return [{'text': full_text, 'start': transcript[0]['start'],
                 'end': transcript[-1]['start'] + transcript[-1].get('duration', 0), 'tokens': total_tokens}]

    chunks = []
    current_pos = 0
    while current_pos < len(transcript):
        chunk_entries = []
        chunk_tokens = 0
        for i in range(current_pos, len(transcript)):
            entry = transcript[i]
            entry_text = f"[{format_timestamp(entry['start'])}] {entry['text']}"
            entry_tokens = estimate_tokens(entry_text)
            if chunk_tokens + entry_tokens > max_tokens and chunk_entries:
                break
            chunk_entries.append(entry)
            chunk_tokens += entry_tokens

        last_entry = chunk_entries[-1]
        chunks.append({
            'text': create_text_from_transcript(chunk_entries),
            'start': chunk_entries[0]['start'],
            'end': last_entry['start'] + last_entry.get('duration', 0),
            'tokens': chunk_tokens,
        })

        overlap_count = max(1, len(chunk_entries) // 10)
        next_start = min(current_pos + len(chunk_entries) - overlap_count, len(transcript) - 1)
        if next_start <= current_pos:
            next_start = current_pos + 1
        current_pos = next_start

    return chunks
