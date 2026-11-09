"""Video Mounter — Clip extraction, highlight reel, vertical export, TTS sync.

Adapted for Synopsis — uses env vars instead of config.py.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CLIP_OUTPUT_DIR = os.path.expanduser(os.getenv("CLIP_OUTPUT_DIR", "~/Documents/Synopsis/clips"))
CLIP_DEFAULT_DURATION = int(os.getenv("CLIP_DEFAULT_DURATION", "45"))
CLIP_PADDING_BEFORE = int(os.getenv("CLIP_PADDING_BEFORE", "5"))
CLIP_PADDING_AFTER = int(os.getenv("CLIP_PADDING_AFTER", "5"))
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() in ("true", "1", "yes")


def _find_ffmpeg():
    for c in ["ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if shutil.which(c):
            return c
    return None


def _find_yt_dlp():
    for c in ["yt-dlp", "/usr/local/bin/yt-dlp", "/opt/homebrew/bin/yt-dlp"]:
        if shutil.which(c):
            return c
    return None


def _subprocess_env():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def ts_to_seconds(ts: str) -> float:
    parts = list(map(int, ts.strip().split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def download_video(url: str, output_dir: str, max_height: int = 1080) -> str:
    yt_dlp_bin = _find_yt_dlp()
    if not yt_dlp_bin:
        raise ValueError("yt-dlp non installé. pip install yt-dlp")
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title).100s [%(id)s].%(ext)s")
    format_spec = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"
    cmd = [yt_dlp_bin, "-f", format_spec, "--merge-output-format", "mp4", "--no-playlist", "--no-mtime", "-o", output_template, "--", url]
    logger.info("Downloading video: %s", url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp: {result.stderr[-800:]}")
    mp4_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".mp4")], key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
    if not mp4_files:
        raise FileNotFoundError(f"Aucun fichier mp4 dans {output_dir}")
    return os.path.join(output_dir, mp4_files[0])


def extract_clip(video_path: str, output_path: str, start_seconds: float, duration: float, subtitle_text: Optional[str] = None) -> str:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise ValueError("ffmpeg non trouvé")
    cmd = [ffmpeg, "-y", "-ss", str(start_seconds), "-t", str(duration), "-i", video_path]
    if subtitle_text:
        safe_text = subtitle_text.replace("'", "'\\\\''").replace(":", "\\:")
        cmd.extend(["-vf", f"drawtext=text='{safe_text}':fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6:x=(w-text_w)/2:y=h-th-20"])
    cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", output_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip: {result.stderr[-500:]}")
    return output_path


def extract_clips_from_chapters(video_path: str, chapters: list[dict], output_dir: str, clip_duration: float = 45, padding_before: float = 5, padding_after: float = 5, burn_subtitles: bool = True) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    clips = []
    for i, ch in enumerate(chapters):
        ts = ch.get("ts_seconds", 0)
        subject = ch.get("subject", f"Chapter {i+1}")
        start = max(0, ts - padding_before)
        dur = clip_duration + padding_before + padding_after
        out = os.path.join(output_dir, f"clip_{i+1:03d}_{ts:06d}.mp4")
        subtitle = subject if burn_subtitles else None
        try:
            extract_clip(video_path, out, start, dur, subtitle_text=subtitle)
            clips.append({"index": i + 1, "path": out, "start_seconds": start, "duration": dur, "chapter": ch})
        except Exception as e:
            logger.warning("Clip %d failed: %s", i + 1, e)
    return clips


def build_highlight_reel(clips: list[dict], output_path: str, title_card_text: Optional[str] = None, tts_audio_path: Optional[str] = None) -> str:
    if not clips:
        raise ValueError("Aucun clip")
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise ValueError("ffmpeg non trouvé")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="reel_")
    try:
        concat_file = os.path.join(tmp, "concat.txt")
        idx, lines, inputs = 0, [], []
        if title_card_text:
            card = _title_card(title_card_text, 3.0, tmp, ffmpeg)
            if card:
                inputs.extend(["-i", card])
                lines.append(f"[{idx}:v]setpts=PTS-STARTPTS[{idx}v]")
                idx += 1
        clip_idxs = []
        for clip in clips:
            inputs.extend(["-i", clip["path"]])
            clip_idxs.append(idx)
            idx += 1
        if clip_idxs:
            labels = "".join(f"[{c}:v][{c}:a]" for c in clip_idxs)
            lines.append(f"{labels}concat=n={len(clip_idxs)}:v=1:a=1[vout][aout]")
        audio_out = "aout"
        if tts_audio_path and os.path.isfile(tts_audio_path):
            inputs.extend(["-i", tts_audio_path])
            lines.append(f"[aout][{idx}:a]amix=inputs=2:duration=first:dropout_transition=3,volume=2.0[amix]")
            audio_out = "amix"
            idx += 1
        with open(concat_file, "w") as f:
            for line in lines:
                f.write(line + ";\n")
        cmd = [ffmpeg, "-y", *inputs, "-filter_complex_script", concat_file, "-map", "[vout]", "-map", f"[{audio_out}]", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-shortest", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_subprocess_env())
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat: {result.stderr[-800:]}")
        return output_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _title_card(text: str, duration: float, tmp: str, ffmpeg: str) -> Optional[str]:
    safe = text.replace("'", "'\\\\''").replace(":", "\\:").replace("\n", " ")[:100]
    out = os.path.join(tmp, "title.mp4")
    cmd = [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=0x1a1a1a:s=1920x1080:d=0.1,r=30", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(duration), "-vf", f"drawtext=text='{safe}':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.3:boxborderw=10", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-c:a", "aac", "-b:a", "128k", "-shortest", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=_subprocess_env())
    return out if r.returncode == 0 else None


def export_vertical(video_path: str, output_path: str, mode: str = "blur") -> str:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise ValueError("ffmpeg non trouvé")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if mode == "blur":
        filt = "[0:v]split[orig][blur];[blur]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:10[bg];[orig]scale=1080:1920:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2"
    elif mode == "crop":
        filt = "crop=min(iw*1920/1080\\,iw):min(ih*1080/1920\\,ih),scale=1080:1920"
    else:
        filt = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
    cmd = [ffmpeg, "-y", "-i", video_path, "-vf", filt, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", output_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_subprocess_env())
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg vertical: {r.stderr[-500:]}")
    return output_path


def create_highlight_reel(
    video_url: str,
    summary_text: str,
    output_dir: str = None,
    video_title: str = None,
    clip_duration: float = None,
    burn_subtitles: bool = True,
    tts_narration: bool = False,
    tts_lang: str = "fr",
    export_vertical_mode: str = None,
) -> dict:
    from lib.fusion import _extract_chapters, _extract_insights

    output_dir = output_dir or CLIP_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    clip_duration = clip_duration or CLIP_DEFAULT_DURATION

    chapters = _extract_chapters(summary_text)
    if not chapters:
        insights = _extract_insights(summary_text)
        chapters = [{"ts_seconds": ts_to_seconds(i["timestamp"]), "subject": i["title"], "description": i["description"]} for i in insights]
    if not chapters:
        raise ValueError("Aucun chapitre ou insight trouvé dans le résumé")

    video_path = download_video(video_url, output_dir)

    if not video_title:
        try:
            import httpx
            video_id_match = re.search(r'(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})', video_url)
            if video_id_match:
                r = httpx.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id_match.group(1)}&format=json", timeout=10)
                video_title = r.json().get("title", "YouTube Video") if r.status_code == 200 else "YouTube Video"
        except Exception:
            video_title = "YouTube Video"

    safe_title = re.sub(r'[<>:"/\\|?*]', '_', video_title)[:60]
    clips_dir = os.path.join(output_dir, f"{safe_title}_clips")
    clips = extract_clips_from_chapters(video_path, chapters, clips_dir, clip_duration=clip_duration, padding_before=CLIP_PADDING_BEFORE, padding_after=CLIP_PADDING_AFTER, burn_subtitles=burn_subtitles)
    if not clips:
        raise RuntimeError("Aucun clip extrait")

    reel_path = os.path.join(output_dir, f"{safe_title}_reel.mp4")
    tts_path = None
    if tts_narration and TTS_ENABLED:
        narrative = "\n".join(f"{c.get('chapter', {}).get('subject', '')}. {c.get('chapter', {}).get('description', '')}" for c in clips)[:3000]
        try:
            from lib.tts_generator import generate_tts
            tts_result = generate_tts(narrative, method="gtts", lang=tts_lang)
            if tts_result.get("success"):
                tts_path = tts_result["audio_path"]
        except Exception as e:
            logger.warning("TTS failed: %s", e)

    build_highlight_reel(clips, reel_path, title_card_text=video_title, tts_audio_path=tts_path)
    if tts_path:
        try:
            os.remove(tts_path)
        except OSError:
            pass

    result = {"reel_path": reel_path, "video_title": video_title, "clip_count": len(clips)}
    if export_vertical_mode:
        vertical_path = os.path.join(output_dir, f"{safe_title}_reel_vertical.mp4")
        export_vertical(reel_path, vertical_path, mode=export_vertical_mode)
        result["vertical_path"] = vertical_path
    try:
        os.remove(video_path)
    except OSError:
        pass
    return result
