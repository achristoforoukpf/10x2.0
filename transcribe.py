#!/usr/bin/env python3
"""Transcribe an .m4a audio file (up to 60 minutes) to text using OpenAI Whisper.

Files larger than the 25 MB API limit (or longer than --chunk-seconds) are split
with ffmpeg into chunks, transcribed individually, and concatenated.

Usage:
    export OPENAI_API_KEY=sk-...
    python transcribe.py input.m4a -o transcript.txt

Requirements:
    - Python 3.9+
    - ffmpeg + ffprobe on PATH
    - `openai` package (pip install -r requirements.txt)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


WHISPER_API_LIMIT_BYTES = 25 * 1024 * 1024
DEFAULT_CHUNK_SECONDS = 600
MAX_DURATION_SECONDS = 60 * 60


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"error: '{name}' not found on PATH; install ffmpeg first")


def get_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(out.strip())


def split_audio(src: Path, out_dir: Path, chunk_seconds: int) -> list[Path]:
    pattern = out_dir / "chunk_%03d.m4a"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "copy", str(pattern),
    ], check=True)
    return sorted(out_dir.glob("chunk_*.m4a"))


def transcribe_file(client, path: Path, model: str, language: str | None) -> str:
    with path.open("rb") as f:
        kwargs = {"model": model, "file": f}
        if language:
            kwargs["language"] = language
        resp = client.audio.transcriptions.create(**kwargs)
    return resp.text


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to .m4a file")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output text file (default: stdout)")
    parser.add_argument("--language",
                        help="ISO-639-1 language code (e.g. en); auto-detected if omitted")
    parser.add_argument("--model", default="whisper-1",
                        help="OpenAI transcription model (default: whisper-1)")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS,
                        help=f"Max seconds per chunk (default {DEFAULT_CHUNK_SECONDS})")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"error: {args.input} not found")

    require_binary("ffmpeg")
    require_binary("ffprobe")

    duration = get_duration(args.input)
    if duration > MAX_DURATION_SECONDS:
        sys.exit(
            f"error: audio is {duration/60:.1f} min; max supported is "
            f"{MAX_DURATION_SECONDS // 60} min"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("error: set OPENAI_API_KEY in your environment")

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("error: the 'openai' package is missing; run "
                 "`pip install -r requirements.txt`")

    client = OpenAI(api_key=api_key)
    pieces: list[str] = []

    size = args.input.stat().st_size
    if size <= WHISPER_API_LIMIT_BYTES and duration <= args.chunk_seconds:
        print(f"transcribing {args.input.name} ({duration/60:.1f} min)...",
              file=sys.stderr)
        pieces.append(transcribe_file(client, args.input, args.model, args.language))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            print(
                f"splitting {duration/60:.1f} min audio into "
                f"~{args.chunk_seconds // 60}-min chunks...",
                file=sys.stderr,
            )
            chunks = split_audio(args.input, tmp_dir, args.chunk_seconds)
            for i, chunk in enumerate(chunks, 1):
                print(f"transcribing chunk {i}/{len(chunks)}...", file=sys.stderr)
                pieces.append(transcribe_file(client, chunk, args.model, args.language))

    text = "\n\n".join(p.strip() for p in pieces if p.strip())

    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output} ({len(text)} chars)", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
