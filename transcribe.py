"""
Transcribe audio with OpenAI Whisper.

Setup:
    pip install -U openai-whisper
    # ffmpeg is required:
    #   Windows: winget install ffmpeg   |  macOS: brew install ffmpeg
    #   Ubuntu:  sudo apt install ffmpeg

Usage:
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --model medium --language hi
    python transcribe.py audio.mp3 --translate          # translate to English
    python transcribe.py audio.mp3 --formats txt srt vtt
"""

import argparse
from pathlib import Path

import whisper


def fmt_time(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def write_txt(result, path: Path):
    path.write_text(result["text"].strip() + "\n", encoding="utf-8")


def write_srt(result, path: Path):
    lines = []
    for i, seg in enumerate(result["segments"], 1):
        lines += [
            str(i),
            f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}",
            seg["text"].strip(),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(result, path: Path):
    lines = ["WEBVTT", ""]
    for seg in result["segments"]:
        lines += [
            f"{fmt_time(seg['start'], '.')} --> {fmt_time(seg['end'], '.')}",
            seg["text"].strip(),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


WRITERS = {"txt": write_txt, "srt": write_srt, "vtt": write_vtt}


def main():
    p = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    p.add_argument("audio", help="Path to audio/video file (mp3, wav, m4a, mp4, ...)")
    p.add_argument("--model", default="small",
                   choices=["tiny", "base", "small", "medium", "large", "turbo"],
                   help="Bigger = more accurate but slower (default: small)")
    p.add_argument("--language", default=None,
                   help="Language code, e.g. en, hi. Omit to auto-detect")
    p.add_argument("--translate", action="store_true",
                   help="Translate speech to English instead of transcribing")
    p.add_argument("--formats", nargs="+", default=["txt"], choices=WRITERS,
                   help="Output formats (default: txt)")
    p.add_argument("--outdir", default=".", help="Output folder")
    args = p.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        raise SystemExit(f"File not found: {audio}")

    print(f"Loading model '{args.model}'...")
    model = whisper.load_model(args.model)

    print(f"Transcribing {audio.name}...")
    result = model.transcribe(
        str(audio),
        language=args.language,
        task="translate" if args.translate else "transcribe",
        fp16=False if model.device.type == "cpu" else True,
        verbose=False,
    )
    print(f"Detected language: {result.get('language')}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        out = outdir / f"{audio.stem}.{fmt}"
        WRITERS[fmt](result, out)
        print(f"Saved {out}")

    print("\n--- Transcript ---\n" + result["text"].strip())


if __name__ == "__main__":
    main()