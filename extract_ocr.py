"""
PRISM — batch OCR extraction for all raw videos.

Writes one UTF-8 text file per video under ocr_text/{video_id}.txt (same layout
as the rest of the pipeline).

Default: RapidOCR (ONNX) at 0.5 FPS. Use --engine paddle if you install PaddleOCR.

Usage:
    python extract_ocr.py
    python extract_ocr.py --preview 2 --fps 0.5
    python extract_ocr.py --engine paddle --lang en
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VIDEO_DIR, OCR_DIR, OCR_EXTRACT_FPS, OCR_ENGINE_DEFAULT
from pathlib import Path


def _sort_videos(video_dir: Path):
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in extensions]

    def sort_key(f):
        name = f.stem
        parts = name.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            return (0, int(parts[1]), name)
        return (1, 0, name)

    videos.sort(key=sort_key)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Extract on-screen text from videos (≥0.5 FPS sampling by default)"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help=f"Frames per second to OCR (default from config OCR_EXTRACT_FPS={OCR_EXTRACT_FPS})",
    )
    parser.add_argument(
        "--engine", type=str, default=None,
        choices=["rapid", "paddle"],
        help=f"OCR backend: rapid=RapidOCR+ONNX (default), paddle=PaddleOCR (default: {OCR_ENGINE_DEFAULT})",
    )
    parser.add_argument(
        "--lang", type=str, default="en",
        help="PaddleOCR lang code (only for --engine paddle)",
    )
    parser.add_argument(
        "--preview", type=int, default=0,
        help="Only process first N videos (0 = all)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing ocr_text/*.txt files",
    )
    args = parser.parse_args()

    fps = args.fps if args.fps is not None else OCR_EXTRACT_FPS
    if fps < 0.5:
        print(f"⚠️  fps={fps} is below 0.5; raising to 0.5 for coverage as requested.")
        fps = max(fps, 0.5)

    engine = (args.engine or OCR_ENGINE_DEFAULT).lower().strip()
    if engine not in ("rapid", "paddle"):
        print(f"⚠️  Unknown engine {engine!r} (set OCR_ENGINE to rapid|paddle); using rapid.")
        engine = "rapid"

    if not VIDEO_DIR.is_dir():
        print(f"❌ VIDEO_DIR not found: {VIDEO_DIR}")
        sys.exit(1)

    videos = _sort_videos(VIDEO_DIR)
    if args.preview > 0:
        videos = videos[: args.preview]

    if not videos:
        print("❌ No videos found.")
        return

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    from data.ocr_video import extract_ocr_from_video
    from tqdm import tqdm

    # One engine for whole run (avoids reload per file).
    ocr_handle = None
    if engine == "rapid":
        from data.ocr_video import _make_rapid_engine

        print("🔤 Loading RapidOCR (ONNX)…")
        ocr_handle = _make_rapid_engine()
    else:
        from data.ocr_video import _make_paddle_engine

        print("🔤 Loading PaddleOCR…")
        ocr_handle = _make_paddle_engine(lang=args.lang)

    print(f"   sample_fps={fps}, engine={engine}, out={OCR_DIR}")
    print()

    for vf in tqdm(videos, desc="OCR"):
        vid = vf.stem
        out_path = OCR_DIR / f"{vid}.txt"
        if out_path.exists() and not args.overwrite:
            tqdm.write(f"  skip {vid} (exists, use --overwrite)")
            continue
        try:
            text = extract_ocr_from_video(
                vf,
                sample_fps=fps,
                engine=engine,
                lang=args.lang,
                ocr_handle=ocr_handle,
            )
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            tqdm.write(f"  ⚠️  {vid}: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
