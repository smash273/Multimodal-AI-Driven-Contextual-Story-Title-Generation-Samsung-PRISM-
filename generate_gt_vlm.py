"""
PRISM - Generate ground truth titles for all videos using Qwen2-VL.

This script processes all raw videos from Google Drive and generates
descriptive titles using the Qwen2-VL vision-language model.
The generated titles are saved as CSV to be used for training.

Usage (on Colab):
    python generate_gt_vlm.py
    python generate_gt_vlm.py --model Qwen/Qwen2-VL-7B-Instruct  # larger model
    python generate_gt_vlm.py --preview 5   # preview first 5 videos
    python generate_gt_vlm.py --resume      # resume interrupted run
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    VIDEO_DIR, PROCESSED, VLM_GT_CSV,
    VLM_MODEL_NAME,
)


def get_video_files():
    """Get all video files sorted by numeric ID."""
    video_dir = VIDEO_DIR
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        print("   Make sure Google Drive is mounted and path is correct.")
        print(f"   Expected path: {video_dir}")
        sys.exit(1)

    # Find all video files
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [
        f for f in video_dir.iterdir()
        if f.suffix.lower() in extensions
    ]

    # Sort by numeric ID first (vs_3, vs_4, ...), then by name.
    # Use a tuple so key types are always comparable.
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
        description="Generate ground truth titles using Qwen2-VL"
    )
    parser.add_argument(
        "--model", type=str, default=VLM_MODEL_NAME,
        help=f"VLM model name (default: {VLM_MODEL_NAME})"
    )
    parser.add_argument(
        "--preview", type=int, default=0,
        help="Preview N videos before full run (0 = run all)"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from previously saved results (default: True)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, don't resume from previous results"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: auto from config)"
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Max number of videos to process (for testing)"
    )
    parser.add_argument(
        "--quantization", type=str, default="none",
        choices=["none", "4bit", "8bit"],
        help="Model quantization mode for GPU memory savings (default: none)"
    )
    parser.add_argument(
        "--max-pixels", "--max_pixels", type=int, default=None, dest="max_pixels",
        help="Override max video pixels sent to VLM (lower = less VRAM)"
    )
    parser.add_argument(
        "--video-fps", "--fps", type=float, default=None, dest="video_fps",
        help="Video sampling FPS for VLM (lower = fewer frames = less VRAM; default from config)"
    )
    parser.add_argument(
        "--max-new-tokens", "--max_new_tokens", type=int, default=None, dest="max_new_tokens",
        help="Max tokens to generate per title (default from config)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature for VLM generation (default from config)"
    )
    parser.add_argument(
        "--no-aux-text", action="store_true",
        help="Do not append OCR/ASR from ocr_text/ and audio_text/ (video-only prompt)"
    )
    args = parser.parse_args()

    resume = not args.no_resume
    output_path = args.output or str(VLM_GT_CSV)

    print("=" * 60)
    print("PRISM - VLM Ground Truth Generator")
    print("=" * 60)

    # Get video files
    video_files = get_video_files()
    print(f"📁 Found {len(video_files)} videos in {VIDEO_DIR}")

    if args.max_videos:
        video_files = video_files[:args.max_videos]
        print(f"   (Processing only first {args.max_videos})")

    if len(video_files) == 0:
        print("❌ No video files found!")
        return

    # Show some samples
    print(f"   Sample: {[f.stem for f in video_files[:5]]}")

    # Initialize labeler
    from data.vlm_labeler import VLMLabeler

    labeler = VLMLabeler(
        model_name=args.model,
        quantization=args.quantization,
        max_pixels=args.max_pixels,
        video_fps=args.video_fps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_aux_text=not args.no_aux_text,
    )
    labeler.load_model()
    print(f"   VLM video settings: max_pixels={labeler.max_pixels}, fps={labeler.video_fps}, "
          f"max_new_tokens={labeler.max_new_tokens}, temperature={labeler.temperature}, "
          f"ocr_asr_hints={'yes' if labeler.use_aux_text else 'no'}")

    video_ids = [f.stem for f in video_files]
    video_paths = [str(f) for f in video_files]

    if args.preview > 0:
        # Preview mode
        print(f"\n🔍 Preview mode: generating titles for {args.preview} videos\n")

        for vid, path in zip(video_ids[:args.preview], video_paths[:args.preview]):
            try:
                title = labeler.generate_title(path, video_id=vid)
                print(f"  {vid}  →  {title}")
            except Exception as e:
                print(f"  {vid}  →  ⚠️ Error: {e}")

        print("\n✅ Preview done.")
        labeler.unload_model()
        return

    # Full run
    os.makedirs(PROCESSED, exist_ok=True)

    results = labeler.generate_titles_batch(
        video_paths=video_paths,
        video_ids=video_ids,
        save_path=output_path,
        resume=resume,
    )

    # Also save as JSON for convenience
    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    non_empty = sum(1 for t in results.values() if t.strip())
    print(f"\n{'=' * 60}")
    print(f"📊 Summary:")
    print(f"   Total videos:  {len(results)}")
    print(f"   Titles generated: {non_empty}")
    print(f"   Empty/failed:  {len(results) - non_empty}")
    print(f"\n💾 Saved to:")
    print(f"   CSV:  {output_path}")
    print(f"   JSON: {json_path}")
    print(f"{'=' * 60}")

    # Cleanup
    labeler.unload_model()
    print("✅ Done!")


if __name__ == "__main__":
    main()
