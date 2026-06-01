"""
PRISM - Generate pseudo-labels for all videos using GPT.
Creates a CSV with video_id → generated title pairs.

Usage:
    python generate_pseudo_labels.py --api_key YOUR_OPENAI_KEY
"""
import os
import sys
import json
import csv
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLIP_DIR, OCR_DIR, ASR_DIR, PROCESSED,
)
from data.data_utils import read_text_file, get_video_ids_numeric
from utils.gpt_generator import get_openai_client, generate_title_gpt


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels with GPT")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--preview", type=int, default=0,
                        help="Preview N videos before full run (0 = run all)")
    args = parser.parse_args()

    client = get_openai_client(args.api_key)

    # Get video IDs
    video_ids = get_video_ids_numeric()
    print(f"📁 Total videos found: {len(video_ids)}")

    os.makedirs(PROCESSED, exist_ok=True)

    if args.preview > 0:
        print(f"\n🔍 Previewing titles for first {args.preview} videos:\n")
        for vid in video_ids[:args.preview]:
            ocr_text = read_text_file(str(OCR_DIR / f"{vid}.txt"))
            asr_text = read_text_file(str(ASR_DIR / f"{vid}.txt"))
            title = generate_title_gpt(client, ocr_text, asr_text, model=args.model)
            print(f"  {vid}  →  {title}")
        print("\n✅ Preview done.")
        return

    # Full run
    json_path = str(PROCESSED / "pseudo_titles.json")
    csv_path = str(PROCESSED / "pseudo_titles.csv")

    results = {}

    print("\n🚀 Generating titles for all videos...")
    for vid in tqdm(video_ids):
        ocr_text = read_text_file(str(OCR_DIR / f"{vid}.txt"))
        asr_text = read_text_file(str(ASR_DIR / f"{vid}.txt"))

        try:
            title = generate_title_gpt(client, ocr_text, asr_text, model=args.model)
        except Exception as e:
            print(f"⚠️  Error on {vid}: {e}")
            title = ""

        results[vid] = title

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "title"])
        for vid, title in results.items():
            writer.writerow([vid, title])

    print(f"\n💾 Saved {len(results)} titles")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
