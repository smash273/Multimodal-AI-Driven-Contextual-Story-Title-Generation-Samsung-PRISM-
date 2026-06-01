"""
PRISM — Standalone Ground Truth Title Generator
Hardware target: RTX A4000 (16 GB VRAM) + 32 GB RAM

DROP THIS FILE on the A4000 machine and run:

    # Recommended — 8-bit quant keeps weights at ~8 GB, leaving 8 GB free for frames
    python run_gt_a4000.py --videos /path/to/videos/raw --out vlm_titles.csv

    # No quantization (max quality) — tight on 16 GB; add --fps 0.75 if OOM
    python run_gt_a4000.py --videos /path/to/videos/raw --out vlm_titles.csv --quant none

    # Preview first 5 videos before committing to full run
    python run_gt_a4000.py --videos /path/to/videos/raw --out vlm_titles.csv --preview 5

    # Resume an interrupted run automatically (default)
    python run_gt_a4000.py --videos /path/to/videos/raw --out vlm_titles.csv --resume

Dependencies (pip install once):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.47 qwen-vl-utils bitsandbytes tqdm
    pip install flash-attn --no-build-isolation   # optional but recommended
"""

"""
Install dependencies (once)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.47 qwen-vl-utils bitsandbytes tqdm
pip install flash-attn --no-build-isolation   # optional, but recommended for A4000


Run i
# Recommended — 8-bit quant, safe on 16 GB
python run_gt_a4000.py --videos /path/to/videos/raw --out vlm_titles.csv
# Max quality — no quantization (might OOM on very long clips, will auto-fallback)
python run_gt_a4000.py --videos /path/to/videos/raw --out vlm_titles.csv --quant none
# Test first 5 before committing to 4500
python run_gt_a4000.py --videos /path/to/videos/raw --out vlm_titles.csv --preview 5
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS — edit these to match your machine
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Resolution passed to the VLM per frame.
# 720×560 = 403 200 px  →  rich detail, needs ~8 GB free VRAM for frames.
# Drop to 360×280 = 100 800 if you hit OOM with --quant none.
MAX_PIXELS = 720 * 560       # 403_200

# Temporal sampling. 1.0 = one frame per second (good for news/action).
# Raise to 1.5 for fast-cut content; lower to 0.5 to save VRAM.
VIDEO_FPS = 1.0

# How many tokens the model may generate for the title.
MAX_NEW_TOKENS = 60

# How much OCR / ASR text (characters) to include as context hints.
# More = better for text-heavy videos; less = faster.
MAX_OCR_CHARS = 1200
MAX_ASR_CHARS = 1500

# Folder names for OCR / ASR sibling text files (relative to video directory).
# If these folders don't exist the model runs video-only — that's fine.
OCR_SUBDIR  = "ocr_text"    # expects <ocr_subdir>/<video_id>.txt
ASR_SUBDIR  = "audio_text"  # expects <asr_subdir>/<video_id>.txt

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

PROMPT = (
    "Watch the video and write ONE short factual title (4–12 words).\n\n"
    "Rules:\n"
    "- Base the title ONLY on what you actually see and hear. "
    "Do NOT invent animals, places, or actions that are not clearly in the video.\n"
    "- If it looks like a news segment, studio interview, or CCTV/security footage, "
    "say that and describe the main action (e.g. anchor reporting, car leaving a driveway).\n"
    "- If OCR/ASR hints are provided below, use them to disambiguate; "
    "if they conflict with the video, trust the video.\n"
    "- No quotation marks around the title. No hashtags or emojis.\n"
    "- Avoid generic clickbait words (amazing, viral, epic, you won't believe).\n\n"
    "Output ONLY the title, no other text."
)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNALS — you should not need to change anything below this line
# ─────────────────────────────────────────────────────────────────────────────

# Qwen-VL hard minimum enforced by qwen-vl-utils
_QWEN_VIDEO_MIN_PIXELS = 128 * 28 * 28   # 100_352


def _detect_attention():
    """Use flash_attention_2 if available (big VRAM saving for video KV-cache)."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


def _resolve_pixels(requested: int):
    safe = max(int(requested), _QWEN_VIDEO_MIN_PIXELS)
    return _QWEN_VIDEO_MIN_PIXELS, safe


def _build_bnb_config(quant: str):
    import torch
    from transformers import BitsAndBytesConfig
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(model_name: str, quant: str):
    """Load Qwen2.5-VL and return (model, processor)."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print("=" * 60)
    print(f"  Model : {model_name}")
    print(f"  Quant : {quant}")

    # dtype
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    print(f"  dtype : {dtype}")

    # attention
    attn = "eager" if quant != "none" else _detect_attention()
    print(f"  Attn  : {attn}")
    print("=" * 60)

    bnb = _build_bnb_config(quant)
    load_kw = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "attn_implementation": attn,
    }
    if bnb is not None:
        load_kw["quantization_config"] = bnb
        load_kw["attn_implementation"] = "eager"  # bnb + flash-attn conflict

    print("⏳ Loading model weights … (first run downloads ~15 GB, cached after)")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kw)
    model.eval()

    min_px, max_px = _resolve_pixels(MAX_PIXELS)
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_px,
        max_pixels=max_px,
        use_fast=False,
    )
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token_id = getattr(tok, "eos_token_id", None)

    print("✅ Model ready\n")
    return model, processor


# ─── text helpers ────────────────────────────────────────────────────────────

def _read_txt(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return text[:max_chars] if max_chars > 0 else text
    except Exception:
        return ""


def _compose_prompt(video_id: str, ocr_dir: Path, asr_dir: Path) -> str:
    ocr = _read_txt(ocr_dir / f"{video_id}.txt", MAX_OCR_CHARS)
    asr = _read_txt(asr_dir / f"{video_id}.txt", MAX_ASR_CHARS)

    if not ocr and not asr:
        return PROMPT

    parts = [
        PROMPT,
        "",
        "Automated hints from the same video (OCR/ASR; may be partial or wrong):",
        "",
    ]
    if ocr:
        parts.append(f"On-screen text (OCR):\n{ocr}")
    if asr:
        if ocr:
            parts.append("")
        parts.append(f"Spoken / transcribed audio (ASR):\n{asr}")
    return "\n".join(parts)


def _is_degenerate(title: str) -> bool:
    if not title or not title.strip():
        return True
    t = title.strip()
    if re.fullmatch(r"[\W_]+", t):
        return True
    alpha = sum(ch.isalpha() for ch in t)
    if len(t) >= 12 and alpha < 3:
        return True
    if len(set(t)) <= 2 and len(t) >= 8:
        return True
    if len(t) >= 10:
        alnum = sum(1 for ch in t if ch.isalnum())
        if alnum < max(3, len(t) // 4):
            return True
    return False


def _clean(title: str) -> str:
    title = title.strip().strip('"').strip("'").strip()
    if "\n" in title:
        title = title.split("\n")[0].strip()
    for prefix in ("Title:", "title:", "Video Title:", "video title:",
                   "The title is:", "Here is the title:"):
        if title.lower().startswith(prefix.lower()):
            title = title[len(prefix):].strip()
    return title


# ─── inference ───────────────────────────────────────────────────────────────

def _run_once(model, processor, video_path: str,
              user_text: str, max_pixels: int, fps: float) -> str:
    """Single generate call — raises RuntimeError on OOM."""
    import torch
    from qwen_vl_utils import process_vision_info

    min_px = _QWEN_VIDEO_MIN_PIXELS
    safe_px = max(int(max_pixels), min_px)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"file://{video_path}",
                "min_pixels": min_px,
                "max_pixels": safe_px,
                "fps": fps,
            },
            {"type": "text", "text": user_text},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    tok = getattr(processor, "tokenizer", None)
    pad_id = getattr(tok, "pad_token_id", None) if tok else None
    eos_id = getattr(tok, "eos_token_id", None) if tok else None
    cfg = model.config
    pad_id = pad_id or getattr(cfg, "pad_token_id", None) or eos_id
    eos_id = eos_id or getattr(cfg, "eos_token_id", None)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=5,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
    title = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    del inputs, out_ids, gen_ids
    gc.collect()
    torch.cuda.empty_cache()
    return title


def generate_title(model, processor, video_path: str,
                   video_id: str, ocr_dir: Path, asr_dir: Path) -> str:
    """Generate a title with OOM fallback (reduce pixels → fps)."""
    user_text = _compose_prompt(video_id, ocr_dir, asr_dir)

    base_px  = int(MAX_PIXELS)
    base_fps = float(VIDEO_FPS)
    floor    = _QWEN_VIDEO_MIN_PIXELS

    attempts = [
        (base_px,                                        base_fps),
        (max(min(base_px, 360 * 280), floor),            max(min(base_fps, 0.5), 0.15)),
        (floor,                                          max(min(base_fps, 0.35), 0.15)),
        (floor,                                          0.15),
    ]
    seen, unique = set(), []
    for p in attempts:
        if p not in seen:
            seen.add(p); unique.append(p)

    last_err = None
    title = ""
    for px, fps in unique:
        try:
            title = _run_once(model, processor, video_path, user_text, px, fps)
            last_err = None
            if not _is_degenerate(title):
                break
        except RuntimeError as e:
            err = str(e).lower()
            last_err = e
            if "out of memory" in err or "cuda" in err or "assert" in err:
                gc.collect()
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise

    if last_err is not None:
        raise last_err

    if _is_degenerate(title):
        return "untitled video"
    return _clean(title) if title else "untitled video"


# ─── batch / IO ──────────────────────────────────────────────────────────────

def _save_csv(results: dict, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "title"])
        for vid, title in sorted(results.items()):
            w.writerow([vid, title])


def _load_resume(path: str) -> dict:
    results = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results[row["video_id"]] = row["title"]
        print(f"📂 Resumed {len(results)} previously generated titles from {path}")
    return results


def _sort_key(f: Path):
    parts = f.stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return (0, int(parts[-1]), f.stem)
    return (1, 0, f.stem)


def get_videos(video_dir: Path):
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    videos.sort(key=_sort_key)
    return videos


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM — Ground Truth Generator (RTX A4000 standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_gt_a4000.py --videos /data/videos/raw --out vlm_titles.csv
  python run_gt_a4000.py --videos /data/videos/raw --out vlm_titles.csv --quant none
  python run_gt_a4000.py --videos /data/videos/raw --out vlm_titles.csv --preview 5
        """,
    )
    parser.add_argument("--videos",   required=True,  help="Directory containing raw video files")
    parser.add_argument("--out",      required=True,  help="Output CSV path (video_id, title)")
    parser.add_argument("--quant",    default="8bit", choices=["none", "4bit", "8bit"],
                        help="Quantization: 8bit (default, safe), none (max quality), 4bit (min VRAM)")
    parser.add_argument("--model",    default=MODEL_NAME, help="HuggingFace model ID")
    parser.add_argument("--fps",      type=float, default=VIDEO_FPS,
                        help=f"Video sampling FPS (default: {VIDEO_FPS})")
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS,
                        help=f"Max pixels per frame (default: {MAX_PIXELS})")
    parser.add_argument("--preview",  type=int, default=0,
                        help="Preview N videos then exit (0 = full run)")
    parser.add_argument("--resume",   action="store_true", default=True,
                        help="Resume from existing output CSV (default: True)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing output")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Process only first N videos (for testing)")
    parser.add_argument("--ocr-dir",  default=None,
                        help=f"OCR text folder (default: <videos>/../{OCR_SUBDIR})")
    parser.add_argument("--asr-dir",  default=None,
                        help=f"ASR text folder (default: <videos>/../{ASR_SUBDIR})")
    parser.add_argument("--save-json", action="store_true",
                        help="Also save a JSON copy of the results")
    args = parser.parse_args()

    # ── resolve paths ──────────────────────────────────────────────────────
    video_dir = Path(args.videos).resolve()
    if not video_dir.is_dir():
        print(f"❌ Video directory not found: {video_dir}")
        sys.exit(1)

    base = video_dir.parent
    ocr_dir = Path(args.ocr_dir).resolve() if args.ocr_dir else base / OCR_SUBDIR
    asr_dir = Path(args.asr_dir).resolve() if args.asr_dir else base / ASR_SUBDIR

    out_path = str(Path(args.out).resolve())
    resume   = not args.no_resume

    # ── print config ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PRISM — Ground Truth Title Generator")
    print("  Hardware: RTX A4000  |  16 GB VRAM  |  32 GB RAM")
    print("=" * 60)
    print(f"  Video dir  : {video_dir}")
    print(f"  OCR dir    : {ocr_dir} {'✅' if ocr_dir.exists() else '⚠️  (not found — video-only mode)'}")
    print(f"  ASR dir    : {asr_dir} {'✅' if asr_dir.exists() else '⚠️  (not found — video-only mode)'}")
    print(f"  Output CSV : {out_path}")
    print(f"  Model      : {args.model}")
    print(f"  Quant      : {args.quant}")
    print(f"  FPS        : {args.fps}")
    print(f"  Max pixels : {args.max_pixels}")
    print("=" * 60 + "\n")

    # ── discover videos ───────────────────────────────────────────────────
    videos = get_videos(video_dir)
    if not videos:
        print("❌ No video files found!")
        sys.exit(1)

    if args.max_videos:
        videos = videos[: args.max_videos]

    print(f"📁 Found {len(videos)} videos")
    print(f"   Sample : {[v.stem for v in videos[:5]]}\n")

    # ── load model ────────────────────────────────────────────────────────
    model, processor = load_model(args.model, args.quant)

    # ── preview mode ──────────────────────────────────────────────────────
    if args.preview > 0:
        print(f"🔍 Preview mode — generating {args.preview} titles\n")
        for v in videos[: args.preview]:
            try:
                t = generate_title(model, processor, str(v), v.stem, ocr_dir, asr_dir)
                print(f"  {v.stem:<20}  →  {t}")
            except Exception as e:
                print(f"  {v.stem:<20}  →  ⚠️  {e}")
        print("\n✅ Preview done.")
        return

    # ── full run ──────────────────────────────────────────────────────────
    from tqdm import tqdm

    results = _load_resume(out_path) if resume else {}

    remaining = [(v.stem, str(v)) for v in videos if v.stem not in results]
    if not remaining:
        print("✅ All titles already generated!")
        return

    print(f"🎬 Generating titles for {len(remaining)} videos …\n")
    t0 = time.time()

    for i, (vid, path) in enumerate(tqdm(remaining, desc="Titles")):
        try:
            title = generate_title(model, processor, path, vid, ocr_dir, asr_dir)
            results[vid] = title

            if (i + 1) % 5 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                eta_s   = (len(remaining) - i - 1) / rate if rate > 0 else 0
                eta_str = f"{eta_s / 3600:.1f}h" if eta_s > 3600 else f"{eta_s / 60:.0f}m"
                tqdm.write(f"  [{vid}]  →  {title}   (ETA: {eta_str})")

        except Exception as e:
            tqdm.write(f"  ⚠️  Error on {vid}: {e}")
            results[vid] = ""

        # Save checkpoint every 10 videos
        if (i + 1) % 10 == 0:
            _save_csv(results, out_path)

        # Periodic VRAM cleanup
        if (i + 1) % 5 == 0:
            gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

    # ── final save ────────────────────────────────────────────────────────
    _save_csv(results, out_path)
    print(f"\n💾 CSV saved → {out_path}")

    if args.save_json:
        json_path = out_path.replace(".csv", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON saved → {json_path}")

    # ── summary ───────────────────────────────────────────────────────────
    non_empty = sum(1 for t in results.values() if t.strip())
    elapsed   = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Total videos   : {len(results)}")
    print(f"  Titles OK      : {non_empty}")
    print(f"  Failed / empty : {len(results) - non_empty}")
    print(f"  Time elapsed   : {elapsed / 3600:.2f} h")
    print(f"{'=' * 60}")

    # cleanup
    del model, processor
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    print("✅ Done!")


if __name__ == "__main__":
    main()
