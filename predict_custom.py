"""
PRISM - Custom video inference.
Extracts features from a new/unseen video and generates a title.

Usage:
    python predict_custom.py --video path/to/video.mp4 \
                              --checkpoint outputs/checkpoints/best_model.pt
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, EMBED_DIM
from data.data_utils import Vocabulary
from data.visual_utils import extract_visual_embedding
from data.audio_utils import extract_audio_embedding
from inference import generate_title, load_model


def build_tokens_custom(
    video_path: str,
    clip_model=None,
    clip_preprocess=None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Build multimodal tokens for a custom video (no OCR/ASR available).

    Uses CLIP for visual and YAMNet for audio.
    OCR and ASR are zero vectors since we haven't extracted them.
    """
    # Visual embedding
    if clip_model is not None:
        visual = extract_visual_embedding(
            video_path, clip_model, clip_preprocess, device
        )
    else:
        visual = np.zeros(EMBED_DIM)
        print("⚠️  No CLIP model provided, using zero visual embedding")

    # Audio embedding
    audio = extract_audio_embedding(video_path, embed_dim=EMBED_DIM)

    # No OCR/ASR for custom videos
    ocr_emb = np.zeros(EMBED_DIM)
    asr_emb = np.zeros(EMBED_DIM)

    # Stack: [visual, ocr, asr] — matching the training format (3 modalities)
    # Note: audio is not used as a separate token in the Transformer model
    # but we could substitute it for one of the zero embeddings
    tokens = np.stack([visual, ocr_emb, asr_emb])
    return torch.tensor(tokens).float()


def main():
    parser = argparse.ArgumentParser(description="PRISM Custom Video Prediction")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(OUTPUT_DIR / "checkpoints" / "best_model.pt"),
    )
    parser.add_argument("--use_clip", action="store_true", help="Use CLIP for visual")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load vocabulary
    vocab = Vocabulary()
    vocab.load(str(OUTPUT_DIR / "vocab.json"))

    # Load model
    model = load_model(args.checkpoint, vocab.vocab_size, device)

    # Optionally load CLIP
    clip_model, clip_preprocess = None, None
    if args.use_clip:
        try:
            import clip
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            print("✅ CLIP model loaded")
        except ImportError:
            print("⚠️  CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

    # Build tokens and predict
    tokens = build_tokens_custom(
        args.video, clip_model, clip_preprocess, device
    )

    with torch.no_grad():
        title = generate_title(model, tokens, vocab, device)

    print(f"\n🆕 UNSEEN VIDEO: {args.video}")
    print(f"   PREDICTED TITLE: {title}")


if __name__ == "__main__":
    main()
