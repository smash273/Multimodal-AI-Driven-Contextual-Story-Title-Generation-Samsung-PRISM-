"""
PRISM - Inference script for generating video titles.

Usage:
    python inference.py --checkpoint outputs/checkpoints/best_model.pt
    python inference.py --checkpoint outputs/checkpoints/best_model.pt --video_id vs_10
"""
import os
import sys
import argparse
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLIP_DIR, OCR_DIR, ASR_DIR,
    BOS_IDX, EOS_IDX, PAD_IDX,
    MAX_TITLE_LEN, EMBED_DIM,
    OUTPUT_DIR, RESULTS_DIR,
)
from data.data_utils import (
    load_clip, load_text, load_gt_csv,
    Vocabulary, get_video_ids,
)
from data.text_encoder import load_bert, encode_text_bert
from models.transformer_model import MultimodalTransformer


def generate_title(
    model: MultimodalTransformer,
    tokens: torch.Tensor,
    vocab: Vocabulary,
    device: str = "cuda",
    max_len: int = MAX_TITLE_LEN,
) -> str:
    """
    Generate a title from multimodal tokens using greedy decoding.

    Args:
        model: Trained MultimodalTransformer
        tokens: (num_modalities, 512) tensor
        vocab: Vocabulary object
        device: Device string
        max_len: Maximum title length

    Returns:
        Generated title string
    """
    model.eval()
    tokens = tokens.unsqueeze(0).to(device)

    # Encode multimodal tokens
    memory = model.encode(tokens)

    # Greedy decode
    generated = [BOS_IDX]

    for _ in range(max_len):
        logits = model.decode_step(generated, memory, device)
        next_token = logits.argmax(dim=-1).item()

        if next_token == EOS_IDX:
            break
        generated.append(next_token)

    return vocab.decode_title(generated[1:])


def load_model(checkpoint_path: str, vocab_size: int, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    model = MultimodalTransformer(vocab_size).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', '?')}")
    print(f"   Loss: {checkpoint.get('loss', '?'):.4f}")

    return model


def build_tokens_for_video(
    video_id: str,
    bert_tokenizer,
    bert_model,
    device: str = "cpu",
) -> torch.Tensor:
    """Build multimodal token tensor for a single video."""
    clip_emb = torch.tensor(load_clip(video_id)).float()

    ocr_text = load_text(video_id, OCR_DIR)
    asr_text = load_text(video_id, ASR_DIR)

    ocr_emb = encode_text_bert(ocr_text, bert_tokenizer, bert_model, device)[:EMBED_DIM]
    asr_emb = encode_text_bert(asr_text, bert_tokenizer, bert_model, device)[:EMBED_DIM]

    return torch.stack([clip_emb, ocr_emb, asr_emb])


def run_inference(args):
    """Run inference on dataset or single video."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    # Load vocabulary
    vocab_path = str(OUTPUT_DIR / "vocab.json")
    vocab = Vocabulary()
    vocab.load(vocab_path)
    print(f"📖 Vocabulary loaded ({vocab.vocab_size} words)")

    # Load model
    model = load_model(args.checkpoint, vocab.vocab_size, device)

    # Load BERT
    print("🤖 Loading BERT text encoder...")
    bert_tokenizer, bert_model = load_bert(device)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.video_id:
        # Single video inference
        print(f"\n🎥 Generating title for video: {args.video_id}")
        tokens = build_tokens_for_video(
            args.video_id, bert_tokenizer, bert_model, device
        )
        title = generate_title(model, tokens, vocab, device)
        print(f"   PREDICTED TITLE: {title}")

    else:
        # Run on all videos
        print("\n📊 Running inference on all videos...")
        gt_df = load_gt_csv()

        results = []

        with torch.no_grad():
            for idx in range(len(gt_df)):
                row = gt_df.iloc[idx]
                vid = row.video_id
                gt_title = row.title

                tokens = build_tokens_for_video(
                    vid, bert_tokenizer, bert_model, device
                )
                pred_title = generate_title(model, tokens, vocab, device)

                results.append({
                    "index": idx,
                    "video_id": vid,
                    "ground_truth": gt_title,
                    "prediction": pred_title,
                })

                print(f"[{idx}] {vid}")
                print(f"  GT  : {gt_title}")
                print(f"  PRED: {pred_title}")
                print("-" * 60)

        # Save results
        results_df = pd.DataFrame(results)
        results_path = str(RESULTS_DIR / "predictions.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved to {results_path}")
        print(f"✅ Inference complete on {len(results)} videos!")


def main():
    parser = argparse.ArgumentParser(description="PRISM Inference")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(OUTPUT_DIR / "checkpoints" / "best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--video_id", type=str, default=None,
        help="Single video ID to generate title for (e.g., vs_10)",
    )
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
