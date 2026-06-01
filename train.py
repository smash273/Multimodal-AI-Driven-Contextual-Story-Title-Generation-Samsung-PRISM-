"""
PRISM - Training script for the Multimodal Transformer model.

Usage:
    python train.py [--epochs 25] [--batch_size 8] [--lr 3e-5]
    python train.py --gt_csv path/to/custom_titles.csv
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    LABEL_SMOOTHING, GRAD_CLIP_NORM,
    PAD_IDX, CHECKPOINT_DIR, OUTPUT_DIR,
)
from data.data_utils import load_gt_csv, build_vocabulary_from_csv
from data.text_encoder import load_bert
from data.dataset import VideoTitleDataset, collate_fn
from models.transformer_model import MultimodalTransformer


def train(args):
    """Main training loop."""

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")

    # Create output directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data ----
    print("📦 Loading ground truth CSV...")
    gt_df = load_gt_csv(args.gt_csv)
    print(f"   Total training videos: {len(gt_df)}")

    if len(gt_df) == 0:
        print("❌ No training data found!")
        print("   Run 'python generate_gt_vlm.py' first to generate titles.")
        return

    # ---- Build vocabulary ----
    print("📖 Building vocabulary...")
    vocab = build_vocabulary_from_csv(args.gt_csv)
    print(f"   Vocab size: {vocab.vocab_size}")

    # Save vocabulary
    vocab_path = str(OUTPUT_DIR / "vocab.json")
    vocab.save(vocab_path)
    print(f"   Saved vocabulary to {vocab_path}")

    # ---- Load BERT ----
    print("🤖 Loading BERT text encoder...")
    bert_tokenizer, bert_model = load_bert(device)

    # ---- Create dataset and loader ----
    print("📊 Creating dataset...")
    dataset = VideoTitleDataset(
        gt_df, vocab,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
        device=device,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # ---- Create model ----
    print("🏗️  Initializing Multimodal Transformer...")
    model = MultimodalTransformer(vocab.vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # ---- Training loop ----
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 60)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for tokens, captions in pbar:
            tokens = tokens.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            logits = model(tokens, captions)
            loss = criterion(
                logits.reshape(-1, vocab.vocab_size),
                captions[:, 1:].reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = str(CHECKPOINT_DIR / "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "vocab_size": vocab.vocab_size,
            }, best_path)
            print(f"   ✅ Best model saved (loss: {avg_loss:.4f})")

    # Save final model
    final_path = str(CHECKPOINT_DIR / "final_model.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "vocab_size": vocab.vocab_size,
    }, final_path)
    print(f"\n💾 Final model saved to {final_path}")
    print(f"💾 Best model saved to {best_path}")
    print("✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train PRISM Multimodal Transformer")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--gt_csv", type=str, default=None,
                        help="Path to ground truth CSV (default: auto-detect)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
