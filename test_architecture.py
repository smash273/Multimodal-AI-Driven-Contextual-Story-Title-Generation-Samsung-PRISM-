"""
PRISM - Architecture Test Script.

Tests the model architecture, dataset, and training loop with synthetic data.
This script does NOT require Google Drive or real data — it verifies that
all components work together correctly.

Usage:
    python test_architecture.py
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX, MAX_TITLE_LEN,
)


def test_model_instantiation():
    """Test 1: Model can be instantiated with various vocab sizes."""
    print("🧪 Test 1: Model instantiation")

    from models.transformer_model import MultimodalTransformer

    for vocab_size in [100, 500, 1000, 5000]:
        model = MultimodalTransformer(vocab_size)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   vocab_size={vocab_size:5d} | params={total_params:,}")

    print("   ✅ PASSED\n")
    return model


def test_forward_pass():
    """Test 2: Forward pass with teacher forcing."""
    print("🧪 Test 2: Forward pass (teacher forcing)")

    from models.transformer_model import MultimodalTransformer

    vocab_size = 500
    model = MultimodalTransformer(vocab_size)
    model.eval()

    batch_size = 4
    num_modalities = 3
    seq_len = 8

    # Simulate input: (B, 3, 512) multimodal tokens
    tokens = torch.randn(batch_size, num_modalities, EMBED_DIM)

    # Simulate captions: (B, seq_len) with BOS at start
    captions = torch.randint(4, vocab_size, (batch_size, seq_len))
    captions[:, 0] = BOS_IDX

    with torch.no_grad():
        logits = model(tokens, captions)

    expected_shape = (batch_size, seq_len - 1, vocab_size)
    assert logits.shape == expected_shape, \
        f"Expected {expected_shape}, got {logits.shape}"

    print(f"   Input tokens:   {tokens.shape}")
    print(f"   Captions:       {captions.shape}")
    print(f"   Output logits:  {logits.shape}")
    print("   ✅ PASSED\n")


def test_encode_decode():
    """Test 3: Encode and decode step (autoregressive inference)."""
    print("🧪 Test 3: Encode + decode step (autoregressive)")

    from models.transformer_model import MultimodalTransformer

    vocab_size = 500
    model = MultimodalTransformer(vocab_size)
    model.eval()
    device = "cpu"

    # Single video input: (1, 3, 512)
    tokens = torch.randn(1, 3, EMBED_DIM)

    with torch.no_grad():
        # Encode
        memory = model.encode(tokens)
        assert memory.shape == (1, 3, EMBED_DIM), \
            f"Memory shape: {memory.shape}"

        # Decode step by step
        generated = [BOS_IDX]
        for step in range(MAX_TITLE_LEN):
            logits = model.decode_step(generated, memory, device)
            next_token = logits.argmax(dim=-1).item()

            if next_token == EOS_IDX:
                break
            generated.append(next_token)

    print(f"   Memory shape:      {memory.shape}")
    print(f"   Generated tokens:  {generated}")
    print(f"   Generated length:  {len(generated)} (including BOS)")
    print("   ✅ PASSED\n")


def test_vocabulary():
    """Test 4: Vocabulary build, encode, decode."""
    print("🧪 Test 4: Vocabulary")

    from data.data_utils import Vocabulary

    vocab = Vocabulary()
    titles = [
        "cat playing with yarn",
        "dog running in park",
        "baby laughing at dog",
        "car crash on highway",
        "sunset over the ocean",
    ]
    vocab.build_from_titles(titles)

    print(f"   Vocab size: {vocab.vocab_size}")
    assert vocab.vocab_size > 4  # At least special tokens + words

    # Encode and decode roundtrip
    test_title = "cat playing with yarn"
    encoded = vocab.encode_title(test_title, max_len=10)
    decoded = vocab.decode_title(encoded)

    print(f"   Original: '{test_title}'")
    print(f"   Encoded:  {encoded}")
    print(f"   Decoded:  '{decoded}'")
    assert decoded == test_title, f"Roundtrip failed: '{decoded}' != '{test_title}'"

    # Save and load
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        vocab_path = f.name

    vocab.save(vocab_path)
    vocab2 = Vocabulary()
    vocab2.load(vocab_path)
    assert vocab2.vocab_size == vocab.vocab_size
    os.unlink(vocab_path)

    print("   ✅ PASSED\n")


def test_dataset_collation():
    """Test 5: Dataset batching and collation."""
    print("🧪 Test 5: Dataset collation")

    from data.data_utils import Vocabulary
    from data.dataset import collate_fn

    vocab = Vocabulary()
    vocab.build_from_titles([
        "test title one two three",
        "another test",
        "short",
    ])

    # Simulate dataset items
    batch = []
    for i in range(4):
        tokens = torch.randn(3, EMBED_DIM)  # 3 modalities
        caption_len = torch.randint(3, 8, (1,)).item()
        caption = torch.randint(4, vocab.vocab_size, (caption_len,))
        caption[0] = BOS_IDX
        batch.append((tokens, caption))

    # Collate
    tokens_batch, captions_batch = collate_fn(batch)

    print(f"   Tokens batch:   {tokens_batch.shape}")
    print(f"   Captions batch: {captions_batch.shape}")

    assert tokens_batch.shape[0] == 4
    assert tokens_batch.shape[1] == 3
    assert tokens_batch.shape[2] == EMBED_DIM
    assert captions_batch.shape[0] == 4

    print("   ✅ PASSED\n")


def test_training_loop():
    """Test 6: Full training loop with synthetic data."""
    print("🧪 Test 6: Training loop (3 epochs, synthetic data)")

    from models.transformer_model import MultimodalTransformer
    from data.dataset import collate_fn

    vocab_size = 200
    model = MultimodalTransformer(vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create synthetic batches
    num_batches = 5
    batch_size = 4
    seq_len = 8

    losses = []

    for epoch in range(3):
        model.train()
        epoch_loss = 0

        for _ in range(num_batches):
            tokens = torch.randn(batch_size, 3, EMBED_DIM).to(device)
            captions = torch.randint(4, vocab_size, (batch_size, seq_len)).to(device)
            captions[:, 0] = BOS_IDX

            optimizer.zero_grad()
            logits = model(tokens, captions)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                captions[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"   Epoch {epoch + 1}/3 | Loss: {avg_loss:.4f}")

    # Loss should decrease
    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    print(f"   Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
    print("   ✅ PASSED\n")


def test_full_inference_pipeline():
    """Test 7: Full inference pipeline (model → generate title)."""
    print("🧪 Test 7: Full inference pipeline")

    from models.transformer_model import MultimodalTransformer
    from data.data_utils import Vocabulary

    # Build vocab
    vocab = Vocabulary()
    vocab.build_from_titles([
        "cat playing with toy",
        "dog running in the park",
        "baby laughing at funny dog",
        "car crash on busy highway",
        "sunset over the ocean waves",
    ])

    # Create and lightly train a model
    model = MultimodalTransformer(vocab.vocab_size)
    device = "cpu"

    # Generate title for random input
    model.eval()
    tokens = torch.randn(1, 3, EMBED_DIM)

    with torch.no_grad():
        memory = model.encode(tokens)
        generated = [BOS_IDX]
        for _ in range(MAX_TITLE_LEN):
            logits = model.decode_step(generated, memory, device)
            next_token = logits.argmax(dim=-1).item()
            if next_token == EOS_IDX:
                break
            generated.append(next_token)

    title = vocab.decode_title(generated[1:])
    print(f"   Generated title: '{title}'")
    print(f"   Token count:     {len(generated) - 1}")
    print("   ✅ PASSED\n")


def test_checkpoint_save_load():
    """Test 8: Save and load model checkpoint."""
    print("🧪 Test 8: Checkpoint save/load")

    import tempfile
    from models.transformer_model import MultimodalTransformer

    vocab_size = 200
    model = MultimodalTransformer(vocab_size)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    torch.save({
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "loss": 2.5,
        "vocab_size": vocab_size,
    }, ckpt_path)

    # Load
    model2 = MultimodalTransformer(vocab_size)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model2.load_state_dict(checkpoint["model_state_dict"])

    # Verify outputs match
    model.eval()
    model2.eval()
    test_input = torch.randn(1, 3, EMBED_DIM)
    test_caption = torch.randint(4, vocab_size, (1, 6))
    test_caption[:, 0] = BOS_IDX

    with torch.no_grad():
        out1 = model(test_input, test_caption)
        out2 = model2(test_input, test_caption)

    assert torch.allclose(out1, out2), "Loaded model produces different output!"
    os.unlink(ckpt_path)

    print(f"   Checkpoint saved/loaded successfully")
    print(f"   Outputs match: ✅")
    print("   ✅ PASSED\n")


def test_bert_encoder():
    """Test 9: BERT text encoder (if transformers is available)."""
    print("🧪 Test 9: BERT text encoder")

    try:
        from data.text_encoder import load_bert, encode_text_bert

        tokenizer, bert_model = load_bert("cpu")
        emb = encode_text_bert("hello world", tokenizer, bert_model, "cpu")

        assert emb.shape == (768,), f"Expected (768,), got {emb.shape}"
        print(f"   Embedding shape: {emb.shape}")

        # Empty text
        emb_empty = encode_text_bert("", tokenizer, bert_model, "cpu")
        assert emb_empty.shape == (768,)
        assert torch.all(emb_empty == 0), "Empty text should give zero vector"
        print(f"   Empty text:      {emb_empty.shape} (all zeros ✅)")

        print("   ✅ PASSED\n")
    except ImportError as e:
        print(f"   ⚠️  SKIPPED (missing dependency: {e})\n")


def main():
    print("=" * 60)
    print("PRISM ARCHITECTURE TEST SUITE")
    print("=" * 60)
    print()

    tests = [
        test_model_instantiation,
        test_forward_pass,
        test_encode_decode,
        test_vocabulary,
        test_dataset_collation,
        test_training_loop,
        test_full_inference_pipeline,
        test_checkpoint_save_load,
        test_bert_encoder,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"   ❌ FAILED: {e}\n")

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed! Architecture is working correctly.")


if __name__ == "__main__":
    main()
