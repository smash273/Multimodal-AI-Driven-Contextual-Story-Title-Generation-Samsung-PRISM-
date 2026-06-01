"""
PRISM - Data loading and preprocessing utilities.
Handles reading CLIP features, audio embeddings, OCR/ASR text, and ground truth.
"""
import os
import json
import csv
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from config import (
    CLIP_DIR, AUDIO_DIR, OCR_DIR, ASR_DIR, GT_DIR,
    VLM_GT_CSV, MANUAL_GT_CSV, get_gt_csv,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX,
    MAX_VOCAB_SIZE,
)


# ============================================================
# RAW FILE LOADERS
# ============================================================

def load_clip(video_id: str) -> np.ndarray:
    """Load pre-extracted CLIP visual features for a video."""
    path = CLIP_DIR / f"{video_id}.npy"
    return np.load(path)


def load_audio_embedding(video_id: str) -> np.ndarray:
    """Load pre-extracted audio embeddings for a video."""
    path = AUDIO_DIR / f"{video_id}.npy"
    return np.load(path)


def load_text(video_id: str, folder: Path) -> str:
    """Load a text file (OCR or ASR) for a video."""
    path = folder / f"{video_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_ocr(video_id: str) -> str:
    """Load OCR text for a video."""
    return load_text(video_id, OCR_DIR)


def load_asr(video_id: str) -> str:
    """Load ASR text for a video."""
    return load_text(video_id, ASR_DIR)


def load_gt_json(video_id: str) -> str:
    """Load ground truth title from JSON format (legacy)."""
    path = GT_DIR / f"{video_id}.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    return data.get("title", "")


def read_text_file(path: str) -> str:
    """Read a text file safely."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    return ""


# ============================================================
# GROUND TRUTH CSV LOADING
# ============================================================

def load_gt_csv(csv_path: str = None) -> pd.DataFrame:
    """
    Load the ground truth titles CSV.

    Priority:
        1. Explicit csv_path if provided
        2. VLM-generated titles (vlm_titles.csv)
        3. Manual titles (manual_titles_template.csv)

    Returns:
        DataFrame with columns: video_id, title
    """
    if csv_path is not None:
        gt_csv = Path(csv_path)
    else:
        gt_csv = get_gt_csv()

    if not gt_csv.exists():
        raise FileNotFoundError(
            f"Ground truth CSV not found: {gt_csv}\n"
            f"Run 'python generate_gt_vlm.py' first to generate titles."
        )

    print(f"📄 Loading GT from: {gt_csv.name}")

    try:
        df = pd.read_csv(gt_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(gt_csv, encoding="latin-1")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure required columns
    if "video_id" not in df.columns or "title" not in df.columns:
        raise ValueError(
            f"CSV must have 'video_id' and 'title' columns. "
            f"Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["title"])
    df["title"] = df["title"].astype(str)

    # Filter out empty titles
    df = df[df["title"].str.strip().str.len() > 0].reset_index(drop=True)

    return df


def get_video_ids() -> list:
    """Get sorted list of video IDs from CLIP features directory."""
    return sorted([p.stem for p in CLIP_DIR.glob("*.npy")])


def get_video_ids_numeric() -> list:
    """Get video IDs sorted by numeric part (vs_3, vs_4, ...)."""
    video_ids = [f.replace(".npy", "") for f in os.listdir(CLIP_DIR) if f.endswith(".npy")]
    return sorted(video_ids, key=lambda x: int(x.split("_")[1]))


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_asr_text(text: str, max_words: int = 50) -> str:
    """Clean and truncate ASR text for encoding."""
    if not text or len(text.strip()) == 0:
        return ""
    text = text.lower()
    lines = list(dict.fromkeys(text.splitlines()))
    text = " ".join(lines)
    words = text.split()
    return " ".join(words[:max_words])


def normalize_embedding(x: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector."""
    return x / (np.linalg.norm(x) + 1e-8)


# ============================================================
# VOCABULARY
# ============================================================

class Vocabulary:
    """Manages word-to-index and index-to-word mappings."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_from_titles(self, titles: list, max_vocab_size: int = MAX_VOCAB_SIZE):
        """Build vocabulary from a list of title strings."""
        counter = Counter()
        for title in titles:
            counter.update(self.tokenize(title))

        vocab_words = [
            PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN
        ] + [w for w, _ in counter.most_common(max_vocab_size - 4)]

        self.word2idx = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    @staticmethod
    def tokenize(text: str) -> list:
        """Simple whitespace tokenizer with lowercasing."""
        return text.lower().split()

    def encode_title(self, text: str, max_len: int = 12) -> list:
        """Encode a title string to a list of token indices."""
        tokens = self.tokenize(text)
        encoded = [BOS_IDX]
        for t in tokens:
            encoded.append(self.word2idx.get(t, UNK_IDX))
        encoded.append(EOS_IDX)

        if len(encoded) < max_len:
            encoded += [PAD_IDX] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]

        return encoded

    def decode_title(self, indices: list) -> str:
        """Decode a list of token indices to a title string."""
        words = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, BOS_IDX):
                continue
            words.append(self.idx2word.get(idx, UNK_TOKEN))
        return " ".join(words)

    def save(self, path: str):
        """Save vocabulary to a JSON file."""
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load vocabulary from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.vocab_size = len(self.word2idx)


def build_vocabulary_from_csv(csv_path: str = None) -> Vocabulary:
    """Build a Vocabulary object from the ground truth CSV."""
    df = load_gt_csv(csv_path)
    vocab = Vocabulary()
    vocab.build_from_titles(df["title"].tolist())
    return vocab
