"""
PRISM - Audio processing utilities.
Handles YAMNet-based audio classification and audio feature extraction.
"""
import csv
import numpy as np
import requests

from config import (
    YAMNET_MODEL_URL, YAMNET_CLASSES_URL,
    AUDIO_SAMPLE_RATE, AUDIO_ENERGY_THRESHOLD,
    VIDEO_DIR, EMBED_DIM,
)


# ============================================================
# YAMNet MODEL LOADING
# ============================================================

_yamnet_model = None
_yamnet_classes = None


def load_yamnet():
    """Load YAMNet model and class labels (lazy singleton)."""
    global _yamnet_model, _yamnet_classes

    if _yamnet_model is None:
        import tensorflow_hub as hub
        print("🔊 Loading YAMNet model...")
        _yamnet_model = hub.load(YAMNET_MODEL_URL)

    if _yamnet_classes is None:
        print("🔊 Loading YAMNet class labels...")
        response = requests.get(YAMNET_CLASSES_URL)
        class_map = response.text.splitlines()
        reader = csv.DictReader(class_map)
        _yamnet_classes = [row["display_name"] for row in reader]
        print(f"   Total YAMNet classes: {len(_yamnet_classes)}")

    return _yamnet_model, _yamnet_classes


# ============================================================
# AUDIO HELPERS
# ============================================================

def load_audio_from_video(video_path, sr=AUDIO_SAMPLE_RATE):
    """Load audio waveform from a video file using librosa."""
    import librosa
    try:
        waveform, _ = librosa.load(str(video_path), sr=sr, mono=True)
        return waveform
    except Exception:
        return None


def audio_energy_level(waveform: np.ndarray) -> float:
    """Compute mean absolute energy of a waveform."""
    return float(np.mean(np.abs(waveform)))


def audio_to_text(video_id: str, top_k: int = 3) -> str:
    """
    Classify audio from a video using YAMNet and return
    a comma-separated string of top-k class labels.
    """
    import tensorflow as tf

    yamnet_model, yamnet_classes = load_yamnet()

    video_path = VIDEO_DIR / f"{video_id}.mp4"
    waveform = load_audio_from_video(video_path)

    if waveform is None or len(waveform) == 0:
        return "No significant audio"

    energy = audio_energy_level(waveform)

    scores, _, _ = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_indices = mean_scores.argsort()[-top_k:][::-1]
    labels = [yamnet_classes[i] for i in top_indices]

    if energy > AUDIO_ENERGY_THRESHOLD:
        labels.append("loud impact")

    return ", ".join(set(labels))


def extract_audio_embedding(video_path, embed_dim: int = EMBED_DIM) -> np.ndarray:
    """
    Extract audio embedding from a video by averaging YAMNet embeddings.
    Pads or truncates to embed_dim.
    """
    import tensorflow as tf

    yamnet_model, _ = load_yamnet()
    wav = load_audio_from_video(video_path)

    if wav is None or len(wav) == 0:
        return np.zeros(embed_dim)

    scores, embeddings, _ = yamnet_model(wav)
    emb = tf.reduce_mean(embeddings, axis=0).numpy()

    # Pad or truncate to embed_dim
    if emb.shape[0] < embed_dim:
        emb = np.pad(emb, (0, embed_dim - emb.shape[0]))
    else:
        emb = emb[:embed_dim]

    return emb
