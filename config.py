"""
PRISM - Video Title Generation
Configuration file for all paths, hyperparameters, and constants.

Designed to run on Google Colab with Google Drive mounted at /content/drive.
"""
import os
from pathlib import Path

# ============================================================
# DATA PATHS (Colab + Google Drive)
# ============================================================
# Default: Colab with Drive mounted at /content/drive
# Override with PRISM_DATA_DIR environment variable if needed
BASE_DIR = Path(os.environ.get(
    "PRISM_DATA_DIR",
    "/content/drive/MyDrive/videostory_prism"
))

VIDEO_DIR   = BASE_DIR / "videos" / "raw"
CLIP_DIR    = BASE_DIR / "clip_features"
AUDIO_DIR   = BASE_DIR / "audio_embeddings"
OCR_DIR     = BASE_DIR / "ocr_text"
ASR_DIR     = BASE_DIR / "audio_text"

# Video OCR extraction (see extract_ocr.py / main.py ocr_extract)
# Sample at least ~0.5 frames/sec by default (more frames → better recall, slower).
OCR_EXTRACT_FPS = float(os.environ.get("OCR_EXTRACT_FPS", "0.5"))
OCR_ENGINE_DEFAULT = os.environ.get("OCR_ENGINE", "rapid").strip().lower()  # rapid | paddle
GT_DIR      = BASE_DIR / "processed" / "texts"
PROCESSED   = BASE_DIR / "processed"

# Ground truth CSV files
MANUAL_GT_CSV = PROCESSED / "manual_titles_template.csv"
VLM_GT_CSV    = PROCESSED / "vlm_titles.csv"

# Which GT to use for training (vlm_titles.csv preferred, fallback to manual)
def get_gt_csv() -> Path:
    """Return the best available ground truth CSV path."""
    if VLM_GT_CSV.exists():
        return VLM_GT_CSV
    if MANUAL_GT_CSV.exists():
        return MANUAL_GT_CSV
    # Default to VLM path (will be created by generate_gt_vlm.py)
    return VLM_GT_CSV

# Output directories (inside project, not on Drive)
OUTPUT_DIR     = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR    = OUTPUT_DIR / "results"

# ============================================================
# SPECIAL TOKENS
# ============================================================
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
EMBED_DIM       = 512
NUM_HEADS       = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
MAX_VOCAB_SIZE  = 5000
MAX_TITLE_LEN   = 12

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE      = 8
LEARNING_RATE   = 3e-5
EPOCHS          = 25
LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM  = 1.0

# ============================================================
# AUDIO CLASSIFICATION (YAMNet)
# ============================================================
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASSES_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "master/research/audioset/yamnet/yamnet_class_map.csv"
)
AUDIO_SAMPLE_RATE = 16000
AUDIO_ENERGY_THRESHOLD = 0.08

# ============================================================
# VISUAL (CLIP)
# ============================================================
NUM_SAMPLE_FRAMES = 16

# ============================================================
# BERT TEXT ENCODER
# ============================================================
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128

# ============================================================
# VLM (Qwen2-VL) for Ground Truth Generation
# ============================================================
VLM_MODEL_NAME    = "Qwen/Qwen2-VL-2B-Instruct"  # Fits on Colab T4 (16GB)
# IMPORTANT: qwen-vl-utils enforces VIDEO_MIN_PIXELS = 128*28*28 = 100352.
# max_pixels must be >= that value or process_vision_info will raise ValueError.
VLM_MAX_PIXELS    = 360 * 280   # 100_800 — just above the library minimum
# Slightly higher FPS = more temporal context (theft, news cuts); lower if OOM.
VLM_VIDEO_FPS     = float(os.environ.get("VLM_VIDEO_FPS", "0.75"))
VLM_MAX_NEW_TOKENS = 50
VLM_TEMPERATURE   = 0.7

# Match PRISM’s multimodal idea (same as training: video + OCR + ASR). Gemini/Drive-style
# systems use a large VLM plus extra signals; we inject OCR/ASR when those .txt files exist.
VLM_USE_AUX_TEXT = os.environ.get("PRISM_VLM_NO_AUX_TEXT", "").strip().lower() not in (
    "1", "true", "yes",
)
VLM_MAX_OCR_CHARS = int(os.environ.get("VLM_MAX_OCR_CHARS", "700"))
VLM_MAX_ASR_CHARS = int(os.environ.get("VLM_MAX_ASR_CHARS", "900"))

VLM_PROMPT = (
    "Watch the video and write ONE short factual title (4–12 words).\n\n"
    "Rules:\n"
    "- Base the title ONLY on what you actually see and hear. Do NOT invent animals, places, "
    "or actions that are not clearly in the video.\n"
    "- If it looks like a news segment, studio interview, or CCTV/security footage, say that "
    "and the main action (e.g. anchor reporting, car leaving a driveway).\n"
    "- If optional OCR/ASR hints are given below, use them to disambiguate; if they conflict "
    "with the video, trust the video.\n"
    "- No quotation marks around the title. No hashtags or emojis.\n"
    "- Avoid generic clickbait words (amazing, viral, epic, you won't believe).\n\n"
    "Output ONLY the title, no other text."
)
