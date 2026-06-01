# PRISM — paste each `# %%` block into a separate Colab code cell (top to bottom).
# Or upload PRISM_Colab.ipynb to Colab.
# Repo clone defaults: BhavyeNijhawan / Multimodal-AI-Driven-Contextual-Story-Title-Generation

# %% CELL 1 — Mount Google Drive
from google.colab import drive

drive.mount("/content/drive")

# %% CELL 2 — GitHub PAT (Colab secret name: GH_PAT)
import os
from getpass import getpass

try:
    from google.colab import userdata
    try:
        os.environ["GH_PAT"] = userdata.get("GH_PAT")
        print("Using GH_PAT from Colab secrets.")
    except userdata.SecretNotFoundError:
        os.environ["GH_PAT"] = getpass("GitHub PAT (repo scope): ").strip()
except Exception:
    os.environ["GH_PAT"] = getpass("GitHub PAT (repo scope): ").strip()

assert os.environ.get("GH_PAT"), "GH_PAT is empty"

# %% CELL 3 — Clone (change OWNER/REPO if you forked)
import shutil

REPO_DIR = "/content/prism_repo"
GITHUB_OWNER = "BhavyeNijhawan"
GITHUB_REPO = "Multimodal-AI-Driven-Contextual-Story-Title-Generation"

token = os.environ["GH_PAT"]
clone_url = f"https://x-access-token:{token}@github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"

if os.path.isdir(REPO_DIR):
    shutil.rmtree(REPO_DIR)

subprocess.run(["git", "clone", clone_url, REPO_DIR], check=True)
print("Cloned to", REPO_DIR)

# %% CELL 4 — Install dependencies
# In Colab, run as two lines in one cell:
# %cd /content/prism_repo
# !pip install -q -r requirements.txt

# %% CELL 5 — (Optional) Hugging Face token — secret name: HF_TOKEN
import subprocess

try:
    from google.colab import userdata
    try:
        _hf = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = _hf
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf
        print("HF_TOKEN loaded from Colab secrets.")
    except userdata.SecretNotFoundError:
        print("No HF_TOKEN secret (optional unless Hub asks for login).")
except Exception as e:
    print("userdata not available:", e)

_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _tok:
    try:
        from huggingface_hub import login
        login(token=_tok, add_to_git_credential=True)
        print("Logged into Hugging Face via Python API.")
    except Exception as e:
        print("HF login skipped/failed (usually safe unless model access is gated):", e)

# %% CELL 6 — Data root on Drive (matches config.py default)
os.environ["PRISM_DATA_DIR"] = "/content/drive/MyDrive/videostory_prism"
print("PRISM_DATA_DIR =", os.environ["PRISM_DATA_DIR"])

# %% CELL 7 — GPU + folder sanity check
# %cd /content/prism_repo
import torch
from pathlib import Path

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

_base = Path(os.environ.get("PRISM_DATA_DIR", "/content/drive/MyDrive/videostory_prism"))
for _name, _sub in [
    ("videos", _base / "videos" / "raw"),
    ("clip_features", _base / "clip_features"),
    ("ocr_text", _base / "ocr_text"),
    ("audio_text", _base / "audio_text"),
    ("audio_embeddings", _base / "audio_embeddings"),
]:
    _ok = _sub.is_dir() and any(_sub.iterdir())
    print(f"{_name}: {'OK' if _ok else 'MISSING or empty'} — {_sub}")

# %% CELL 8 — Architecture test (no dataset)
# %cd /content/prism_repo
# !python main.py test

# %% CELL 9 — VLM preview (3 videos)
# %cd /content/prism_repo
# !python main.py vlm_label --preview 3

# %% CELL 10 — VLM all videos
# %cd /content/prism_repo
# !python main.py vlm_label

# %% CELL 11 — Train
# %cd /content/prism_repo
# !python main.py train --epochs 25 --batch_size 8

# %% CELL 12 — Infer
# %cd /content/prism_repo
# !python main.py infer

# %% CELL 13 — Eval
# %cd /content/prism_repo
# !python main.py eval
