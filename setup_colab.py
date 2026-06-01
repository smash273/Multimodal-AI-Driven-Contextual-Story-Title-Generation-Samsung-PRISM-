"""
PRISM - Colab Setup Script.

Run this as the first step on Google Colab to set up the environment.

Usage (in Colab):
    !python setup_colab.py
"""
import os
import sys
import subprocess


def check_colab():
    """Check if we're running on Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive():
    """Mount Google Drive on Colab."""
    if not check_colab():
        print("⚠️  Not on Colab, skipping Drive mount")
        return False

    from google.colab import drive
    drive.mount("/content/drive")
    print("✅ Google Drive mounted at /content/drive")
    return True


def verify_data():
    """Verify that the data directory exists and has expected contents."""
    from config import BASE_DIR, VIDEO_DIR, CLIP_DIR, AUDIO_DIR, OCR_DIR, ASR_DIR

    print("\n📂 Verifying data directories...")

    dirs_to_check = {
        "Base directory": BASE_DIR,
        "Videos":         VIDEO_DIR,
        "CLIP features":  CLIP_DIR,
        "Audio embeddings": AUDIO_DIR,
        "OCR text":       OCR_DIR,
        "ASR text":       ASR_DIR,
    }

    all_ok = True
    for name, path in dirs_to_check.items():
        if path.exists():
            # Count files
            count = len(list(path.iterdir())) if path.is_dir() else 0
            print(f"   ✅ {name}: {path} ({count} items)")
        else:
            print(f"   ❌ {name}: {path} (NOT FOUND)")
            all_ok = False

    if not all_ok:
        print("\n⚠️  Some directories are missing!")
        print("   Make sure your Google Drive has the correct data structure:")
        print("   videostory_prism/")
        print("   ├── videos/raw/        # .mp4 video files")
        print("   ├── clip_features/     # .npy CLIP features")
        print("   ├── audio_embeddings/  # .npy audio embeddings")
        print("   ├── ocr_text/          # .txt OCR files")
        print("   └── audio_text/        # .txt ASR files")

    return all_ok


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "-r", "requirements.txt"
    ])
    print("   ✅ Dependencies installed")


def check_gpu():
    """Check GPU availability."""
    import torch
    print(f"\n🔧 System Info:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {gpu_mem:.1f} GB")


def main():
    print("=" * 60)
    print("PRISM - Colab Setup")
    print("=" * 60)

    # Step 1: Mount Drive
    mount_drive()

    # Step 2: Install deps
    install_dependencies()

    # Step 3: Check GPU
    check_gpu()

    # Step 4: Verify data
    data_ok = verify_data()

    print("\n" + "=" * 60)
    if data_ok:
        print("✅ Setup complete! Ready to go.")
        print("\nNext steps:")
        print("  1. python main.py vlm_label --preview 3   # Test VLM on 3 videos")
        print("  2. python main.py vlm_label               # Generate all titles")
        print("  3. python main.py train --epochs 25       # Train the model")
        print("  4. python main.py infer                   # Run inference")
        print("  5. python main.py eval                    # Evaluate results")
    else:
        print("⚠️  Setup incomplete — fix data paths before training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
