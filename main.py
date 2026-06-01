"""
PRISM - Video Title Generation Pipeline
Main entry point for training, inference, and evaluation.

Usage:
    python main.py vlm_label [--preview 5] [--model Qwen/Qwen2-VL-2B-Instruct]
    python main.py train [--epochs 25] [--batch_size 8]
    python main.py infer [--checkpoint path/to/model.pt] [--video_id vs_10]
    python main.py eval [--results path/to/predictions.csv]
    python main.py predict --video path/to/video.mp4
    python main.py test   # test architecture with dummy data
    python main.py ocr_extract  # RapidOCR @ 0.5 FPS → ocr_text/*.txt
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print("""
╔════════════════════════════════════════════════════════════╗
║                 PRISM Video Title Generator                ║
║         Multimodal Transformer for Video Storytelling      ║
║                    + Qwen2-VL Labeler                      ║
╚════════════════════════════════════════════════════════════╝
    """)


def main():
    print_banner()

    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print()
        print("Commands:")
        print("  vlm_label - Generate ground truth titles using Qwen2-VL")
        print("  train     - Train the multimodal transformer model")
        print("  infer     - Run inference on dataset")
        print("  eval      - Evaluate predictions against ground truth")
        print("  predict   - Generate title for a custom video")
        print("  pseudo     - Generate pseudo-labels using GPT")
        print("  ocr_extract - Extract OCR text from videos (RapidOCR / PaddleOCR)")
        print("  test       - Test model architecture with dummy data")
        print()
        print("Typical workflow on Colab:")
        print("  0. python main.py ocr_extract        # Optional: fill ocr_text/ (≥0.5 FPS)")
        print("  1. python main.py vlm_label          # Generate titles from videos")
        print("  2. python main.py train --epochs 25   # Train the model")
        print("  3. python main.py infer               # Run inference")
        print("  4. python main.py eval                # Evaluate results")
        return

    command = sys.argv[1]
    # Remove the command from argv so sub-scripts can parse their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "vlm_label":
        from generate_gt_vlm import main as vlm_main
        vlm_main()

    elif command == "train":
        from train import main as train_main
        train_main()

    elif command == "infer":
        from inference import main as infer_main
        infer_main()

    elif command == "eval":
        from evaluate import main as eval_main
        eval_main()

    elif command == "predict":
        from predict_custom import main as predict_main
        predict_main()

    elif command == "pseudo":
        from generate_pseudo_labels import main as pseudo_main
        pseudo_main()

    elif command == "test":
        from test_architecture import main as test_main
        test_main()

    elif command == "ocr_extract":
        from extract_ocr import main as ocr_main
        ocr_main()

    else:
        print(f"❌ Unknown command: {command}")
        print("   Available commands: vlm_label, train, infer, eval, predict, pseudo, ocr_extract, test")


if __name__ == "__main__":
    main()
