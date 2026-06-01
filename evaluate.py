"""
PRISM - Evaluate model predictions against ground truth.
Computes exact match accuracy and per-video comparison.

Usage:
    python evaluate.py --results outputs/results/predictions.csv
"""
import os
import sys
import argparse
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR


def compute_exact_match(gt: str, pred: str) -> bool:
    """Check if prediction exactly matches ground truth (case-insensitive)."""
    return gt.strip().lower() == pred.strip().lower()


def compute_word_overlap(gt: str, pred: str) -> float:
    """Compute word overlap (F1-like) between ground truth and prediction."""
    gt_words = set(gt.strip().lower().split())
    pred_words = set(pred.strip().lower().split())

    if len(gt_words) == 0 and len(pred_words) == 0:
        return 1.0
    if len(gt_words) == 0 or len(pred_words) == 0:
        return 0.0

    common = gt_words & pred_words
    precision = len(common) / len(pred_words) if pred_words else 0
    recall = len(common) / len(gt_words) if gt_words else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_repetitive(text: str, threshold: float = 0.6) -> bool:
    """Check if generated text is repetitive (e.g., 'car car car car...')."""
    words = text.strip().lower().split()
    if len(words) <= 2:
        return False
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(words) > threshold


def evaluate(results_path: str):
    """Run evaluation on prediction results."""
    df = pd.read_csv(results_path)
    df["prediction"] = df["prediction"].fillna("")

    total = len(df)
    exact_matches = 0
    partial_scores = []
    empty_preds = 0
    repetitive_preds = 0

    print("=" * 80)
    print("PRISM EVALUATION REPORT")
    print("=" * 80)

    for _, row in df.iterrows():
        gt = str(row["ground_truth"])
        pred = str(row["prediction"])

        if compute_exact_match(gt, pred):
            exact_matches += 1

        partial_scores.append(compute_word_overlap(gt, pred))

        if len(pred.strip()) == 0:
            empty_preds += 1

        if is_repetitive(pred):
            repetitive_preds += 1

    # Summary
    print(f"\n📊 Results Summary:")
    print(f"   Total videos:       {total}")
    print(f"   Exact matches:      {exact_matches} ({exact_matches/total*100:.1f}%)")
    print(f"   Avg word overlap:   {sum(partial_scores)/len(partial_scores):.3f}")
    print(f"   Empty predictions:  {empty_preds} ({empty_preds/total*100:.1f}%)")
    print(f"   Repetitive preds:   {repetitive_preds} ({repetitive_preds/total*100:.1f}%)")

    # Distribution
    perfect = sum(1 for s in partial_scores if s >= 1.0)
    good = sum(1 for s in partial_scores if 0.5 <= s < 1.0)
    partial = sum(1 for s in partial_scores if 0.0 < s < 0.5)
    zero = sum(1 for s in partial_scores if s == 0.0)

    print(f"\n📈 Score Distribution:")
    print(f"   Perfect (F1=1.0):   {perfect} ({perfect/total*100:.1f}%)")
    print(f"   Good (F1≥0.5):      {good} ({good/total*100:.1f}%)")
    print(f"   Partial (0<F1<0.5): {partial} ({partial/total*100:.1f}%)")
    print(f"   Zero (F1=0.0):      {zero} ({zero/total*100:.1f}%)")

    print("\n" + "=" * 80)

    # Show worst predictions
    print("\n❌ Worst Predictions (Repetitive or Empty):")
    for _, row in df.iterrows():
        pred = str(row["prediction"])
        if is_repetitive(pred) or len(pred.strip()) == 0:
            print(f"   [{row.get('video_id', '?')}] GT: {row['ground_truth']}")
            print(f"   {'':>12} PRED: {pred if pred.strip() else '(empty)'}")
            print()

    return {
        "total": total,
        "exact_match": exact_matches,
        "exact_match_pct": exact_matches / total * 100,
        "avg_word_overlap": sum(partial_scores) / len(partial_scores),
        "empty_preds": empty_preds,
        "repetitive_preds": repetitive_preds,
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Evaluation")
    parser.add_argument(
        "--results", type=str,
        default=str(RESULTS_DIR / "predictions.csv"),
        help="Path to predictions CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"❌ Results file not found: {args.results}")
        print("   Run inference first: python inference.py")
        return

    evaluate(args.results)


if __name__ == "__main__":
    main()
