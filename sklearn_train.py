"""
sklearn_train.py
----------------
Simple scikit-learn training example designed to run inside the
ghcr.io/paulgarias/python-env-crispie container.

What this script does:
  1. Loads the Iris dataset (built-in, no external data needed).
  2. Splits into training and test sets (80/20, stratified).
  3. Builds a pipeline: StandardScaler -> RandomForestClassifier.
  4. Performs 5-fold cross-validation and reports per-fold accuracy.
  5. Trains a final model on the full training set.
  6. Evaluates on the held-out test set and prints a classification report.
  7. Saves the trained model to results/iris_rf_model.joblib.
  8. Saves a confusion-matrix plot to results/confusion_matrix.png.

Usage (see README.md for full container instructions):
  python sklearn_train.py [--output-dir RESULTS_DIR]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe inside containers
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Random Forest on Iris data.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where model and plots will be saved (default: results).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    print("=" * 60)
    print("Dataset:   Iris")
    print(f"Samples:   {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"Classes:   {list(iris.target_names)}")
    print("=" * 60)

    # ── 2. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.random_state
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test  set: {len(X_test)} samples")

    # ── 3. Pipeline ───────────────────────────────────────────────────────
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ── 4. Cross-validation ───────────────────────────────────────────────
    print("\n--- 5-Fold Cross-Validation (training set) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    for fold_idx, score in enumerate(cv_scores, start=1):
        print(f"  Fold {fold_idx}: {score:.4f}")
    print(f"\n  Mean accuracy : {cv_scores.mean():.4f}")
    print(f"  Std  accuracy : {cv_scores.std():.4f}")

    # ── 5. Final training ─────────────────────────────────────────────────
    print("\n--- Training final model on full training set ---")
    pipeline.fit(X_train, y_train)
    print("  Done.")

    # ── 6. Test-set evaluation ────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    test_acc = (y_pred == y_test.values).mean()

    print(f"\n--- Test-Set Accuracy: {test_acc:.4f} ---\n")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # ── 7. Save model ─────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "iris_rf_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved  →  {model_path}")

    # ── 8. Confusion matrix plot ──────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Iris Random Forest", pad=12)
    fig.tight_layout()

    plot_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved   →  {plot_path}")

    print("\n" + "=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()