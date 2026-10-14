"""Re-run notebook 2 with the ORIGINAL (2022) split style: a random split of face crops, not grouped by photo.

The 2022 notebook did this (one unseeded run, so it cannot be recreated exactly):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    model.fit(X_train, Y_train, epochs=20, validation_split=0.1)     # last 10% of the training crops
    model.evaluate(X_test, Y_test)                                   # final epoch-20 model

This script does the same with a fixed seed, for several seeds, and with the same model as notebook 2
(InceptionV3 cut at `mixed7`, frozen, plus Flatten / Dense 1024 / Dropout 0.3 / Dense 3). It also counts how
many test crops come from a photo that also contributed training crops, which is why this split is optimistic.

It needs data/X.npy, data/Y.npy and data/groups.npy (written by notebook 1). It saves no model weights.

    python scripts/random_split_rerun.py --seeds 42 43 44 --out-dir docs/runs
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CLASSES = ["without_mask", "with_mask", "mask_weared_incorrect"]


def build_model(tf):
    """The notebook 2 model: frozen InceptionV3 up to `mixed7` plus a small trainable head."""
    base = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights="imagenet", input_shape=(100, 100, 3))
    for layer in base.layers:
        layer.trainable = False
    x = tf.keras.layers.Flatten()(base.get_layer("mixed7").output)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base.input, outputs=x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


def run_seed(seed: int, X, Y, groups, epochs: int) -> dict:
    import keras
    import sklearn
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=seed)  # random, by crop
    X_train, Y_train, X_test, Y_test = X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]

    seen = set(groups[train_idx].tolist())
    shared = int(sum(g in seen for g in groups[test_idx].tolist()))

    model = build_model(tf)
    t0 = time.time()
    history = model.fit(X_train, Y_train, epochs=epochs, validation_split=0.1, verbose=2)
    train_seconds = time.time() - t0

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)  # final-epoch model, as in 2022
    y_true = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))
    correct = int((y_true == y_pred).sum())
    print(f"seed {seed}: test accuracy {test_acc:.4f} ({correct}/{len(y_true)}), {train_seconds:.0f} s training")
    return {
        "seed": seed,
        "split": "train_test_split(test_size=0.1, random_state=seed) by face crop; validation_split=0.1 (last 10% of train)",
        "sizes": {"train": int(len(train_idx) * 0.9), "val": int(len(train_idx)) - int(len(train_idx) * 0.9), "test": int(len(test_idx))},
        "test_crops_sharing_a_photo_with_training": shared,
        "test_crops_sharing_share": shared / len(test_idx),
        "epochs": epochs,
        "train_seconds": round(train_seconds, 1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "correct": correct,
        "test_crops": int(len(test_idx)),
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "versions": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs" / "runs")
    args = parser.parse_args()

    X = np.load(args.data_dir / "X.npy").astype("float32") / 255.0
    Y = np.load(args.data_dir / "Y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    print(X.shape, Y.shape, "distinct source photos:", len(set(groups.tolist())))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        result = run_seed(seed, X, Y, groups, args.epochs)
        results.append(result)
        (args.out_dir / f"2026-10-14-random-split-seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    accs = np.array([r["test_accuracy"] for r in results])
    print(f"seeds {args.seeds}: mean {accs.mean():.4f}, min {accs.min():.4f}, max {accs.max():.4f}")


if __name__ == "__main__":
    main()
