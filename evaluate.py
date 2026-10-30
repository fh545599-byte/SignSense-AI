"""Evaluate the trained SignSense AI model on the official GTSRB test set.

Usage:
    python evaluate.py --data path/to/gtsrb

The data folder must contain Test.csv and the Test/ image folder
(download from the Kaggle link in README.md).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).parent
IMG_SIZE = 32


def main(data_dir: str) -> None:
    data = Path(data_dir)
    model = tf.keras.models.load_model(BASE / "model" / "signsense_model.keras")
    with open(BASE / "model" / "class_names.json") as f:
        names = {int(k): v for k, v in json.load(f).items()}

    df = pd.read_csv(data / "Test.csv")
    X = np.array(
        [
            np.array(Image.open(data / p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
            for p in df["Path"]
        ],
        dtype="float32",
    )
    y_true = df["ClassId"].values

    y_pred = np.argmax(model.predict(X, batch_size=128, verbose=0), axis=1)

    print(f"Test images : {len(y_true)}")
    print(f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Wrongly rejected as 'not a sign': {(y_pred == 43).mean():.4f}\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(43)),
            target_names=[names[i] for i in range(43)],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gtsrb", help="Folder with Test.csv and Test/")
    main(parser.parse_args().data)