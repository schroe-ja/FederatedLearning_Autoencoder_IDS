import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["ae", "dt"],
    required=True
)

parser.add_argument("--dataset", required=True)
parser.add_argument("--threshold", type=float)
parser.add_argument("--quantization", type=float)

args = parser.parse_args()

if args.model == "ae":
    from ae_loader import predict_ae
else:
    from tree_loader import predict_tree


df = pd.read_csv(args.dataset)

df = df[df["label"].isin([0, 1])].copy()
df = df.sort_values("timestamp")

feature_columns = [
    f"feature_{i}" for i in range(16)
]

X = df[feature_columns].to_numpy(
    dtype=np.float32
)

y_true = df["label"].to_numpy(dtype=int)

predictions = []
scores = []
inference_times = []

if args.model == "dt":
    X = np.clip(
        X * args.quantization,
        -32768,
        32767
    ).astype(np.int16)

start_total = time.perf_counter()

for features in X:

    start = time.perf_counter_ns()

    if args.model == "ae":
        prediction, score = predict_ae(
            features,
            args.threshold
        )
    else:
        prediction = predict_tree(features)
        score = prediction

    end = time.perf_counter_ns()

    predictions.append(prediction)
    scores.append(score)

    inference_times.append(
        (end - start) / 1_000_000
    )

total_time = time.perf_counter() - start_total

y_pred = np.asarray(predictions)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

metrics = {
    "model": args.model,
    "samples": len(y_true),
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(
        y_true, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_true, y_pred, zero_division=0
    ),
    "f1": f1_score(
        y_true, y_pred, zero_division=0
    ),
    "MCC": matthews_corrcoef(y_true, y_pred),
    "FPR": fp / (fp + tn) if fp + tn else np.nan,
    "FNR": fn / (fn + tp) if fn + tp else np.nan,
    "mean_inference_ms": np.mean(inference_times),
    "median_inference_ms": np.median(inference_times),
    "p95_inference_ms": np.percentile(
        inference_times, 95
    ),
    "total_time_s": total_time,
    "samples_per_second": len(y_true) / total_time
}

df["prediction"] = predictions
df["score"] = scores
df["inference_time_ms"] = inference_times

Path("results").mkdir(exist_ok=True)

df.to_csv(
    f"results/{args.model}_predictions.csv",
    index=False
)

pd.DataFrame([metrics]).to_csv(
    f"results/{args.model}_metrics.csv",
    index=False
)

print(pd.Series(metrics))