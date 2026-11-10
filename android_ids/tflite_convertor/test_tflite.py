"""Checks the TFLite models on the PC (no Android, no server needed).

    python test_tflite.py          # all 10 devices, prints a summary table at the end
    python test_tflite.py <model_dir> <train_file> <test_file>   # explicit paths
"""
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "client", "app", "src", "main", "assets")
MODEL_DIR = os.path.join(ASSETS, "model")
DATA_DIR = os.path.join(ASSETS, "data")
NAMES = ["laptop", "mobile", "panel", "bridge", "hue", "camera", "lamp", "socket", "speaker", "home_assistant"]


def load(model_dir, name):
    it = tf.lite.Interpreter(model_path=os.path.join(model_dir, f"{name}.tflite"))
    it.allocate_tensors()
    return it


def run(it, inputs):
    for d, x in zip(it.get_input_details(), inputs):
        it.set_tensor(d["index"], np.asarray(x, dtype=np.float32).reshape(d["shape"]))
    it.invoke()
    return [it.get_tensor(d["index"]).copy() for d in it.get_output_details()]


def read(path):
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    y = np.array([int(float(r[0])) for r in rows])
    x = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)
    return y, x


def test_device(model_dir, train_file, test_file):
    init, train, opt, inf = (load(model_dir, n) for n in ("initialize", "train_head", "optimizer", "inference"))
    n_params = len(train.get_input_details()) - 2
    batch = train.get_input_details()[0]["shape"][0]
    n_opt_state = len(opt.get_input_details()) - 2 * n_params
    assert n_opt_state == 0, "optimizer has state (e.g. Adam) - the Java client cannot handle that, use SGD"
    print(f"parameter tensors: {n_params}, train batch size: {batch}")

    ytr, xtr = read(train_file)
    yte, xte = read(test_file)
    xb = xtr[ytr == 0]  # the autoencoder only trains on benign rows
    n_mal = int((yte == 1).sum())
    print(f"train rows: {len(xtr)} (benign {len(xb)}), test rows: {len(xte)} (malicious {n_mal})")
    assert len(xb) >= batch, "fewer benign rows than the batch size"

    params = run(init, [0.0])
    rng = np.random.default_rng(0)

    def errors(x):
        return np.array([float(np.mean((run(inf, [r] + params)[0].ravel() - r) ** 2)) for r in x])

    before = errors(xte[yte == 0]).mean()
    print(f"benign test MSE before training: {before:.5f}")
    for epoch in range(30):
        idx = rng.permutation(len(xb))
        losses = []
        for s in range(0, len(idx) - batch + 1, batch):
            b = xb[idx[s : s + batch]]
            out = run(train, [b, b] + params)
            losses.append(float(out[0]))
            params = run(opt, params + out[1:])
        if epoch % 5 == 0 or epoch == 29:
            print(f"epoch {epoch:2d}  loss {np.mean(losses):.5f}")

    e = errors(xte)
    thr = np.percentile(e[yte == 0], 85)
    acc = float(((e > thr) == (yte == 1)).mean())
    after = e[yte == 0].mean()
    mal = e[yte == 1].mean() if n_mal else float("nan")
    print(f"benign test MSE after training: {after:.5f}, threshold (85th pct): {thr:.5f}")
    if n_mal:
        print(f"malicious test MSE: {mal:.5f}")
    print(f"accuracy: {acc:.3f}")
    print("OK: model trains and evaluates in the TFLite format")
    return dict(train=len(xtr), test=len(xte), before=before, after=after, mal=mal, thr=thr, acc=acc,
                baseline=n_mal / len(xte))


def paths(i):
    return (os.path.join(DATA_DIR, f"partition_{i}_train.txt"), os.path.join(DATA_DIR, f"partition_{i}_test.txt"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3:
        test_device(*args)
    else:
        devices = [int(args[0])] if len(args) == 1 else range(10)
        results = {}
        for i in devices:
            print(f"\n=== Device {i} ({NAMES[i]}) ===")
            results[i] = test_device(MODEL_DIR, *paths(i))
        if len(results) > 1:
            print("\n=== Summary ===")
            print("device            train  test  benign_before  benign_after  malicious   threshold  accuracy  baseline")
            for i, r in results.items():
                print(f"{i} {NAMES[i]:<15} {r['train']:>6} {r['test']:>5} {r['before']:>14.4f} {r['after']:>13.4f} "
                      f"{r['mal']:>10.4f} {r['thr']:>10.4f} {r['acc']:>9.3f} {r['baseline']:>9.3f}")
