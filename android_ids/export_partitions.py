"""Exports the device T-Pot data for the Android app."""
import os
import sys


_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):
    if os.path.isdir(os.path.join(_p, "src")) and _p not in sys.path:
        sys.path.insert(0, _p)

from src.dataset_load_tpot import load_device_dataset, DEVICE_NAMES

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _here, "client", "app", "src", "main", "assets", "data"
)
print("writing to", out)
os.makedirs(out, exist_ok=True)
for i in range(len(DEVICE_NAMES)):
    trainloader, _, X_test_full, _, y_true, _, _ = load_device_dataset(i)
    X_train = trainloader.dataset.tensors[0].numpy()
    with open(os.path.join(out, f"partition_{i}_train.txt"), "w") as f:
        for row in X_train:
            f.write("0," + ",".join(f"{v:.6f}" for v in row) + "\n")
    with open(os.path.join(out, f"partition_{i}_test.txt"), "w") as f:
        for row, y in zip(X_test_full, y_true):
            f.write(f"{int(y)}," + ",".join(f"{v:.6f}" for v in row) + "\n")
    print(i, DEVICE_NAMES[i], len(X_train), len(X_test_full))
