"""Export mobile and panel datasets for the Android clients."""

import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))

for _p in (_here, os.path.dirname(_here)):
    if os.path.isdir(os.path.join(_p, "src")) and _p not in sys.path:
        sys.path.insert(0, _p)

from src.tinyfl.sota.dataset_load_tpot import DatasetLoad
from src.tinyfl.constants import DEVICE_NAMES

ld = DatasetLoad()

out = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(
        _here, "client", "app", "src", "main", "assets", "data"
    )
)

os.makedirs(out, exist_ok=True)


def export_loader(loader, path, require_benign=False):
    count = 0

    with open(path, "w") as f:
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                X = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                X = batch
                y = None

            X = X.detach().cpu().numpy()

            if y is None:
                labels = np.zeros(len(X), dtype=int)
            else:
                labels = y.detach().cpu().numpy().reshape(-1)

            if len(labels) != len(X):
                raise ValueError("Feature/label count mismatch")

            for row, label in zip(X, labels):
                label = int(label)

                if label not in (0, 1):
                    raise ValueError(f"Invalid label: {label}")

                if require_benign and label != 0:
                    raise ValueError(
                        "Autoencoder training data must be benign-only"
                    )

                f.write(
                    f"{label}," +
                    ",".join(f"{float(v):.6f}" for v in row) +
                    "\n"
                )

                count += 1

    return count


for device_name in ("mobile", "panel"):
    i = DEVICE_NAMES.index(device_name)

    trainloader, validationloader = ld.load_device_dataset(device_name)

    train_path = os.path.join(
        out, f"partition_{device_name}_train.txt"
    )

    val_path = os.path.join(
        out, f"partition_{device_name}_val.txt"
    )

    n_train = export_loader(
        trainloader, train_path, require_benign=True
    )

    n_val = export_loader(
        validationloader, val_path, require_benign=False
    )

    print(
        f"{device_name}: partition={i}, "
        f"train={n_train}, validation={n_val}"
    )