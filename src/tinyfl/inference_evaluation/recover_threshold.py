# Script to calculate the inference threshold
from pathlib import Path

import numpy as np
import torch

from src.tinyfl import constants
from src.tinyfl.autoencoder import Autoencoder
from src.tinyfl.sota.dataset_load_tpot import DatasetLoad


model = Autoencoder(constants.INPUT_DIM)

model.load_state_dict(
    torch.load(
        "/path/to/final_model.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

loader = DatasetLoad()

_, _, _, _, X_Validation, _, _, _ = (
    loader.load_centralized_dataset()
)

with torch.no_grad():

    X_val = torch.as_tensor(
        X_Validation,
        dtype=torch.float32
    )

    reconstructed = model(X_val)

    errors_val = torch.mean(
        torch.square(X_val - reconstructed),
        dim=1
    ).cpu().numpy()

threshold = float(np.percentile(errors_val, 95))

Path("ae_threshold.txt").write_text(
    f"{threshold}\n"
)

print(f"AE threshold: {threshold}")
