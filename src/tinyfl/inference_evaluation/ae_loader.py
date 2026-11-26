import torch
import numpy as np

from src.tinyfl.autoencoder import Autoencoder
from src.tinyfl import constants

model = Autoencoder(constants.INPUT_DIM)

model.load_state_dict(
    torch.load(
        "final_model.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


@torch.inference_mode()
def predict_ae(features, threshold):

    x = torch.as_tensor(
        features,
        dtype=torch.float32
    ).reshape(1, constants.INPUT_DIM)

    reconstructed = model(x)

    error = torch.mean(
        torch.square(x - reconstructed)
    ).item()

    return int(error > threshold), error


@torch.inference_mode()
def extract_dt_features(features):

    x = torch.as_tensor(
        features,
        dtype=torch.float32
    ).reshape(1, constants.INPUT_DIM)

    encoded = model.encoder(x).numpy()

    reconstructed = model(x)

    error = torch.mean(
        torch.square(x - reconstructed),
        dim=1
    ).numpy()

    stats = np.std(x.numpy(), axis=1)

    dt_features = np.column_stack([
        encoded * 10000,
        error * 10000,
        stats * 10000
    ])

    return np.clip(
        dt_features,
        -32768,
        32767
    ).astype(np.int16).reshape(14)