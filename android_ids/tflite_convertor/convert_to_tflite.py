import os
import shutil
import sys

import numpy as np
import tensorflow as tf

from tfltransfer import bases, heads, optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)
from src.tinyfl import constants

INPUT_DIM = 16  # = INPUT_DIM in dataset_load_tpot.py

feature_weights = np.ones(INPUT_DIM, dtype=np.float32)

idx = {
    name: i for i, name in enumerate(constants.FEATURES)
}

important_features = [
    "url",
    "command_hash",
    "username_hash",
    "protocol_hash",
    "successful",
]

for feature in important_features:
    feature_weights[idx[feature]] = 2.0

for c in constants.BINFO_NUMERIC_COLS:
    feature_weights[idx[f"binfo_{c}"]] = 2.0

for c in constants.BINFO_TEXT_COLS:
    feature_weights[idx[f"binfo_{c}"]] = 2.0

def weighted_mse(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:

    feature_weights_tf = tf.constant(
        feature_weights,
        dtype=tf.float32,
    )

    loss = tf.square(y_pred - y_true)

    return tf.reduce_mean(loss * feature_weights_tf)

base = tf.keras.Sequential(
    [tf.keras.Input(shape=(INPUT_DIM,)), tf.keras.layers.Lambda(lambda x: x)]
)
base.compile(loss="mse", optimizer="sgd")
base.export("identity_model")

head = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(INPUT_DIM,)),

        # Encoder
        tf.keras.layers.Dense(64),
        tf.keras.layers.LayerNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(32),
        tf.keras.layers.LayerNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(12),

        # Decoder
        tf.keras.layers.Dense(32),
        tf.keras.layers.LayerNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(64),
        tf.keras.layers.LayerNormalization(epsilon=1e-5),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(INPUT_DIM),
    ]
)

head.compile(
    loss=weighted_mse,
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-4,
    ),
)

# generates a directory tflite_model with five tflite models.
# copy them to client/app/src/main/assets/model.
base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    INPUT_DIM, base_path, heads.KerasModelHead(head), optimizers.Adam(5e-4), train_batch_size=32
)
converter.convert_and_save("tflite_model")

assets_model = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "client", "app", "src", "main", "assets", "model"
)
os.makedirs(assets_model, exist_ok=True)
for name in os.listdir("tflite_model"):
    if name.endswith(".tflite"):
        shutil.copy(os.path.join("tflite_model", name), os.path.join(assets_model, name))
print("copied models to", os.path.normpath(assets_model))
