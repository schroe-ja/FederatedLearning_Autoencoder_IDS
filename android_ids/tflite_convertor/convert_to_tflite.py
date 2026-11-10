import os
import shutil

import tensorflow as tf

from tfltransfer import bases, heads, optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

INPUT_DIM = 11  # = INPUT_DIM in dataset_load_tpot.py

base = tf.keras.Sequential(
    [tf.keras.Input(shape=(INPUT_DIM,)), tf.keras.layers.Lambda(lambda x: x)]
)
base.compile(loss="mse", optimizer="sgd")
base.save("identity_model", save_format="tf")

# Head model = autoencoder (11 -> 48 -> 24 -> 6 -> 24 -> 48 -> 11).
head = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(6),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dense(INPUT_DIM),
    ]
)
head.compile(loss="mse", optimizer="sgd")

# generates a directory tflite_model with five tflite models.
# copy them to client/app/src/main/assets/model.
base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    INPUT_DIM, base_path, heads.KerasModelHead(head), optimizers.SGD(1e-2), train_batch_size=32
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
