"""if-and-auto: A Flower / PyTorch app."""

import time
import numpy as np

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve,auc, average_precision_score, classification_report, precision_score

from src.tinyfl.autoencoder import Autoencoder
from src.tinyfl.sota.task import test as test_fn
from src.tinyfl.autoencoder import train as train_fn
from src.tinyfl import constants
from src.tinyfl.sota.dataset_load_tpot import DatasetLoad # T-Pot dataset, one client per IoT device

torch.use_deterministic_algorithms(True)

# Flower ClientApp
app = ClientApp()
ld = DatasetLoad()

def phase_recall(y_pred_malicious, phase_malicious_test, model_name):
    y_pred = np.asarray(y_pred_malicious).ravel()
    phases = np.asarray(phase_malicious_test).ravel()

    if len(y_pred) != len(phases):
        raise ValueError("Predictions and phase labels have different lengths")

    results = {}

    for phase in ["scan", "distribution"]:
        mask = phases == phase

        if not np.any(mask):
            print(f"{model_name} - {phase}: no test samples")
            continue

        recall = float(np.mean(y_pred[mask] == 1))
        results[phase] = recall

        print(
            f"{model_name} - {phase} detection rate: "
            f"{recall:.2%} "
            f"({np.sum(y_pred[mask] == 1)}/{np.sum(mask)})"
        )

    return results

@staticmethod
def select_client_groups(df, device_name, device_names, group_column):
    if group_column not in df.columns:
        raise ValueError(
            f"Missing grouping column: {group_column}"
        )

    client_index = device_names.index(device_name)
    num_clients = len(device_names)

    def assigned_client(value):
        key = str(value).encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") % num_clients

    assignments = df[group_column].map(assigned_client)

    return df.loc[
        assignments == client_index
    ].copy()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    start_time = time.time()

    # Load the model and initialize it with the received weights
    input_dim: int = context.run_config["input-dim"]
    model = Autoencoder(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cpu")
    model.to(device)

    # Load the data, one client per IoT device
    device_name = str(context.node_config["which_dataset"])
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    partition = partition_id
    #which_dataset: int = context.run_config["which_dataset"]

    #trainloader, validaton_loader, _ ,_, _,_,_ = load_mono_dataset(partition, num_partitions,which_dataset=which_dataset)
    trainloader, validaton_loader = ld.load_device_dataset(
        device_name
    )

    # Call the training function
    train_loss, val_loss = train_fn(
        model,
        trainloader,
        validaton_loader,
        partition,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"], #learning rate
        context.run_config["mu"], #mu for Fedprox
        device,
    )
    end_time = time.time()
    training_time = end_time - start_time

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "num-examples": len(trainloader.dataset),
        "avg_training_time": float("{:.2f}".format(training_time)),  # New metric
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record,})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    start_time = time.time()

    # Load the model and initialize it with the received weights
    input_dim: int = context.run_config["input-dim"]
    model = Autoencoder(input_dim)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cpu")
    model.to(device)

    # Load the data, one client per IoT device
    partition_id = context.node_config["partition-id"]
    device_name = str(context.node_config["which_dataset"])
    partition = partition_id
    #which_dataset: int = context.run_config["which_dataset"]

    trainloader, validaton_loader = ld.load_device_dataset(
        device_name
    )

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in validaton_loader:
            x = batch[0].to(device)

            reconstructed = model(x)

            # Reconstruction error per sample
            errors = torch.mean(
                (reconstructed - x) ** 2,
                dim=1
            )

            total_loss += errors.sum().item()
            total_examples += x.size(0)

    if total_examples == 0:
        raise ValueError(
            f"No validation samples for device {device_name}"
        )

    val_loss = total_loss / total_examples

    testing_time = time.time() - start_time

    print(
        f"Device {device_name} | "
        f"Validation MSE: {val_loss:.6f} | "
        f"Evaluation time: {testing_time:.2f}s"
    )

    metrics = {
        "val_loss": float(val_loss),
        "num-examples": int(total_examples),
        "evaluation_time": float(testing_time),
    }

    return Message(
        content=RecordDict({
            "metrics": MetricRecord(metrics)
        }),
        reply_to=msg,
    )
