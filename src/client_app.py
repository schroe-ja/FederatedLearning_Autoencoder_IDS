"""if-and-auto: A Flower / PyTorch app."""

import time

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve,auc, average_precision_score, classification_report, precision_score

from src.task import Autoencoder
from src.task import test as test_fn
from src.task import train as train_fn

# from src.dataset_load import load_cross_data, load_mono_dataset # original IoTID20/CiC-BoTIoT
from src.dataset_load_tpot import load_device_dataset, DEVICE_NAMES # T-Pot dataset, one client per IoT device

torch.use_deterministic_algorithms(True)

# Flower ClientApp
app = ClientApp()

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
    partition_id = context.node_config["partition-id"]
    partition = partition_id
    #which_dataset: int = context.run_config["which_dataset"]

    #trainloader, validaton_loader, _ ,_, _,_,_ = load_mono_dataset(partition, num_partitions,which_dataset=which_dataset)
    trainloader, validaton_loader, _, _, _, _, _ = load_device_dataset(partition)

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
        "num-examples": len(trainloader),
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
    partition = partition_id
    #which_dataset: int = context.run_config["which_dataset"]

    #_,_,X_test_full, X_test_validation, y_true,X_train_dt,y_dt = load_mono_dataset(partition, num_partitions,which_dataset=which_dataset)
    #_,_,X_test_full, X_test_validation, y_true,X_train_dt,y_dt = load_cross_data(partition, num_partitions, which_dataset=which_dataset)
    _, _, X_test_full, X_test_validation, y_true, X_train_dt, y_dt = load_device_dataset(partition)

    # Call the evaluation function
    threshold, y_pred_percentile, errors_full, errors_val, y_pred, y_proba, _, _ = test_fn(
        model, # Autoencoder
        X_test_full, X_test_validation, # Data for Testing
        partition, # Partition ID
        device,
        X_train_dt,y_dt # Data for Decision Tree; remove this for no dt
    )

    # Metrics for Client Eval
    # Confusion matrix
    cmpercentile = confusion_matrix(y_true, y_pred_percentile)
    cm_dt = confusion_matrix(y_true,y_pred) #remove this for no dt

    print(f"\nConfusion Matrix Classify and Device {partition} ({DEVICE_NAMES[partition]}):")
    print(cm_dt) #remove this for no dt

    # Use Confusion Matrix to calculate fnr and fpr for Percentile and DT
    fprpercentile=cmpercentile[0][1]/ (cmpercentile[0][0]+cmpercentile[0][1])
    fnrpercentile=cmpercentile[1][0]/ (cmpercentile[1][1]+cmpercentile[1][0])

    fprdt=cm_dt[0][1] / (cm_dt[0][0]+cm_dt[0][1]) #remove this for no dt
    fnrdt=cm_dt[1][0] / (cm_dt[1][1]+cm_dt[1][0]) #remove this for no dt

    #  ROC AUC for Percentile and DT
    b = roc_auc_score(y_true, y_proba) #remove this for no dt
    c = roc_auc_score(y_true,errors_full)

    a = average_precision_score(y_true, y_proba)
    d = average_precision_score(y_true, errors_full)
    pr_precision = precision_score(y_true, y_pred, zero_division=0)

    print(f"Device {partition} ({DEVICE_NAMES[partition]}) ROC AUC DT",b, "ROC AUC Error", c)
    print(f"Device {partition} ({DEVICE_NAMES[partition]}) Fprpercentile",float("{:.2f}".format(fprpercentile)),"Fnrpercentile",float("{:.2f}".format(fnrpercentile)),"FPR DT",
          float("{:.2f}".format(fprdt)),"FNR DT",float("{:.2f}".format(fnrdt))) #remove this for no dt

    end_time = time.time()
    testing_time = end_time - start_time
    
    # Construct and return reply Message
    # Return the evaluation metrics
    metrics = {
        "threshold": float(threshold),
        "fprpercentile": float("{:.4f}".format(fprpercentile)),
        "fnrpercentile": float("{:.4f}".format(fnrpercentile)),
        #"FPR DT": float("{:.4f}".format(fprdt)), #remove this for no dt
        #"FNR DT": float("{:.4f}".format(fnrdt)), #remove this for no dt
        "PR AUC": float("{:.4f}".format(a)), 
        #"ROC AUC DT": float("{:.4f}".format(b)), #remove this for no dt
        "ROC AUC Error": float("{:.4f}".format(c)),
        "PR AUC Error": float("{:.4f}".format(d)),
        "Precision":float(("{:.4f}".format(pr_precision))),
        "mttd": float("{:.4f}".format(testing_time)),  # New metric
        "num-examples": len(y_true),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
