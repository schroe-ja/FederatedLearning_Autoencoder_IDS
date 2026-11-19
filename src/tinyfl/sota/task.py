"""if-and-auto: A Flower / PyTorch app."""

import os
import time
import copy
import psutil
import copy

import emlearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.tinyfl.sota.dataset_load_tpot import DatasetLoad
from src.tinyfl import constants
from src.tinyfl.evaluation_metrics import EvaluationMetrics
from src.tinyfl.autoencoder import Autoencoder, phase_recall, malicious_probability, train

ld = DatasetLoad()

def test(net, X_benign_test, X_malicious_test, X_iotpot_test, X_Validation, partition_id, device, X_train_dt, y_dt): 
    """Validate the model on the test set."""

    EvaluationMetrics.print_memory_usage()
    net.eval()
    print("testing device", partition_id)
    net.to(device)

    with torch.no_grad():
         
        X_train_dt = torch.FloatTensor(X_train_dt).to(device) 
        X_benign_test = torch.FloatTensor(X_benign_test).to(device)
        X_malicious_test = torch.FloatTensor(X_malicious_test).to(device)
        X_iotpot_test = torch.as_tensor(X_iotpot_test, dtype=torch.float32, device=device)
        
        #encoding layer for classifier 
        X_train_encoded = net.encoder(X_train_dt).detach().cpu().numpy() 
        X_benign_encoded = net.encoder(X_benign_test).numpy()
        X_malicious_encoded = net.encoder(X_malicious_test).numpy()
        
        # Validation reconstruction errors
        X_val_tensor = torch.FloatTensor(X_Validation).to(device)
        recon_val = net(X_val_tensor)
        errors_val = torch.mean(torch.square(X_val_tensor - recon_val), dim=1).cpu().numpy()

        # reconstruction error for classifier
        recon_dt = net(X_train_dt)
        errors_dt = torch.mean(torch.square(X_train_dt - recon_dt), dim=1).cpu().numpy() 

        dt_stats = np.std(X_train_dt.cpu().numpy(), axis=1) 
        
        # reconstruction error of full Test set
        recon_benign = net(X_benign_test)
        errors_benign = torch.mean(
            torch.square(X_benign_test - recon_benign), dim=1
        ).cpu().numpy()
        stats_benign = np.std(X_benign_test.cpu().numpy(), axis=1)

        recon_malicious = net(X_malicious_test)
        errors_malicious = torch.mean(
            torch.square(X_malicious_test - recon_malicious), dim=1
        ).cpu().numpy()
        stats_malicious = np.std(X_malicious_test.cpu().numpy(), axis=1)

        recon_iotpot = net(X_iotpot_test)
        errors_iotpot = torch.mean(
            torch.square(X_iotpot_test - recon_iotpot), dim=1
        ).cpu().numpy()
        stats_iotpot = X_iotpot_test.std(
            dim=1,
            correction=0
        ).cpu().numpy()

        X_iotpot_encoded = net.encoder(X_iotpot_test).detach().cpu().numpy()
        
        mu_val = np.mean(errors_val) 
        sigma_val = np.std(errors_val) + 1e-8  

    # For training (used in classifier). Z-score normalisation
    errors_dt_norm = errors_dt 

    # For testing (used in classifier). Z-score normalisation
    errors_benign_norm = errors_benign  
    errors_malicious_norm = errors_malicious  

    X_features = np.column_stack([ 
    X_train_encoded * 10000,  # AE latent * 10000 because int16 would make 0.006 to 0
    errors_dt_norm * 10000,  # Recon error * 10000 because int16 would make 0.006 to 0
    dt_stats * 10000  # Flow stats * 10000 because int16 would make 0.006 to 0
    ])  # 8D total with 67 architecture, new architecture 8d since bottleneck is 6
    
    # DecisionTree (DT), alternative to unsupervised Thresholding,small for Tinyml deployment
    dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced')  # 16 leaves max  
    dt_tinyml = DecisionTreeClassifier(max_depth=4, class_weight='balanced')  # 16 leaves max  ; This is for exporting to MCU

    X_features_benign = np.column_stack([
        X_benign_encoded * 10000,
        errors_benign_norm * 10000,
        stats_benign * 10000
    ])  

    X_features_malicious = np.column_stack([
        X_malicious_encoded * 10000,
        errors_malicious_norm * 10000,
        stats_malicious * 10000
    ])  

    X_features_iotpot = np.column_stack([
        X_iotpot_encoded * 10000,
        errors_iotpot * 10000,
        stats_iotpot * 10000,
    ])
    
    # DT fit and predict
    dt.fit(X_features, y_dt)  
    y_pred_benign = dt.predict(X_features_benign)  
    y_proba_benign = malicious_probability(dt, X_features_benign)  
    y_pred_malicious = dt.predict(X_features_malicious)  
    y_proba_malicious = malicious_probability(dt, X_features_malicious)  

    # Unsupervised Thresholding, Percentile
    threshold_percentile = np.percentile(errors_val, 95)
    print("Threshold percentile 95")

    y_pred_benign_percentile = (errors_benign > threshold_percentile).astype(int)
    y_pred_malicious_percentile = (errors_malicious > threshold_percentile).astype(int)
    y_pred_iotpot_percentile = (errors_iotpot > threshold_percentile).astype(int)
    y_pred_iotpot = dt.predict(X_features_iotpot)
    y_proba_iotpot = malicious_probability(dt, X_features_iotpot)

    EvaluationMetrics.print_memory_usage()

    X_train_int16 = np.clip(X_features, -32768, 32767).astype(np.int16) 
    dt_tinyml.fit(X_train_int16, y_dt) 

    return (
        threshold_percentile,
        errors_val,
        errors_benign,
        errors_malicious,
        errors_iotpot,
        y_pred_benign_percentile,
        y_pred_malicious_percentile,
        y_pred_iotpot_percentile,
        y_pred_benign,
        y_proba_benign,
        y_pred_malicious,
        y_proba_malicious,
        y_pred_iotpot,
        y_proba_iotpot,
        dt,
        dt_tinyml
    )

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    start_time = time.time()

    # Load the model and initialize it with the received weights
    model = Autoencoder(constants.INPUT_DIM)

    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cpu")
    model.to(device)

    _, _, X_benign_test, X_malicious_test, X_Validation, X_train_dt, y_dt, phase_malicious_test = ld.load_centralized_dataset()

    X_iotpot_test = ld.load_binary_info_test(constants.IOTPOT_BINARY_INFO_FILE)
    y_iotpot_test = np.ones(len(X_iotpot_test), dtype=np.int64)

    # Evaluate the global model on the test set
    (
        threshold_ae,
        errors_val,
        errors_benign,
        errors_malicious,
        errors_iotpot,
        y_pred_benign_percentile,
        y_pred_malicious_percentile,
        y_pred_iotpot_percentile,
        y_pred_benign,
        y_proba_benign,
        y_pred_malicious,
        y_proba_malicious,
        y_pred_iotpot,
        y_proba_iotpot,
        dt,
        dt_tinyml
    ) = test(
        model,
        X_benign_test,
        X_malicious_test,
        X_iotpot_test,
        X_Validation,
        1000,
        device,
        X_train_dt,
        y_dt
    )

    y_true = np.concatenate([
        np.zeros(len(X_benign_test), dtype=int),
        np.ones(len(X_malicious_test), dtype=int),
    ])

    # Map test() outputs to the names used by the metric code
    y_pred_benign_ae = y_pred_benign_percentile
    y_pred_malicious_ae = y_pred_malicious_percentile

    y_pred_benign_dt = y_pred_benign
    y_pred_malicious_dt = y_pred_malicious

    y_proba_benign_dt = y_proba_benign
    y_proba_malicious_dt = y_proba_malicious

    phase_results_ae = phase_recall(
        y_pred_malicious_ae,
        phase_malicious_test,
        "AE"
    )

    phase_results_dt = phase_recall(
        y_pred_malicious_dt,
        phase_malicious_test,
        "AE + Decision Tree"
    )

    # Flatten arrays so metrics receive one-dimensional inputs
    errors_benign = np.asarray(errors_benign).ravel()
    errors_malicious = np.asarray(errors_malicious).ravel()

    y_pred_benign_ae = np.asarray(y_pred_benign_ae).ravel().astype(int)
    y_pred_malicious_ae = np.asarray(y_pred_malicious_ae).ravel().astype(int)
    y_pred_benign_dt = np.asarray(y_pred_benign_dt).ravel().astype(int)
    y_pred_malicious_dt = np.asarray(y_pred_malicious_dt).ravel().astype(int)

    y_proba_benign_dt = np.asarray(y_proba_benign_dt).ravel()
    y_proba_malicious_dt = np.asarray(y_proba_malicious_dt).ravel()

    # Combine benign first, then malicious, matching y_true and the masks
    y_pred_percentile = np.concatenate([
        y_pred_benign_ae,
        y_pred_malicious_ae,
    ])

    y_pred = np.concatenate([
        y_pred_benign_dt,
        y_pred_malicious_dt,
    ])

    y_proba = np.concatenate([
        y_proba_benign_dt,
        y_proba_malicious_dt,
    ])

    errors_full = np.concatenate([
        errors_benign,
        errors_malicious,
    ])

    benign_mask = (y_true == 0)
    malicious_mask = (y_true == 1)

    if not benign_mask.any() or not malicious_mask.any():
        raise ValueError("The evaluation set must contain bothc benign (0) and malicious (1) samples.")

    # Assumes prediction 1 means malicious/flagged.
    benign_fp_ae = np.mean(y_pred_percentile[benign_mask] == 1)
    malicious_recall_ae = np.mean(y_pred_percentile[malicious_mask] == 1)

    benign_fp_dt = np.mean(y_pred[benign_mask] == 1)
    malicious_recall_dt = np.mean(y_pred[malicious_mask] == 1)

    # IoTPOT contains only malicious samples, so report detection rate/recall.
    y_iotpot_true = np.ones(len(X_iotpot_test), dtype=int)

    y_pred_iotpot_ae = np.asarray(y_pred_iotpot_percentile).ravel().astype(int)
    y_pred_iotpot_dt = np.asarray(y_pred_iotpot).ravel().astype(int)

    iotpot_recall_ae = np.mean(y_pred_iotpot_ae == 1)
    iotpot_recall_dt = np.mean(y_pred_iotpot_dt == 1)

    print(
        f"Benign test: {np.sum(y_pred_percentile[benign_mask] == 1)} / "
        f"{benign_mask.sum()} flagged "
        f"({benign_fp_ae:.2%} false-positive rate, AE)"
    )
    print(
        f"Malicious test: {np.sum(y_pred_percentile[malicious_mask] == 1)} / "
        f"{malicious_mask.sum()} detected "
        f"({malicious_recall_ae:.2%} recall, AE)"
    )
    print(
        f"Benign test: {np.sum(y_pred[benign_mask] == 1)} / "
        f"{benign_mask.sum()} flagged "
        f"({benign_fp_dt:.2%} false-positive rate, DT)"
    )
    print(
        f"Malicious test: {np.sum(y_pred[malicious_mask] == 1)} / "
        f"{malicious_mask.sum()} detected "
        f"({malicious_recall_dt:.2%} recall, DT)"
    )
    print(
        f"IoTPOT malicious test: {y_pred_iotpot_ae.sum()} / "
        f"{len(y_iotpot_true)} detected "
        f"({iotpot_recall_ae:.2%} recall, AE)"
    )
    print(
        f"IoTPOT malicious test: {y_pred_iotpot_dt.sum()} / "
        f"{len(y_iotpot_true)} detected "
        f"({iotpot_recall_dt:.2%} recall, DT)"
    )
    
    c_model = emlearn.convert(dt_tinyml, method='inline')
    c_model.save(file='tree_model.h', name='my_model') 

    end_time = time.time()

    ids_eval = EvaluationMetrics(
        y_true, y_pred, y_pred_percentile, y_proba, errors_full, start_time, end_time
    )
    metrics = ids_eval.metrics()

    metrics["IoTPOT Recall AE"] = float(f"{iotpot_recall_ae:.4f}")
    metrics["IoTPOT Recall DT"] = float(f"{iotpot_recall_dt:.4f}")
    metrics["IoTPOT Detected AE"] = int(y_pred_iotpot_ae.sum())
    metrics["IoTPOT Detected DT"] = int(y_pred_iotpot_dt.sum())
    metrics["IoTPOT Samples"] = int(len(y_iotpot_true))

    metrics["Scan Recall AE"] = phase_results_ae.get("scan", 0.0)
    metrics["Scan Recall DT"] = phase_results_dt.get("scan", 0.0)

    metrics["Distribution Recall AE"] = phase_results_ae.get("distribution", 0.0)
    metrics["Distribution Recall DT"] = phase_results_dt.get("distribution", 0.0)

    return MetricRecord(metrics)
