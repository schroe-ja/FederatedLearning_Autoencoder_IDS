"""if-and-auto: A Flower / PyTorch app."""

import os
import time
import copy
import psutil

import emlearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score,roc_auc_score, average_precision_score, matthews_corrcoef,accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# from src.dataset_load import load_centralized_dataset,load_crossdataset, INPUT-DIM # original IoTID20/CiC-BoTIoT
from src.dataset_load_tpot import load_centralized_dataset, INPUT_DIM # T-Pot dataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim,dropout_rate=0.4): 
        super(Autoencoder, self).__init__()
        # Encoder: Compresses data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.BatchNorm1d(48),      # Helps training stability
            nn.ReLU(),      
            
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(24, 6)         
        )
        
        # Decoder: Reconstructs data
        self.decoder = nn.Sequential(
            nn.Linear(6, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            
            nn.Linear(24, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(0.2),
            
            nn.Linear(48, input_dim) 
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(net, trainloader, validaton_loader,partition_id, epochs, lr,mu ,device):
    """Train the model on the training set."""

    print_memory_usage()
    net.to(device)
    global_params = [p.detach().clone() for p in net.parameters()]
    optimizer = torch.optim.AdamW(net.parameters(),lr=lr, weight_decay=0.05) # weight_decay L2 regularization
    criterion = nn.MSELoss()
    print("training device", partition_id)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    for epoch in range(epochs):
        net.train()
        train_loss = 0
        for data in trainloader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            #loss = criterion(outputs, inputs)
            
            #FedProx           
            proximal_term = 0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                 proximal_term += torch.sum((local_weights - global_weights.to(device))**2)
            loss = criterion(outputs, inputs) + (mu / 2.0) * proximal_term
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss/len(trainloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for data in validaton_loader:
                inputs = data[0].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(validaton_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            #best_model_wts = copy.deepcopy(net.state_dict()) 
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Stop if overfitting detected
        if patience_counter >= patience:
            print(f'\n Early stopping triggered at epoch {epoch+1}')
            print(f'   Validation loss has not improved for {patience} epochs')
            print(f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')
            break

    print_memory_usage()  
     
    avg_trainloss = train_loss/len(trainloader)
    avg_valloss = val_loss / len(validaton_loader)
    print(f'Train Loss: {avg_trainloss:.6f} | Val Loss: {avg_valloss:.6f}')

    return avg_trainloss, avg_valloss    

def test(net, X_test_full, X_Validation,partition_id, device,X_train_dt,y_dt): #remove this for no dt
    """Validate the model on the test set."""

    print_memory_usage()
    net.eval()
    print("testing device", partition_id)
    net.to(device)
    with torch.no_grad():
         
        X_train_dt = torch.FloatTensor(X_train_dt).to(device) #remove this for no dt
        X_test_full = torch.FloatTensor(X_test_full).to(device)
        
        #encoding layer for classifier 
        X_train_encoded = net.encoder(X_train_dt).detach().numpy() #remove this for no dt
        X_test_encoded = net.encoder(X_test_full).numpy()
        
        # Validation reconstruction errors
        X_val_tensor = torch.FloatTensor(X_Validation).to(device)
        recon_val = net(X_val_tensor)
        errors_val = torch.mean(torch.square(X_val_tensor - recon_val), dim=1).cpu().numpy()

        # reconstruction error for classifier
        recon_dt = net(X_train_dt)
        errors_dt = torch.mean(torch.square(X_train_dt - recon_dt), dim=1).cpu().numpy() #remove this for no dt

        dt_stats = np.std(X_train_dt.cpu().numpy(), axis=1) #remove this for no dt
        
        # reconstruction error of full Test set
        X_test_full_tensor = X_test_full
        recon_full = net(X_test_full_tensor)
        errors_full = torch.mean(torch.square(X_test_full_tensor - recon_full), dim=1).cpu().numpy()
        stats_test = np.std(X_test_full_tensor.cpu().numpy(), axis=1)
        
        mu_val = np.mean(errors_val) #remove this for no dt
        sigma_val = np.std(errors_val) + 1e-8  #remove this for no dt

    # For training (used in classifier). Z-score normalisation
    errors_dt_norm = errors_dt #remove this for no dt

    # For testing (used in classifier). Z-score normalisation
    errors_full_norm = errors_full   #remove this for no dt

    X_features = np.column_stack([ #remove this for no dt
    X_train_encoded *10000,  # AE latent * 10000 because int16 would make 0.006 to 0
    errors_dt_norm*10000,  # Recon error * 10000 because int16 would make 0.006 to 0
    dt_stats*10000  # Flow stats * 10000 because int16 would make 0.006 to 0
    ])  # 8D total with 67 architecture, new architecture 8d since bottleneck is 6
    
    # DecisionTree (DT), alternative to unsupervised Thresholding,small for Tinyml deployment
    dt = DecisionTreeClassifier(max_depth=4,class_weight='balanced')  # 16 leaves max  #remove this for no dt
    dt_tinyml = DecisionTreeClassifier(max_depth=4,class_weight='balanced')  # 16 leaves max  #remove this for no dt; This is for exporting to MCU

    X_features_test = np.column_stack([X_test_encoded*10000, errors_full_norm*10000, stats_test*10000]) #remove this for no dt 
    
    # DT fit and predict
    dt.fit(X_features, y_dt)  #remove this for no dt
    y_pred = dt.predict(X_features_test) #remove this for no dt
    y_proba = dt.predict_proba(X_features_test)[:, 1]  #remove this for no dt

    # Unsupervised Thresholding, Percentile
    threshold_percentile = np.percentile(errors_val,85)
    print("Threshold percentile 85")

    y_pred_percentile = (errors_full > threshold_percentile).astype(int)

    print_memory_usage()

    X_train_int16 = np.clip(X_features, -32768, 32767).astype(np.int16) # remove this for no dt
    dt_tinyml.fit(X_train_int16, y_dt) # remove this for no dt

    return threshold_percentile, y_pred_percentile , errors_full, errors_val,y_pred,y_proba,dt,dt_tinyml #remove everything with dt for no dt

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    start_time = time.time()

    # Load the model and initialize it with the received weights
    model = Autoencoder(INPUT_DIM)

    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cpu")
    model.to(device)
    
    # Load data set
    # which_dataset 2 for BoTIoT; 3 for IoTID20 just like client

    _,_,X_test_full, X_test_validation, y_true,X_train_dt,y_dt = load_centralized_dataset()
    # which_dataset 0 for Training BoTIoT -> Testing IoTID20; everything else for Training IoTID20 -> Testing BoTIoT just like client
    #_,_,X_test_full, X_test_validation, y_true,X_train_dt,y_dt = load_crossdataset(which_dataset = 1)  #remove everythin with dt for no dt

    # Evaluate the global model on the test set
    threshold_ae, y_pred_percentile, errors_full, errors_val,y_pred,y_proba,dt,dt_tinyml = test( #remove everythin with dt for no dt
        model,
        X_test_full, X_test_validation, 1000,
        device, 
        X_train_dt,y_dt #remove this for no dt
    )

    # Metrics
    # Confusion matrix
    cmpercentile = confusion_matrix(y_true, y_pred_percentile)
    cm_dt = confusion_matrix(y_true,y_pred) #remove this for no dt
    #print(f"\nConfusion Matrix percentile Percentile:")
    #print(cmpercentile)
    #print(f"\nConfusion Matrix Proba:")
    #print(cm_proba)
  
    print(f"\nConfusion Matrix Classify:")
    print(cm_dt) #remove this for no dt

    tn_dt, fp_dt, fn_dt, tp_dt = confusion_matrix(y_true, y_pred).ravel() #remove this for no dt
    tn_percentile, fp_percentile, fn_percentile, tp_percentile = confusion_matrix(y_true, y_pred_percentile).ravel()

    # Use Confusion Matrix to calculate fnr and fpr for Percentile and DT 
    fprpercentile=cmpercentile[0][1]/ (cmpercentile[0][0]+cmpercentile[0][1]) 
    fnrpercentile=cmpercentile[1][0]/ (cmpercentile[1][1]+cmpercentile[1][0]) 
    
    fprdt=cm_dt[0][1] / (cm_dt[0][0]+cm_dt[0][1]) #remove this for no dt
    fnrdt=cm_dt[1][0] / (cm_dt[1][1]+cm_dt[1][0]) #remove this for no dt
    
    # calculate intrusion detection capability
    idc_dt = intrusion_detection_capability(y_true, y_pred) #remove this for no dt
    idc_percentile = intrusion_detection_capability(y_true, y_pred_percentile)

    print("Intrusion Detection Capability:", "DT",idc_dt,"percentile",idc_percentile)
    
    # AP and ROC AUC for Percentile and DT
    a = average_precision_score(y_true, y_proba) #AP; #remove this for no dt
    b = roc_auc_score(y_true, y_proba) #ROC AUC  #remove this for no dt
    c = roc_auc_score(y_true,errors_full)
    d = average_precision_score(y_true,errors_full)
    #print("Testing if it is right ",y_pred[0],y_true[0],y_proba) #remove this for no dt
    # Clasification Report
    #crpercentile = classification_report(y_true,y_pred_percentile) #remove this for no dt
    crc = classification_report(y_true,y_pred) #remove this for no dt
    
    print(crc) #remove this for no dt
    print(f"ROC AUC DT",b, "ROC AUC Error", c) #remove this for no dt
    print(f"Fprpercentile",float("{:.2f}".format(fprpercentile)),"Fnrpercentile",float("{:.2f}".format(fnrpercentile)),"FPR DT",float("{:.2f}".format(fprdt)),"FNR DT",
          float("{:.2f}".format(fnrdt))) #remove this for no dt
    
    # calculate Matthews Corrcoef
    mcc_dt = matthews_corrcoef(y_true,y_pred) #remove this for no dt
    mcc_percentile = matthews_corrcoef(y_true,y_pred_percentile) 
    
    # calculate F1-Score
    f1_dt = f1_score(y_true, y_pred) #remove this for no dt
    f1_percentile = f1_score(y_true, y_pred_percentile)
    
    # calculate False Discovery Rate
    fdr_dt = fp_dt / (fp_dt + tp_dt) if (fp_dt + tp_dt) > 0 else 0 #remove this for no dt
    fdr_percentile = fp_percentile / (fp_percentile + tp_percentile) if (fp_percentile + tp_percentile) > 0 else 0
    print(f"F1-Score DT: {f1_dt:.4f}",f"F1-Score percentile: {f1_percentile:.4f}") #remove this for no dt
    print(f"False Discovery Rate DT: {fdr_dt:.4f}", f"False Discovery Rate percentile: {fdr_percentile:.4f}")
    print("MCC DT", mcc_dt, "MCC percentile", mcc_percentile)
    
    c_model = emlearn.convert(dt_tinyml, method='inline')
    c_model.save(file='tree_model.h', name='my_model')

    acc_ae = accuracy_score(y_true,y_pred_percentile)
    acc_dt = accuracy_score(y_true,y_pred) #remove this for no dt

    end_time = time.time()
    testing_time = end_time - start_time
    print(testing_time)

    # Construct and return reply Message
    # Return the evaluation metrics
    return MetricRecord({
        #"threshold": float(threshold_ae),
        "FPR AE": float("{:.4f}".format(fprpercentile)),
        "FNR AE": float("{:.4f}".format(fnrpercentile)),
        "MCC DT": float("{:.4f}".format(mcc_dt)), #remove this for no dt
        "MCC AE": float("{:.4f}".format(mcc_percentile)),
        "FPR DT": float("{:.4f}".format(fprdt)), #remove this for no dt
        "FNR DT": float("{:.4f}".format(fnrdt)), #remove this for no dt
        "ROC AUC DT": float("{:.4f}".format(b)), #remove this for no dt
        "ROC AUC AE": float("{:.4f}".format(c)),
        "AP DT": float("{:.4f}".format(a)), #remove this for no dt
        "AP AE": float("{:.4f}".format(d)),
        "IDC DT": float("{:.4f}".format(idc_dt)), #remove this for no dt
        "IDC AE": float("{:.4f}".format(idc_percentile)),
        "F1-Score DT": float("{:.4f}".format(f1_dt)), #remove this for no dt
        "F1-Score AE": float("{:.4f}".format(f1_percentile)),
        "FDR DT": float("{:.4f}".format(fdr_dt)), #remove this for no dt
        "FDR AE": float("{:.4f}".format(fdr_percentile)),                
        "Confusion Matrix DT": cm_dt.flatten().tolist(), #remove this for no dt
        "Confusion Matrix AE": cmpercentile.flatten().tolist(),
        "Accuracy AE": float("{:.4f}".format(acc_ae)),
        "Accuracy DT": float("{:.4f}".format(acc_dt)),  #remove this for no dt
        #"time": float(("{:.4f}".format(testing_time)))
    })
    
def intrusion_detection_capability(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("TN",tn,"FP",fp,"FN",fn,"TP",tp)
    N = tp + tn + fp + fn

    # probabilities
    p_tp = tp / N
    p_tn = tn / N
    p_fp = fp / N
    p_fn = fn / N

    p_attack = (tp + fn) / N
    p_normal = (tn + fp) / N

    p_pred_attack = (tp + fp) / N
    p_pred_normal = (tn + fn) / N

    # Mutual Information
    terms = []

    if p_tp > 0:
        terms.append(p_tp * np.log2(p_tp / (p_attack * p_pred_attack)))

    if p_fn > 0:
        terms.append(p_fn * np.log2(p_fn / (p_attack * p_pred_normal)))

    if p_fp > 0:
        terms.append(p_fp * np.log2(p_fp / (p_normal * p_pred_attack)))

    if p_tn > 0:
        terms.append(p_tn * np.log2(p_tn / (p_normal * p_pred_normal)))

    mutual_information = sum(terms)

    # Entropy of actual classes
    Hx = 0
    if p_attack > 0:
        Hx -= p_attack * np.log2(p_attack)
    if p_normal > 0:
        Hx -= p_normal * np.log2(p_normal)

    # IDC
    IDC = mutual_information / Hx if Hx != 0 else 0

    return IDC

def print_memory_usage():

    process = psutil.Process(os.getpid())

    # Convert bytes to Megabytes
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Current RAM usage: {mem_mb:.2f} MB")