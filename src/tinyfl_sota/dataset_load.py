"""if-and-auto: A Flower / PyTorch app."""
import os
import sys

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def log1p_static_scale(df):
    """
    Applies Log1p and divides by a static constant.
    Prevents feature misalignment in Non-IID scenarios.
    """
    # 1. Select only numeric columns (exclude labels/IDs)
    # If you already dropped them, you can just use df.columns
    if hasattr(df, 'values'):
            X = df.values
    else:
        X = df
    
    # 2. Apply Log1p: ln(1 + x)
    # This compresses the massive range of flow duration/bytes
    # 0 -> 0.0
    # 1,000,000 -> 13.8
    # 100,000,000 -> 18.4
    X = np.log1p(np.clip(X, 0, None))
    
    # 3. Divide by Static Constant (20.0)
    # We use 20.0 because log(Max Network Value) is roughly 18-19.
    # This forces almost all data into the [0.0, 1.0] range.
    X_scaled = X /20
    
    return X_scaled.astype(np.float32)
rng = np.random.default_rng(seed=42)

def load_cross_data(partition_id: int, num_partitions: int, which_dataset: int):
    
    if which_dataset == 0:
    # load Training Dataset CIC-BoTIoT, Testing Dataset IoTID20 for Clients
        dfs=[]
        j = partition_id *5 # 
        if (partition_id >= 2): # Clients 3 and 4 load Datasplit with high Flowduration 
            for i in range(5):
                
                # load 5000 benign Samples from Training Dataset
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+60+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True) 
        else:    # Clients 1 and 2 load Datasplit with low Flowduration 
            for i in range(5):
                
                # load 5000 benign Samples from Training Dataset
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
    
        # load anomaly samples from Training Dataset for training Decisiontree
        df_attack = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_{partition_id+1}.csv")
            
        # load anomaly samples from testing Dataset for evaluation
        attacks = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_{partition_id+1}.csv")
    
        # load benign samples from testing Dataset for evaluation
        bengin = pd.read_csv(f"small_IoTID20_dataset_benign_clean/split_{partition_id+1}.csv")

    # concat the samples for evaluation into X_test_attacks and change here to change the ratio of the anomalies in testing
    X_test_attacks = pd.concat([bengin,attacks.sample(n= int((len(bengin)*0.1)),random_state=42)])

    print(f"Before Training {partition_id}",len(df_benign))
    print(f"Before Testing {partition_id}",len(X_test_attacks))
    # Features that are not used 
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']

    # replace '_' to ' ' from features. Makes it so the feature names are written the same
    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    df_attack.columns = df_attack.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')    
    
    # "Feature Selection" remove the label_col features  
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)

    # remove the label_col features
    df_attack = df_attack.drop([col for col in label_col if col in df_attack.columns], axis=1)

    # remove the label_col features
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)

    # label attack and benign data of Testing data for evaluation 
    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0)
    else: 
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)
    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    # remove label features
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    df_attack = df_attack.drop(columns=["Label"])
    print(f"After Training {partition_id}",len(df_benign))
    print(f"After Testing {partition_id}",len(X_test_attacks))
    # this is just to check
    num_ones = target.sum()
    num_zeros = len(target) - num_ones

    print(f"Benign Testing {partition_id}",num_zeros)
    print(f"Anomaly Testing {partition_id}",num_ones)
  
    # split training data into Training and validation set
    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    # split splitted training data into Training and training set for Decision Tree
    X_train_, X_benign_dt   = train_test_split(X_train_, test_size=0.2,random_state=42) #remove this for no dt

    # Scaling
    X_train = log1p_static_scale(X_train_)
    
    X_Validation = log1p_static_scale(X_val)
    X_benign_dt = log1p_static_scale(X_benign_dt) #remove this for no dt
    sample_attack = int(len(X_benign_dt)*0.1)    
    df_attack = rng.choice(df_attack , size=sample_attack, axis=0, replace=False)
    df_attack = log1p_static_scale(df_attack)
    
    # training data for Decisiontree
    X_train_dt = np.vstack([X_benign_dt, df_attack]) #remove this for no dt
    y_true_early = np.hstack([ #remove this for no dt
    np.zeros(len(X_benign_dt)),  # Benign holdout
    np.ones(len(df_attack))   # All attacks
    ])
    #
    y_true = target
    _,X_train_dt,_, y_dt = train_test_split(X_train_dt,y_true_early, test_size=0.25,random_state=42,stratify=y_true_early) #remove this for no dt

    print("Training Data amount:",len(X_train))
    X_test_full = log1p_static_scale(X_test_attacks)
    X_test_full,_,y_true, _ = train_test_split(X_test_full,y_true, test_size=0.25,random_state=42,stratify=y_true)

    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1))
  
    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_dt,y_dt #remove this for no dt


def load_mono_dataset(partition_id: int, num_partitions: int,which_dataset:int):

    if which_dataset ==0 :
        #Training and Testing CIC-BoTIoT
        dfs = []
        i=0
        j = partition_id *5
        if (partition_id >= 2): # Clients 3 and 4 load Datasplit with high Flowduration 
            for i in range(5):
                
                # load 5000 benign Samples
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+60+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        else:     # Clients 1 and 2 load Datasplit with low Flowduration 
            for i in range(5):
                
                # load 5000 benign Samples
                df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/split_{partition_id+1+j+i}.csv")
                dfs.append(df)
                df_benign = pd.concat(dfs,ignore_index=True)
        print(f"Client {partition_id} Mean Duration: {df_benign['Flow Duration'].mean()}")
        # load 1000 anomaly samples
        X_test_attacks = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_{partition_id+1}.csv")
        print("load_data_BoTIoT split")
    
    print(f"Before Training {partition_id}",len(df_benign))
    print(f"Before Testing {partition_id}",len(X_test_attacks))
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']
    
    # replace '_' to ' ' from features. Makes it so the feature names are written the same
    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')
    
    # remove the label_col features
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)
    #brauch ich nicht in dieser Function
    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)

    else:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0) #brauch ich nicht in dieser Function
        
    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    # drop label columns
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    print(f"After Training {partition_id}",len(df_benign))
    print(f"After Testing {partition_id}",len(X_test_attacks))
    # split training data into Training and validation set
    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    # split splitted training data into Training and testing set
    X_train_, X_test   = train_test_split(X_train_, test_size=0.25,random_state=42)
    
    # scaling
    X_train = log1p_static_scale(X_train_)

    X_Validation = log1p_static_scale(X_val)
    # ratio of anomalies in testing and training of DT
    sample_attack = int(len(X_test)*0.1)
    X_test_attacks = rng.choice(X_test_attacks , size=sample_attack, axis=0, replace=False)
    # testing set that will be split into testing set and dt train set
    X_test_full = np.vstack([X_test, X_test_attacks]) #remove this for no dt
    y_true_early = np.hstack([ #remove this for no dt
    np.zeros(len(X_test)),  
    np.ones(len(X_test_attacks))   
    ])

    X_test_full = log1p_static_scale(X_test_full)
    # split testing data into testing and dt training set
    X_test_full,X_train_dt,y_true, y_dt = train_test_split(X_test_full,y_true_early, test_size=0.25,random_state=42,stratify=y_true_early) #remove this for no dt
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1))
    # Dataloaders for training
    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)

    print("Samples train",len(X_train),"val",len(X_Validation),"X_train_dt",len(X_train_dt),"attacksx",len(X_test_attacks))
    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_dt,y_dt #remove this for no dt

def load_centralized_dataset(which_dataset):

    if which_dataset ==0 :
        # load Testing Dataset CIC-BoTIoT for Server side Evaluation

        dfs=[]
        i=0
        for i in range(5):
            
            # load 5000 benign Samples with random flowduration
            df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)
        # load anomaly samples with random flow duration
        X_test_attacks = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_29.csv")
        # calibration split for calibrating the tflite model for ESP32 Deployment
        calibration_split = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/glo_split_8.csv")

        print("load_data_BoTIoT 90/10")
    elif which_dataset ==1 :
    # load Training Dataset IoTID20, Testing Dataset CIC-BoTIoT for Server side Evaluation

        dfs=[]
        i=0
        for i in range(5):
            
            # load 5000 benign Samples with random flowduration
            df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)
    
        # load anomaly samples with random flow duration
        X_test_attacks = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_29.csv")
        # calibration split for calibrating the tflite model for ESP32 Deployment
        calibration_split = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/glo_split_8.csv")
        print("load_data_IoTID20 90/10")
    
    print(f"Before Training ",len(df_benign))
    print(f"Before Testing",len(X_test_attacks))
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']
    
    # replace '_' to ' ' from features. Makes it so the feature names are written the same
    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')
    calibration_split.columns = calibration_split.columns.str.replace('_', ' ')
    
    # remove the label_col features
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)
    calibration_split = calibration_split.drop([col for col in label_col if col in calibration_split.columns], axis=1)

    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)

    else:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0) #brauch ich nicht in dieser Function
        
    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    # drop label columns
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    calibration_split = calibration_split.drop(columns=["Label"])
    print(f"After Training",len(df_benign))
    print(f"After Testing",len(X_test_attacks))
    
    # Split df_benign into Validation(X_val), Benign data for Test(X_test) and Training (X_train_)
    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    X_train_, X_test   = train_test_split(X_train_, test_size=0.25,random_state=42)

    # scaling
    X_train = log1p_static_scale(X_train_)
    
    calibration_split = log1p_static_scale(calibration_split)

    X_Validation = log1p_static_scale(X_val)
    
    # changing the Ratio of how many Anomaly samples in Testing
    sample_attack = int(len(X_test)*0.1)    
    X_test_attacks = rng.choice(X_test_attacks , size=sample_attack, axis=0, replace=False)
    
    # testing set that will be split into testing set and dt train set
    X_test_full = np.vstack([X_test, X_test_attacks]) 
    y_true_early = np.hstack([ 
    np.zeros(len(X_test)),  # Benign holdout
    np.ones(len(X_test_attacks))   # All attacks
    ])

    X_test_full = log1p_static_scale(X_test_full)
    # split testing data into testing and dt training set
    X_test_full,X_train_dt,y_true, y_dt = train_test_split(X_test_full,y_true_early, test_size=0.25,random_state=42,stratify=y_true_early) #remove X_train_dt and y_dt for no dt
    # Dataloaders for training
    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1)) 

    #Exporting Datasets into .h files for MCU deployment
    lenge = len(X_Validation) 

    # --- Write the C Header File ---
    filename = f"mcu_val_data.h"
    print(f"Exporting {lenge} samples to {filename}...")

    with open(filename, "w") as f:
        f.write("/* Auto-generated TinyML Test Dataset */\n")
        f.write("#pragma once\n\n")
        
        f.write(f"const int MCU_val_SAMPLES = {lenge};\n")
        f.write(f"const int MCU_val_FEATURES = {X_Validation.shape[1]};\n\n")

        # Write X data (Features)
        f.write(f"const float mcu_val_x[{lenge}][{X_Validation.shape[1]}] = {{\n")
        for i, row in enumerate(X_Validation):
            # Format numbers to 6 decimal places to save string space
            row_str = ", ".join([f"{val:.15f}" for val in row])
            f.write(f"    {{{row_str}}}")
            if i < lenge - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

    lenge = len(X_test_full) 

    y_mcu = y_true
    # --- Write the C Header File ---
    filename = f"mcu_test_data.h"
    print(f"Exporting {lenge} samples to {filename}...")

    with open(filename, "w") as f:
        f.write("/* Auto-generated TinyML Test Dataset */\n")
        f.write("#pragma once\n\n")
        
        f.write(f"const int MCU_TEST_SAMPLES = {lenge};\n")
        f.write(f"const int MCU_NUM_FEATURES = {X_test_full.shape[1]};\n\n")

        # Write X data (Features)
        f.write(f"const float mcu_test_x[{lenge}][{X_test_full.shape[1]}] = {{\n")
        for i, row in enumerate(X_test_full):
            # Format numbers to 6 decimal places to save string space
            row_str = ", ".join([f"{val:.15f}" for val in row])
            f.write(f"    {{{row_str}}}")
            if i < lenge - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # Write Y data (Labels)
        f.write(f"const int mcu_test_y[{lenge}] = {{\n    ")
        y_str = ", ".join([str(int(val)) for val in y_mcu])
        f.write(y_str)
        f.write("\n};\n")
        
    np.save("calibration_data.npy", calibration_split) # calibration dataset for tflite convert

    print("Export complete!")

    print("Samples train",len(X_train),"val",len(X_Validation),"X_train_dt",len(X_train_dt),"attacksx",len(X_test_attacks))

    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_dt,y_dt #remove this for no dt

def load_crossdataset(which_dataset):
    
    if which_dataset == 0:
    # load Training Dataset CIC-BoTIoT, Testing Dataset IoTID20 for Server side Evaluation

        dfs = []
        for i in range(5):
            
            #5000 benign samples of training dataset. Used for calculating Threshold and DT training
            df = pd.read_csv(f"small_BoTIoT_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)

        # 1000 anomaly samples of training dataset. Used for DT training
        df_attack = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_29.csv")
        # anomaly samples of testing dataset.             
        attacks = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_29.csv")
        # benign samples of testing dataset.
        bengin = pd.read_csv(f"small_IoTID20_dataset_benign_clean/split_1.csv")
        print("load_data")
    if which_dataset == 1:
        # load Training Dataset IoTID20, Testing Dataset CIC-BoTIoT for Server side Evaluation
        dfs = []

        for i in range(5):
            
            #5000 benign samples of training dataset. Used for calculating Threshold and DT training
            df = pd.read_csv(f"small_IoTID20_dataset_benign_clean_noleak/glo_split_{1+i}.csv")
            dfs.append(df)
        df_benign = pd.concat(dfs,ignore_index=True)
       
        # 1000 anomaly samples of training dataset. Used for DT training
        df_attack = pd.read_csv(f"small_IoTID20_dataset_allattacks_clean/split_29.csv")
        
        # anomaly samples of testing dataset.             
        attacks = pd.read_csv(f"small_BoTIoT_dataset_allattacks_clean/split_29.csv")

        # benign samples of testing dataset.
        bengin = pd.read_csv(f"small_BoTIoT_dataset_benign_clean/split_1.csv") 
        print("load_data2")
    # concat the samples for evaluation into X_test_attacks and change here to change the ratio of the anomalies in testing
    X_test_attacks = pd.concat([bengin,attacks.sample(n= int((len(bengin)*0.1)),random_state=42)])

    print(f"Before Training ",len(df_benign))
    print(f"Before Testing ",len(X_test_attacks))
    label_col = ['Flow ID', 'Flow_ID', 'Src IP', 'Src_IP', 'Dst IP', 'Dst_IP',
               'Src Port', 'Src_Port', 'Dst Port', 'Dst_Port', 'Timestamp', 'Fwd PSH Flags', 'Fwd_PSH_Flags',
            'Fwd URG Flags', 'Fwd_URG_Flags',"Fwd Byts/b Avg","Fwd_Byts/b_Avg","Fwd Pkts/b Avg","Fwd_Pkts/b_Avg",
           "Fwd Blk Rate Avg","Fwd_Blk_Rate_Avg","Bwd Byts/b Avg","Bwd_Byts/b_Avg","Bwd Pkts/b Avg","Bwd_Pkts/b_Avg",
           "Bwd Blk Rate Avg","Bwd_Blk_Rate_Avg","Init Fwd Win Byts","Init_Fwd_Win_Byts", "Fwd Seg Size Min","Fwd_Seg_Size_Min"
           ,'Cat','Sub Cat','Attack']
    
    # replace '_' to ' ' from features. Makes it so the feature names are written the same
    df_benign.columns = df_benign.columns.str.replace('_', ' ')
    df_attack.columns = df_attack.columns.str.replace('_', ' ')
    X_test_attacks.columns = X_test_attacks.columns.str.replace('_', ' ')    
    
    # "Feature Selection" Remove features
    df_benign = df_benign.drop([col for col in label_col if col in df_benign.columns], axis=1)
    df_attack = df_attack.drop([col for col in label_col if col in df_attack.columns], axis=1)
    X_test_attacks = X_test_attacks.drop([col for col in label_col if col in X_test_attacks.columns], axis=1)

    # label attack and benign data of Testing data for evaluation 
    if which_dataset == 0:
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!="Normal" else 0)
    else: 
        X_test_attacks["target"] = X_test_attacks["Label"].apply(lambda x: 1 if x!=0 else 0)

    target = X_test_attacks["target"].to_numpy()
    print("Testing", len(target))
    # remove label column
    X_test_attacks = X_test_attacks.drop(columns=["Label"])
    X_test_attacks = X_test_attacks.drop(columns=["target"])
    df_benign = df_benign.drop(columns=["Label"])
    df_attack = df_attack.drop(columns=["Label"])
    print(f"After Training ",len(df_benign))
    print(f"After Testing ",len(X_test_attacks))
    # this is just to check
    num_ones = target.sum()
    num_zeros = len(target) - num_ones

    print(f"Benign Testing ",num_zeros)
    print(f"Anomaly Testing ",num_ones)
  
    # split training data into Training and validation set and benign data for DT training
    X_train_, X_val   = train_test_split(df_benign, test_size=0.2,random_state=42)
    X_train_, X_benign_dt   = train_test_split(X_train_, test_size=0.2,random_state=42) #remove  X_benign_dt for no dt

    # Scaling
    X_train = log1p_static_scale(X_train_)

    X_Validation = log1p_static_scale(X_val)
    X_benign_dt = log1p_static_scale(X_benign_dt) #remove this for no dt
    # i need to change this
    sample_attack = int(len(X_benign_dt)*0.1)    
    df_attack = rng.choice(df_attack , size=sample_attack, axis=0, replace=False)
    df_attack = log1p_static_scale(df_attack)
    X_train_dt = np.vstack([X_benign_dt, df_attack]) #remove this for no dt
    y_true_early = np.hstack([ #remove this for no dt
    np.zeros(len(X_benign_dt)),  # Benign holdout
    np.ones(len(df_attack))   # All attacks
    ])
    # 
    y_true = target
    _,X_train_dt,_, y_dt = train_test_split(X_train_dt,y_true_early, test_size=0.25,random_state=42,stratify=y_true_early) #remove this for no dt
    print(len(X_train_dt)) #remove this for no dt

    print("Training Data amount:",len(X_train))
    X_test_full = log1p_static_scale(X_test_attacks)
    X_test_full,_,y_true, _ = train_test_split(X_test_full,y_true, test_size=0.25,random_state=42,stratify=y_true)

    # Dataloaders for training
    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)
    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64 ,shuffle=True)
    print("Full",len(y_true),"Benign",np.sum(y_true == 0),"Anomaly",np.sum(y_true == 1))
    
    #Exporting Datasets into .h files for MCU deployment
    lenge = len(X_Validation) 

    # --- Write the C Header File ---
    filename = f"mcu_val_data.h"
    print(f"Exporting {lenge} samples to {filename}...")

    with open(filename, "w") as f:
        f.write("/* Auto-generated TinyML Test Dataset */\n")
        f.write("#pragma once\n\n")
        
        f.write(f"const int MCU_val_SAMPLES = {lenge};\n")
        f.write(f"const int MCU_val_FEATURES = {X_Validation.shape[1]};\n\n")

        # Write X data (Features)
        f.write(f"const float mcu_val_x[{lenge}][{X_Validation.shape[1]}] = {{\n")
        for i, row in enumerate(X_Validation):
            # Format numbers to 6 decimal places to save string space
            row_str = ", ".join([f"{val:.15f}" for val in row])
            f.write(f"    {{{row_str}}}")
            if i < lenge - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
    lenge = len(X_test_full) 

    y_mcu = y_true
    # --- Write the C Header File ---
    filename = f"mcu_test_data.h"
    print(f"Exporting {lenge} samples to {filename}...")

    with open(filename, "w") as f:
        f.write("/* Auto-generated TinyML Test Dataset */\n")
        f.write("#pragma once\n\n")
        
        f.write(f"const int MCU_TEST_SAMPLES = {lenge};\n")
        f.write(f"const int MCU_NUM_FEATURES = {X_test_full.shape[1]};\n\n")

        # Write X data (Features)
        f.write(f"const float mcu_test_x[{lenge}][{X_test_full.shape[1]}] = {{\n")
        for i, row in enumerate(X_test_full):
            # Format numbers to 6 decimal places to save string space
            row_str = ", ".join([f"{val:.6f}" for val in row])
            f.write(f"    {{{row_str}}}")
            if i < lenge - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")

        # Write Y data (Labels)
        f.write(f"const int mcu_test_y[{lenge}] = {{\n    ")
        y_str = ", ".join([str(int(val)) for val in y_mcu])
        f.write(y_str)
        f.write("\n};\n")

    return trainloader, validaton_loader ,X_test_full, X_Validation, y_true,X_train_dt,y_dt #remove this for no dt


    
