import os
import sys
import hashlib

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.tinyfl import constants

def url_hash(value) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 256

def _load_binary_info_by_src_ip(connection_file: str, binary_info_file: str, malicious_format: bool) -> dict:

    if not os.path.exists(connection_file) or not os.path.exists(binary_info_file):
        return {}

    conn = pd.read_csv(connection_file)
    info = pd.read_csv(binary_info_file)

    if malicious_format:
        # repariert den kaputten Spalten-Header
        combined_col = [c for c in info.columns if c.startswith("obfuscation[")][0]

        def _split_obfuscation(s):
            parts = str(s).split(",")
            parts = (parts + ["0", "0", "0", "0"])[:4]
            return parts

        parts = info[combined_col].apply(_split_obfuscation)

        info["obfuscation_Size(Bytes)"] = parts.apply(lambda p: p[0])
        info["obfuscation_Strings"] = parts.apply(lambda p: p[1])
        info["obfuscation_Strings_per_kB"] = parts.apply(lambda p: p[2])
        info["obfuscation_Entropy"] = parts.apply(lambda p: p[3])


    per_sha = {}
    for _, row in info.iterrows():
        values = {}
        for c in constants.BINFO_NUMERIC_COLS:
            v = pd.to_numeric(row[c], errors="coerce")
            values[f"binfo_{c}"] = 0.0 if pd.isna(v) else float(v)
        for c in constants.BINFO_TEXT_COLS:
            values[f"binfo_{c}"] = url_hash(row[c])
        per_sha[row["filename"]] = values

    lookup = {}
    device_ip_col = "src_ip" if malicious_format else "dst_ip"
    for device_ip, sha in zip(conn[device_ip_col], conn["sha256"]):
        if sha in per_sha:
            lookup[device_ip] = per_sha[sha]
    return lookup

# Load IOTPOT
def load_binary_info_test(file: str) -> np.ndarray:
    info = pd.read_csv(file)

    # Handle the combined/malformed obfuscation column if present.
    combined_cols = [c for c in info.columns if c.startswith("obfuscation[")]
    if combined_cols:
        parts = info[combined_cols[0]].apply(
            lambda s: (str(s).split(",") + ["0"] * 4)[:4]
        )
        info["obfuscation_Size(Bytes)"] = parts.apply(lambda p: p[0])
        info["obfuscation_Strings"] = parts.apply(lambda p: p[1])
        info["obfuscation_Strings_per_kB"] = parts.apply(lambda p: p[2])
        info["obfuscation_Entropy"] = parts.apply(lambda p: p[3])

    X = pd.DataFrame(0.0, index=info.index, columns=constants.FEATURES)

    for c in constants.BINFO_NUMERIC_COLS:
        X[f"binfo_{c}"] = pd.to_numeric(info[c], errors="coerce").fillna(0.0)

    for c in constants.BINFO_TEXT_COLS:
        X[f"binfo_{c}"] = info[c].apply(url_hash)

    return scale_features(X[constants.FEATURES].to_numpy())

# Feature creation
def create_features(df: pd.DataFrame, is_malicious: bool) -> pd.DataFrame:

    df = df.copy()
    raw_count = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    df["count"] = np.log1p(raw_count)
    df["url"] = df["url"].apply(url_hash)

    if is_malicious:
        lookup = _load_binary_info_by_src_ip(constants.MALICIOUS_CONNECTION_FILE, constants.MALICIOUS_BINARY_INFO_FILE, malicious_format=True)
    else:
        lookup = _load_binary_info_by_src_ip(constants.BENIGN_CONNECTION_FILE, constants.BENIGN_BINARY_INFO_FILE, malicious_format=False)

    binfo_cols = [f"binfo_{c}" for c in constants.BINFO_NUMERIC_COLS + constants.BINFO_TEXT_COLS]
    for col in binfo_cols:
        df[col] = df["src_ip"].apply(lambda ip: lookup.get(ip, {}).get(col, 0.0))

    df["target"] = 1 if is_malicious else 0

    return df[constants.FEATURES + ["target"]]

def scale_features(X: np.ndarray) -> np.ndarray:

    X = X.astype(np.float32).copy()
    idx = {name: i for i, name in enumerate(constants.FEATURES)}

    X[:,idx["count"]] = X[:,idx["count"]] / 20.0
    X[:,idx["url"]] = X[:,idx["url"]] / 255.0

    for c in constants.BINFO_TEXT_COLS:
        X[:,idx[f"binfo_{c}"]] = X[:,idx[f"binfo_{c}"]] / 255.0  # hashed -> [0,255]
    for c in constants.BINFO_NUMERIC_COLS:
        col = idx[f"binfo_{c}"]
        X[:,col] = np.log1p(np.clip(X[:,col], 0, None)) / 20.0

    return X

# Loading the data and building splits
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:

    benign_raw = pd.read_csv(constants.BENIGN_FILE)
    malicious_raw = pd.read_csv(constants.MALICIOUS_FILE)

    benign = create_features(benign_raw, is_malicious=False).drop(columns=["target"]).reset_index(drop=True)
    malicious = create_features(malicious_raw, is_malicious=True).drop(columns=["target"]).reset_index(drop=True)

    return benign, malicious

def partition(df: pd.DataFrame, partition_id: int, num_partitions: int) -> pd.DataFrame:

    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    shard_indices = np.array_split(np.arange(len(shuffled)), num_partitions)

    return shuffled.iloc[shard_indices[partition_id]].reset_index(drop=True)

def build_splits(benign: pd.DataFrame, malicious: pd.DataFrame):

    benign_train, benign_test = train_test_split(benign, test_size=0.2, random_state=42)
    benign_train_ae, benign_val_ae = train_test_split( benign_train, test_size=0.2, random_state=42)
    benign_train_ae, benign_for_classifier = train_test_split( benign_train_ae, test_size=0.2, random_state=42)

    if len(malicious) >= 4:
        mal_train, mal_test = train_test_split(malicious, test_size=0.2, random_state=42)
    else:
        mal_train, mal_test = malicious, malicious # too few malicious samples

    X_train = scale_features(benign_train_ae[constants.FEATURES].to_numpy())

    X_Validation = scale_features(benign_val_ae[constants.FEATURES].to_numpy())

    X_train_classifier = scale_features(pd.concat([benign_for_classifier, mal_train], ignore_index=True)[constants.FEATURES].to_numpy())

    y_class = np.concatenate( [np.zeros(len(benign_for_classifier)), np.ones(len(mal_train))])

    #X_test_full = scale_features(pd.concat([benign_test, mal_test], ignore_index=True)[FEATURES].to_numpy())
    X_benign_test = scale_features(benign_test[constants.FEATURES].to_numpy())
    X_malicious_test = scale_features(mal_test[constants.FEATURES].to_numpy())

    y_true = np.concatenate([np.zeros(len(benign_test)), np.ones(len(mal_test))])

    trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)

    validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64, shuffle=True)

    #return trainloader, validaton_loader, X_test_full, X_Validation, y_true, X_train_classifier, y_class
    return (
        trainloader, validaton_loader,
        X_benign_test, X_malicious_test, X_Validation,
        X_train_classifier, y_class
    )

def load_mono_dataset(partition_id: int, num_partitions: int):

    benign, malicious = load_raw()

    benign_part = partition(benign, partition_id, num_partitions)
    malicious_part = partition(malicious, partition_id, num_partitions)

    return build_splits(benign_part, malicious_part)

def load_device_dataset(partition_id: int):

    device_name = constants.DEVICE_NAMES[partition_id]
    scenario_names = constants.DEVICE_SCENARIOS[device_name]

    dfs = []
    for scenario in scenario_names:
        path = os.path.join(constants.SCENARIOS_DIR, scenario, "network_traffic.csv")
        dfs.append(pd.read_csv(path))
    benign_raw = pd.concat(dfs, ignore_index=True)

    malicious_raw = pd.read_csv(constants.MALICIOUS_FILE)

    benign = create_features(benign_raw, is_malicious=False).drop(columns=["target"]).reset_index(drop=True)
    malicious = create_features(malicious_raw, is_malicious=True).drop(columns=["target"]).reset_index(drop=True)

    return build_splits(benign, malicious)

def load_centralized_dataset():
    return build_splits(*load_raw())
