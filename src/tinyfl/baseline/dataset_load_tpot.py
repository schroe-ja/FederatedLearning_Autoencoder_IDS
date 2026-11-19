import os
import sys
import hashlib

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.tinyfl import constants

class DatasetLoad():
    def add_metadata(self, df, phase, scenario=None, device=None):
        df = df.copy()

        df["phase"] = phase
        df["scenario_group"] = scenario
        df["device_group"] = device

        return df

    @staticmethod
    def _url_hash(value) -> int:
        digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 256

    @staticmethod
    def _load_binary_info_by_src_ip(connection_file: str, binary_info_file: str, malicious_format: bool) -> dict:

        if not os.path.exists(connection_file) or not os.path.exists(binary_info_file):
            return {}

        conn = pd.read_csv(connection_file)
        info = pd.read_csv(binary_info_file)

        if malicious_format:
            # repariert den kaputten Spalten-Header
            combined_col = [c for c in info.columns if c.startswith("obfuscation[")]

            if combined_col:
                combined_col = combined_col[0]

                def _split_obfuscation(s):
                    parts = str(s).split(",")
                    parts = (parts + ["0", "0", "0", "0"])[:4]
                    return parts

                parts = info[combined_col].apply(_split_obfuscation)

                info["obfuscation_Size(Bytes)"] = parts.apply(lambda p: p[0])
                info["obfuscation_Strings"] = parts.apply(lambda p: p[1])
                info["obfuscation_Strings_per_kB"] = parts.apply(lambda p: p[2])
                info["obfuscation_Entropy"] = parts.apply(lambda p: p[3])

        
        conn["src_ip"] = conn["src_ip"].astype(str).str.strip()
        conn["sha256"] = conn["sha256"].astype(str).str.strip().str.lower()

        info["filename"] = info["filename"].astype(str).str.strip().str.lower()

        per_sha = {}

        for _, row in info.iterrows():
            values = {}

            for c in constants.BINFO_NUMERIC_COLS:
                v = pd.to_numeric(row[c], errors="coerce")
                values[f"binfo_{c}"] = 0.0 if pd.isna(v) else float(v)

            for c in constants.BINFO_TEXT_COLS:
                values[f"binfo_{c}"] = DatasetLoad._url_hash(row[c])

            per_sha[row["filename"]] = values

        conn["binary_features"] = conn["sha256"].map(per_sha)

        conn = conn.dropna(subset=["binary_features"])

        lookup = {}

        for ip, group in conn.drop_duplicates(
            subset=["src_ip", "sha256"]
        ).groupby("src_ip"):

            binary_values = list(group["binary_features"])

            aggregated = {}

            for c in constants.BINFO_NUMERIC_COLS:
                col = f"binfo_{c}"
                values = np.array(
                    [b[col] for b in binary_values],
                    dtype=np.float32
                )
                aggregated[col] = float(np.mean(values))

            for c in constants.BINFO_TEXT_COLS:
                col = f"binfo_{c}"
                values = [b[col] for b in binary_values]

                # Temporary baseline: most frequent encoded value
                aggregated[col] = max(set(values), key=values.count)

            lookup[ip] = aggregated

        return lookup

    # Load IOTPOT
    def load_binary_info_test(self, file: str) -> np.ndarray:
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
        X["event_type"] = constants.EVENT_TYPES["binary_information"]

        for c in constants.BINFO_NUMERIC_COLS:
            X[f"binfo_{c}"] = pd.to_numeric(info[c], errors="coerce").fillna(0.0)

        for c in constants.BINFO_TEXT_COLS:
            X[f"binfo_{c}"] = info[c].apply(self._url_hash)

        return self.scale_features(X[constants.FEATURES].to_numpy())

    # Feature creation
    #def create_features(self, df: pd.DataFrame, is_malicious: bool) -> pd.DataFrame:
#
    #    df = df.copy()
#
    #    for col in constants.FEATURES:
    #        df[col] = 0.0
#
    #    raw_count = pd.to_numeric(df["count"], errors="coerce").fillna(0)
#
    #    df["count"] = np.log1p(raw_count)
    #    df["url"] = df["url"].apply(self._url_hash)
#
    #    if is_malicious:
    #        lookup = self._load_binary_info_by_src_ip(constants.MALICIOUS_CONNECTION_FILE, constants.MALICIOUS_BINARY_INFO_FILE, malicious_format=True)
    #    else:
    #        lookup = self._load_binary_info_by_src_ip(constants.BENIGN_CONNECTION_FILE, constants.BENIGN_BINARY_INFO_FILE, malicious_format=False)
#
    #    binfo_cols = [f"binfo_{c}" for c in constants.BINFO_NUMERIC_COLS + constants.BINFO_TEXT_COLS]
    #    for col in binfo_cols:
    #        df[col] = (
    #            df["src_ip"]
    #            .astype(str)
    #            .str.strip()
    #            .map(lambda ip: lookup.get(ip, {}).get(col, 0.0))
    #        )
#
    #    df["event_type"] = constants.EVENT_TYPES["network_traffic"]
    #    df["label"] = 1 if is_malicious else 0
#
    #    return df[constants.FEATURES + ["label"]]
    def create_features(
    self,
    df: pd.DataFrame,
    is_malicious: bool
) -> pd.DataFrame:

        # Keep the original raw data unchanged.
        df = df.copy()
    
        # Create a separate feature DataFrame.
        X = pd.DataFrame(
            0.0,
            index=df.index,
            columns=constants.FEATURES
        )
    
        # Extract count from the ORIGINAL data.
        raw_count = pd.to_numeric(
            df["count"], errors="coerce"
        ).fillna(0).clip(lower=0)
    
        X["count"] = np.log1p(raw_count)
    
        # Extract URL from the ORIGINAL data.
        X["url"] = df["url"].apply(self._url_hash)
    
        # Load binary information associated with source IP.
        if is_malicious:
            lookup = self._load_binary_info_by_src_ip(
                constants.MALICIOUS_CONNECTION_FILE,
                constants.MALICIOUS_BINARY_INFO_FILE,
                malicious_format=True
            )
        else:
            lookup = self._load_binary_info_by_src_ip(
                constants.BENIGN_CONNECTION_FILE,
                constants.BENIGN_BINARY_INFO_FILE,
                malicious_format=False
            )
    
        # Add binary information features.
        binfo_cols = [
            f"binfo_{c}"
            for c in (
                constants.BINFO_NUMERIC_COLS
                + constants.BINFO_TEXT_COLS
            )
        ]
    
        src_ips = df["src_ip"].astype(str).str.strip()
    
        for col in binfo_cols:
            X[col] = src_ips.map(
                lambda ip: lookup.get(ip, {}).get(col, 0.0)
            )
    
        # Network traffic event type.
        X["event_type"] = constants.EVENT_TYPES["network_traffic"]
    
        # Binary classification: benign=0, malicious=1.
        X["label"] = 1 if is_malicious else 0
    
        return X[constants.FEATURES + ["label"]]
    

    def create_event_features(self, df, event_type):

        X = pd.DataFrame(
            0.0,
            index=df.index,
            columns=constants.FEATURES
        )

        X["event_type"] = constants.EVENT_TYPES[event_type]

        if event_type == "network_traffic":

            X["count"] = np.log1p(
                pd.to_numeric(df["count"], errors="coerce").fillna(0)
            )

            X["url"] = df["url"].apply(self._url_hash)

            return X

        if event_type == "binary_information":

            for c in constants.BINFO_NUMERIC_COLS:
                if c in df.columns:
                    X[f"binfo_{c}"] = pd.to_numeric(
                        df[c], errors="coerce"
                    ).fillna(0)

            for c in constants.BINFO_TEXT_COLS:
                if c in df.columns:
                    X[f"binfo_{c}"] = df[c].apply(self._url_hash)

            #if "count" in df.columns:
            #    X["binfo_count"] = pd.to_numeric(
            #        df["count"], errors="coerce"
            #    ).fillna(0)

            return X

        if event_type == "command":

            X["count"] = np.log1p(
                pd.to_numeric(df["count"], errors="coerce").fillna(0)
            )

            X["command_hash"] = df["command"].apply(self._url_hash)

            return X

        if event_type == "login_attempt":

            X["count"] = np.log1p(
                pd.to_numeric(df["count"], errors="coerce").fillna(0)
            )

            X["username_hash"] = df["username:password"].apply(
                self._url_hash
            )

            X["protocol_hash"] = df["protocol"].apply(
                self._url_hash
            )

            X["successful"] = (
                df["successful"]
                .astype(str)
                .str.lower()
                .map({"true": 1.0, "false": 0.0})
                .fillna(0.0)
            )

            return X

        raise ValueError(f"Unknown event type: {event_type}")


    def scale_features(self, X: np.ndarray) -> np.ndarray:

        X = X.astype(np.float32).copy()
        idx = {name: i for i, name in enumerate(constants.FEATURES)}

        X[:,idx["count"]] = X[:,idx["count"]] / 20.0
        #X[:,idx["url"]] = X[:,idx["url"]] / 255.0
        for col in [
            "url",
            "command_hash",
            "username_hash",
            "protocol_hash",
        ]:
            X[:, idx[col]] /= 255.0

        for c in constants.BINFO_TEXT_COLS:
            X[:,idx[f"binfo_{c}"]] /= 255.0  # hashed -> [0,255]
        for c in constants.BINFO_NUMERIC_COLS:
            col = idx[f"binfo_{c}"]
            X[:,col] = np.log1p(np.clip(X[:,col], 0, None)) / 20.0
        
        X[:, idx["event_type"]] /= 4.0

        return X

    def partition(self, df: pd.DataFrame, partition_id: int, num_partitions: int) -> pd.DataFrame:

        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        shard_indices = np.array_split(np.arange(len(shuffled)), num_partitions)

        return shuffled.iloc[shard_indices[partition_id]].reset_index(drop=True)

    def build_splits(self, benign: pd.DataFrame, malicious: pd.DataFrame):

        benign_train, benign_test = train_test_split(benign, test_size=0.2, random_state=42)
        benign_train_ae, benign_val_ae = train_test_split( benign_train, test_size=0.2, random_state=42)
        benign_train_ae, benign_for_classifier = train_test_split( benign_train_ae, test_size=0.2, random_state=42)

        if len(malicious) >= 2:
            mal_train, mal_test = train_test_split(malicious, test_size=0.2, random_state=42)
        else:
            mal_train = malicious.copy()
            mal_test = malicious.iloc[0:0].copy()
        phase_malicious_test = mal_test["phase"].to_numpy()

        X_train = self.scale_features(benign_train_ae[constants.FEATURES].to_numpy())
        X_Validation = self.scale_features(benign_val_ae[constants.FEATURES].to_numpy())

        #X_train_classifier = self.scale_features(pd.concat([benign_for_classifier, mal_train], ignore_index=True)[constants.FEATURES].to_numpy())

        X_train_classifier = self.scale_features(
            pd.concat(
                [benign_for_classifier, mal_train],
                ignore_index=True
            )[constants.FEATURES].to_numpy()
        )

        y_class = np.concatenate([
            np.zeros(len(benign_for_classifier), dtype=int),
            np.ones(len(mal_train), dtype=int)
        ])

        #X_test_full = scale_features(pd.concat([benign_test, mal_test], ignore_index=True)[FEATURES].to_numpy())
        X_benign_test = self.scale_features(benign_test[constants.FEATURES].to_numpy())

        #y_true = np.concatenate([np.zeros(len(benign_test)), np.ones(len(mal_test))])

        X_malicious_test = self.scale_features(mal_test[constants.FEATURES].to_numpy())

        trainloader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=64, shuffle=True)

        validaton_loader = DataLoader(TensorDataset(torch.FloatTensor(X_Validation)), batch_size=64, shuffle=True)

        assert set(benign_train.index).isdisjoint(benign_test.index)
        assert set(benign_train_ae.index).isdisjoint(benign_val_ae.index)
        assert set(benign_for_classifier.index).isdisjoint(benign_test.index)

        if len(malicious) >= 2:
            assert set(mal_train.index).isdisjoint(mal_test.index)

        #return trainloader, validaton_loader, X_test_full, X_Validation, y_true, X_train_classifier, y_class
        return (
            trainloader, 
            validaton_loader,
            X_benign_test, 
            X_malicious_test, 
            X_Validation,
            X_train_classifier, 
            y_class,
            phase_malicious_test,
        )

    def load_device_dataset(self, partition_id: int):

        device_name = constants.DEVICE_NAMES[partition_id]
        scenario_names = constants.DEVICE_SCENARIOS[device_name]

        dfs = []
        for scenario in scenario_names:
            path = os.path.join(constants.SCENARIOS_DIR, scenario, "network_traffic.csv")
            dfs.append(pd.read_csv(path))
        benign_raw = pd.concat(dfs, ignore_index=True)

        #malicious_raw = pd.read_csv(constants.MALICIOUS_FILE)

        benign = self.create_features(benign_raw, is_malicious=False).drop(columns=["label"]).reset_index(drop=True)
        #malicious = self.create_features(malicious_raw, is_malicious=True).drop(columns=["label"]).reset_index(drop=True)

        if device_name == "laptop":

            commands_raw = pd.read_csv(
                constants.BENIGN_BINARY_COMMANDS_FILE
            )

            login_attempts_raw = pd.read_csv(
                constants.BENIGN_LOGIN_ATTEMPTS_FILE
            )

            binary_info_raw = pd.read_csv(
                constants.BENIGN_BINARY_INFO_FILE
            )

            commands = self.create_event_features(
                commands_raw, "command"
            )

            login_attempts = self.create_event_features(
                login_attempts_raw, "login_attempt"
            )

            binary_info = self.create_event_features(
                binary_info_raw, "binary_information"
            )

            benign = pd.concat(
                [benign, commands, login_attempts, binary_info],
                ignore_index=True
            )


        malicious_frames = []

        malicious_network_raw = pd.read_csv(
            constants.MALICIOUS_NETWORK_TRAFFIC_FILE
        )
        malicious_frames.append(
            self.add_metadata(
                self.create_features(
                    malicious_network_raw, is_malicious=True
                ).drop(columns=["label"]),
                "distribution"
            )
        )

        malicious_commands_raw = pd.read_csv(
            constants.MALICIOUS_COMMANDS_FILE
        )
        malicious_frames.append(
            self.add_metadata(
                self.create_event_features(
                    malicious_commands_raw, "command"
                ),
                "distribution"
            )
        )

        malicious_logins_raw = pd.read_csv(
            constants.MALICIOUS_LOGIN_ATTEMPTS_FILE
        )
        malicious_frames.append(
            self.add_metadata(
                self.create_event_features(
                    malicious_logins_raw, "login_attempt"
                ),
                "scan"
            )
        )

        malicious_binary_raw = pd.read_csv(
            constants.MALICIOUS_BINARY_INFO_FILE
        )
        malicious_frames.append(
            self.add_metadata(
                self.create_event_features(
                    malicious_binary_raw, "binary_information"
                ),
                "distribution"
            )
        )

        malicious = pd.concat(
            malicious_frames,
            ignore_index=True
        )

        return self.build_splits(benign, malicious)

    def load_centralized_dataset(self):

        all_scenarios = sorted({
            scenario
            for scenarios in constants.DEVICE_SCENARIOS.values()
            for scenario in scenarios
        })
        dfs = []
        mal_dir = constants.MALICIOUS_DATASET_DIR

        for scenario in all_scenarios:
            path = os.path.join(
                constants.SCENARIOS_DIR,
                scenario,
                "network_traffic.csv"
            )
            dfs.append(pd.read_csv(path))

        benign_raw = pd.concat(dfs, ignore_index=True)

        constants.MALICIOUS_CONNECTION_FILE = os.path.join(
            mal_dir, "network_binary_connection.csv"
        )

        constants.MALICIOUS_BINARY_INFO_FILE = os.path.join(
            mal_dir, "binary_information.csv"
        )

        benign = self.create_features(
            benign_raw,
            is_malicious=False
        ).drop(columns=["label"]).reset_index(drop=True)

        
        mal_network = self.create_features(
            pd.read_csv(os.path.join(mal_dir, "network_traffic.csv")),
            is_malicious=True,
        ).drop(columns=["label"])

        mal_network = self.add_metadata(
            mal_network, "distribution"
        )

        mal_commands = self.create_event_features(
            pd.read_csv(os.path.join(mal_dir, "commands.csv")),
            "command",
        )
        mal_commands = self.add_metadata(mal_commands, "distribution")

        mal_logins = self.create_event_features(
            pd.read_csv(os.path.join(mal_dir, "login_attempts.csv")),
            "login_attempt",
        )
        mal_logins = self.add_metadata(mal_logins, "scan")

        mal_binary = self.create_event_features(
            pd.read_csv(os.path.join(mal_dir, "binary_information.csv")),
            "binary_information",
        )
        mal_binary = self.add_metadata(mal_binary, "distribution")

        commands_raw = pd.read_csv(
            constants.BENIGN_BINARY_COMMANDS_FILE
        )

        login_attempts_raw = pd.read_csv(
            constants.BENIGN_LOGIN_ATTEMPTS_FILE
        )

        binary_info_raw = pd.read_csv(
            constants.BENIGN_BINARY_INFO_FILE
        )

        commands = self.create_event_features(
            commands_raw, "command"
        )

        login_attempts = self.create_event_features(
            login_attempts_raw, "login_attempt"
        )

        binary_info = self.create_event_features(
            binary_info_raw, "binary_information"
        )

        benign = pd.concat(
            [benign, commands, login_attempts, binary_info],
            ignore_index=True
        )

        malicious = pd.concat(
            [mal_network, mal_commands, mal_logins, mal_binary],
            ignore_index=True
        )

        return self.build_splits(benign, malicious)
