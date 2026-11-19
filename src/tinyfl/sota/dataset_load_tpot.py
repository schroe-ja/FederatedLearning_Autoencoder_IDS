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

from sklearn.model_selection import train_test_split, GroupShuffleSplit

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

        random_state = 42

        benign = benign.copy().reset_index(drop=True)
        malicious = malicious.copy().reset_index(drop=True)

        if "scenario_group" not in benign.columns:
            raise ValueError("Benign data must contain scenario_group")

        benign_groups = benign["scenario_group"].fillna("unknown").astype(str)

        if benign_groups.nunique() >= 2:
        
            held_out_scenarios = {"speaker_power"}

            benign_test = benign[
                benign["scenario_group"].isin(held_out_scenarios)
            ].copy()

            benign_train = benign[
                ~benign["scenario_group"].isin(held_out_scenarios)
            ].copy()

        else:
            if len(benign) < 2:
                raise ValueError("Not enough benign samples to split")

            benign_train, benign_test = train_test_split(
                benign,
                test_size=0.2,
                random_state=random_state
            )

        benign_train_ae, benign_val_ae = train_test_split(
            benign_train,
            test_size=0.2,
            random_state=random_state
        )

        benign_train_ae, benign_for_classifier = train_test_split(
            benign_train_ae,
            test_size=0.2,
            random_state=random_state
        )

        if "phase" not in malicious.columns:
            raise ValueError("Malicious data must contain phase")

        if len(malicious) >= 2:

            if "attack_group" in malicious.columns:
                attack_groups = malicious["attack_group"].astype(str)

                if attack_groups.nunique() >= 2:
                    splitter = GroupShuffleSplit(
                        n_splits=1,
                        test_size=0.2,
                        random_state=random_state
                    )

                    train_idx, test_idx = next(
                        splitter.split(malicious, groups=attack_groups)
                    )

                    mal_train = malicious.iloc[train_idx].copy()
                    mal_test = malicious.iloc[test_idx].copy()

                else:
                    mal_train, mal_test = train_test_split(
                        malicious,
                        test_size=0.2,
                        random_state=random_state
                    )

            else:
                # No reliable attack-group metadata
                mal_train, mal_test = train_test_split(
                    malicious,
                    test_size=0.2,
                    random_state=random_state
                )

        else:
            mal_train = malicious.copy()
            mal_test = malicious.iloc[0:0].copy()

        phase_malicious_test = mal_test["phase"].to_numpy()

        X_train = self.scale_features(
            benign_train_ae[constants.FEATURES].to_numpy()
        )

        X_Validation = self.scale_features(
            benign_val_ae[constants.FEATURES].to_numpy()
        )

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

        X_benign_test = self.scale_features(
            benign_test[constants.FEATURES].to_numpy()
        )

        X_malicious_test = self.scale_features(
            mal_test[constants.FEATURES].to_numpy()
        )

        trainloader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=64,
            shuffle=True
        )

        validaton_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_Validation)),
            batch_size=64,
            shuffle=False
        )

        if benign_groups.nunique() >= 2:
            assert set(
                benign_train["scenario_group"]
            ).isdisjoint(
                set(benign_test["scenario_group"])
            )

        if (
            "attack_group" in malicious.columns
            and malicious["attack_group"].nunique() >= 2
        ):
            assert set(
                mal_train["attack_group"]
            ).isdisjoint(
                set(mal_test["attack_group"])
            )

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
    
    def build_client_ae_splits(self, benign: pd.DataFrame):
        benign = benign.copy().reset_index(drop=True)

        benign_train, benign_val = train_test_split(
            benign,
            test_size=0.2,
            random_state=42
        )

        X_train = self.scale_features(
            benign_train[constants.FEATURES].to_numpy()
        )

        X_val = self.scale_features(
            benign_val[constants.FEATURES].to_numpy()
        )

        trainloader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=64,
            shuffle=True
        )

        validaton_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=64,
            shuffle=False
        )

        return trainloader, validaton_loader

    def load_device_dataset(self, device_name: str):

        benign_frames = []

        for scenario in constants.DEVICE_SCENARIOS[device_name]:

            csv_path = os.path.join(
                constants.SCENARIOS_DIR,
                scenario,
                "network_traffic.csv"
            )

            df = pd.read_csv(csv_path)

            features = self.create_features(
                df,
                is_malicious=False
            )

            features["scenario_group"] = scenario
            features["device_group"] = device_name
            features["phase"] = "benign"

            benign_frames.append(features)

        benign = pd.concat(
            benign_frames,
            ignore_index=True
        )
        # malicious = self.create_features(malicious_raw, is_malicious=True).drop(columns=["label"]).reset_index(drop=True)

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

            for event_df, event_name in [
                (commands, "commands"),
                (login_attempts, "login_attempts"),
                (binary_info, "binary_information"),
            ]:
                event_df["scenario_group"] = f"laptop_{event_name}"
                event_df["device_group"] = "laptop"
                event_df["phase"] = "benign"

            benign = pd.concat(
                [benign, commands, login_attempts, binary_info],
                ignore_index=True
            )

        return self.build_client_ae_splits(benign)

    def load_centralized_dataset(self):

        all_scenarios = sorted({
            scenario
            for scenarios in constants.DEVICE_SCENARIOS.values()
            for scenario in scenarios
        })

        mal_dir = constants.MALICIOUS_DATASET_DIR

        benign_frames = []

        for scenario in all_scenarios:
            path = os.path.join(
                constants.SCENARIOS_DIR,
                scenario,
                "network_traffic.csv"
            )

            raw = pd.read_csv(path)

            features = self.create_features(
                raw,
                is_malicious=False
            ).drop(columns=["label"])

            features["scenario_group"] = scenario
            features["phase"] = "benign"

            benign_frames.append(features)

        benign = pd.concat(
            benign_frames,
            ignore_index=True
        )

        constants.MALICIOUS_CONNECTION_FILE = os.path.join(
            mal_dir, "network_binary_connection.csv"
        )

        constants.MALICIOUS_BINARY_INFO_FILE = os.path.join(
            mal_dir, "binary_information.csv"
        )

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

        commands["scenario_group"] = "benign_commands"
        commands["phase"] = "benign"

        login_attempts["scenario_group"] = "benign_login_attempts"
        login_attempts["phase"] = "benign"

        binary_info["scenario_group"] = "benign_binary_information"
        binary_info["phase"] = "benign"

        benign = pd.concat(
            [benign, commands, login_attempts, binary_info],
            ignore_index=True
        )

        malicious = pd.concat(
            [mal_network, mal_commands, mal_logins, mal_binary],
            ignore_index=True
        )

        return self.build_splits(benign, malicious)
