
import os
import numpy as np
import psutil

from typing import Any

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score,
)

from flwr.app import MetricRecord


# Self written evaluation metrics
class EvaluationMetrics:
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_percentile: np.ndarray,
        y_proba: np.ndarray,
        errors_full: np.ndarray,
        st: float,
        et: float,
        datasets: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
    
        self.y_true = np.asarray(y_true).ravel().astype(int)
        self.y_pred = np.asarray(y_pred).ravel().astype(int)
        self.y_pred_percentile = np.asarray(
            y_pred_percentile
        ).ravel().astype(int)
        self.y_proba = np.asarray(y_proba).ravel()
        self.errors_full = np.asarray(errors_full).ravel()
        self.st = st
        self.et = et
        self.datasets = datasets if datasets is not None else {}

    def _conf_mat(self, percentile: bool = True) -> np.ndarray:
        if percentile:
            return confusion_matrix(
                self.y_true,
                self.y_pred_percentile,
                labels=[0, 1],
            )

        return confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=[0, 1],
        )

    def _conf_mat_ravel(
        self,
        percentile: bool = True,
    ) -> dict[str, int]:

        tn, fp, fn, tp = self._conf_mat(percentile).ravel()
        suffix = "ae" if percentile else "dt"

        return {
            f"tn_{suffix}": int(tn),
            f"fn_{suffix}": int(fn),
            f"tp_{suffix}": int(tp),
            f"fp_{suffix}": int(fp),
        }

    @staticmethod
    def _merge_dicts(
        *dict_args: dict[str, Any],
    ) -> dict[str, Any]:
        result = {}

        for dictionary in dict_args:
            result.update(dictionary)

        return result

    @staticmethod
    def _safe_div(
        numerator: int | float,
        denominator: int | float,
    ) -> float | None:
        if denominator == 0:
            return None

        return float(numerator / denominator)

    def intrusion_detection_capability(
        self,
        percentile: bool = True,
    ) -> float | None:

        """
        Calculation of Intrusion Detection Capability (C_ID) metric as per the paper:
        G. Gu, P. Fogla, D. Dagon, W. Lee, and B. Skoric, Measuring intrusion detection
        capability: an information-theoretic approach. Mar. 2006, Pages: 101. doi:
        10.1145/1128817.1128834
        """

        cm = self._conf_mat_ravel(percentile)
        suffix = "ae" if percentile else "dt"

        tn = cm[f"tn_{suffix}"]
        fn = cm[f"fn_{suffix}"]
        tp = cm[f"tp_{suffix}"]
        fp = cm[f"fp_{suffix}"]

        N = tp + tn + fp + fn

        if N == 0:
            return None

        # probabilities
        p_tp = tp / N
        p_tn = tn / N
        p_fp = fp / N
        p_fn = fn / N

        p_attack = (tp + fn) / N
        p_normal = (tn + fp) / N

        p_pred_attack = (tp + fp) / N
        p_pred_normal = (tn + fn) / N

        # IDC requires both actual classes
        if p_attack == 0 or p_normal == 0:
            return None

        # Mutual Information
        terms = []

        if p_tp > 0:
            terms.append(
                p_tp * np.log2(p_tp / (p_attack * p_pred_attack))
            )

        if p_fn > 0:
            terms.append(
                p_fn * np.log2(p_fn / (p_attack * p_pred_normal))
            )

        if p_fp > 0:
            terms.append(
                p_fp * np.log2(p_fp / (p_normal * p_pred_attack))
            )

        if p_tn > 0:
            terms.append(
                p_tn * np.log2(p_tn / (p_normal * p_pred_normal))
            )

        mutual_information = sum(terms)

        # Entropy of actual classes
        Hx = 0

        if p_attack > 0:
            Hx -= p_attack * np.log2(p_attack)

        if p_normal > 0:
            Hx -= p_normal * np.log2(p_normal)

        # IDC
        IDC = mutual_information / Hx if Hx != 0 else None

        return float(IDC) if IDC is not None else None

    @staticmethod
    def print_memory_usage() -> dict[str, float]:

        process = psutil.Process(os.getpid())

        ram = psutil.virtual_memory()
        mem_mb = process.memory_info().rss / (1024 * 1024) # Convert bytes to Megabytes
        cpu_usage = psutil.cpu_percent(interval=1)

        return {
            "CPU usage (%)": float("{:.2f}".format(cpu_usage)),
            "RAM usage (%)": float("{:.2f}".format(ram.percent)),
            "RAM usage (MB)": float("{:.2f}".format(mem_mb)),
        }

    def fnr_fpr(self) -> dict[str, float]:

        cm_ae = self._conf_mat()
        cm_dt = self._conf_mat(False)

        print(f"\nConfusion Matrix Classify:")
        print(cm_dt)

        # Use Confusion Matrix to calculate fnr and fpr for Percentile and DT
        tn_ae, fp_ae, fn_ae, tp_ae = cm_ae.ravel()
        tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

        fpr_ae = fp_ae / (tn_ae + fp_ae)
        fnr_ae = fn_ae / (tp_ae + fn_ae)

        fpr_dt = fp_dt / (tn_dt + fp_dt)
        fnr_dt = fn_dt / (tp_dt + fn_dt)

        return {
            "FPR AE": float("{:.4f}".format(fpr_ae)),
            "FNR AE": float("{:.4f}".format(fnr_ae)),
            "FPR DT": float("{:.4f}".format(fpr_dt)),
            "FNR DT": float("{:.4f}".format(fnr_dt)),
        }

    def avg_pre(self) -> dict[str, float]:

        aps_ae = average_precision_score(
            self.y_true,
            self.errors_full,
        )
        aps_dt = average_precision_score(
            self.y_true,
            self.y_proba,
        )

        return {
            "AP AE": float("{:.4f}".format(aps_ae)),
            "AP DT": float("{:.4f}".format(aps_dt)),
        }

    def roc_auc(self) -> dict[str, float]:

        ras_ae = roc_auc_score(
            self.y_true,
            self.errors_full,
        )
        ras_dt = roc_auc_score(
            self.y_true,
            self.y_proba,
        )

        return {
            "ROC AUC AE": float("{:.4f}".format(ras_ae)),
            "ROC AUC DT": float("{:.4f}".format(ras_dt)),
        }

    def mcc(self) -> dict[str, float]:

        mcc_ae = matthews_corrcoef(
            self.y_true,
            self.y_pred_percentile,
        )
        mcc_dt = matthews_corrcoef(
            self.y_true,
            self.y_pred,
        )

        return {
            "MCC AE": float("{:.4f}".format(mcc_ae)),
            "MCC DT": float("{:.4f}".format(mcc_dt)),
        }

    def accuracy(self) -> dict[str, float]:

        acc_ae = accuracy_score(
            self.y_true,
            self.y_pred_percentile,
        )
        acc_dt = accuracy_score(
            self.y_true,
            self.y_pred,
        )

        return {
            "Accuracy AE": float("{:.4f}".format(acc_ae)),
            "Accuracy DT": float("{:.4f}".format(acc_dt)),
        }

    def fdr(self) -> dict[str, float]:

        cm_ae = self._conf_mat()
        cm_dt = self._conf_mat(percentile=False)

        tn_ae, fp_ae, fn_ae, tp_ae = cm_ae.ravel()
        tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

        fdr_ae = fp_ae / (fp_ae + tp_ae) if (fp_ae + tp_ae) > 0 else 0
        fdr_dt = fp_dt / (fp_dt + tp_dt) if (fp_dt + tp_dt) > 0 else 0

        return {
            "FDR AE": float("{:.4f}".format(fdr_ae)),
            "FDR DT": float("{:.4f}".format(fdr_dt)),
        }

    def f1_sc(self) -> dict[str, float]:

        f1_ae = f1_score(
            self.y_true,
            self.y_pred_percentile,
        )
        f1_dt = f1_score(
            self.y_true,
            self.y_pred,
        )

        return {
            "F1-Score AE": float("{:.4f}".format(f1_ae)),
            "F1-Score DT": float("{:.4f}".format(f1_dt)),
        }

    def test_time(self) -> float:

        testing_time = self.et - self.st

        return float(testing_time)

    @staticmethod
    def _dataset_model_metrics(
        name: str,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, int | float]:

        y_true = np.asarray(y_true).ravel().astype(int)
        y_pred = np.asarray(y_pred).ravel().astype(int)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"{name}: true labels and predictions have different lengths"
            )

        if len(y_true) == 0:
            raise ValueError(f"{name}: empty evaluation dataset")

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        prefix = f"{name}/"

        result = {
            f"{prefix}Samples {model_name}": int(len(y_true)),
            f"{prefix}Correct {model_name}": int(np.sum(y_true == y_pred)),
            f"{prefix}Incorrect {model_name}": int(np.sum(y_true != y_pred)),
            f"{prefix}Accuracy {model_name}": float(
                accuracy_score(y_true, y_pred)
            ),
        }

        if np.any(y_true == 0):
            result.update({
                f"{prefix}TN {model_name}": int(tn),
                f"{prefix}FP {model_name}": int(fp),
                f"{prefix}FPR {model_name}": float(
                    fp / (tn + fp)
                ),
                f"{prefix}Specificity {model_name}": float(
                    tn / (tn + fp)
                ),
            })

        if np.any(y_true == 1):
            result.update({
                f"{prefix}TP {model_name}": int(tp),
                f"{prefix}FN {model_name}": int(fn),
                f"{prefix}Recall {model_name}": float(
                    tp / (tp + fn)
                ),
                f"{prefix}TPR {model_name}": float(
                    tp / (tp + fn)
                ),
                f"{prefix}FNR {model_name}": float(
                    fn / (tp + fn)
                ),
            })

        return result

    def dataset_metrics(self) -> dict[str, int | float]:

        metrics = {}

        for name, dataset in self.datasets.items():
            y_true = dataset["y_true"]

            metrics.update(
                self._dataset_model_metrics(
                    name,
                    "AE",
                    y_true,
                    dataset["ae_pred"],
                )
            )

            metrics.update(
                self._dataset_model_metrics(
                    name,
                    "DT",
                    y_true,
                    dataset["dt_pred"],
                )
            )

            print(f"\n========== {name.upper()} ==========")

            for model_name in ("AE", "DT"):
                print(f"\n{model_name}:")

                prefix = f"{name}/"
                suffix = f" {model_name}"

                for key, value in metrics.items():
                    if key.startswith(prefix) and key.endswith(suffix):
                        metric_name = key[len(prefix):-len(suffix)]

                        if metric_name in {
                            "Accuracy", "Recall", "TPR",
                            "FNR", "FPR", "Specificity",
                        }:
                            print(f"  {metric_name}: {value:.2%}")
                        else:
                            print(f"  {metric_name}: {value}")

        return metrics

    def metrics(self) -> MetricRecord:

        metrics = self._merge_dicts(
            self.accuracy(),
            self.fnr_fpr(),
            self.fdr(),
            self.f1_sc(),
            self.mcc(),
            self.roc_auc(),
            self.avg_pre(),
            self.print_memory_usage(),
            {
                "IDC AE": float("{:.4f}".format(
                    self.intrusion_detection_capability()
                )),
                "IDC DT": float("{:.4f}".format(
                    self.intrusion_detection_capability(False)
                )),
                "Evaluation Time": float(
                    "{:.4f}".format(self.test_time())
                ),
            },
            self.dataset_metrics(),
        )

        return MetricRecord(metrics)
