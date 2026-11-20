import os
import numpy as np
import psutil

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score

from flwr.app import MetricRecord

# Self written evaluation metrics
class EvaluationMetrics:
    def __init__(self, y_true, y_pred, y_pred_percentile, y_proba, errors_full, st, et) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_percentile = y_pred_percentile
        self.y_proba = y_proba
        self.errors_full = errors_full
        self.st = st
        self.et = et
    
    def _conf_mat(self, percentile=True) -> np.ndarray:
        if percentile:
            return confusion_matrix(self.y_true, self.y_pred_percentile)

        return confusion_matrix(self.y_true, self.y_pred)
    
    def _conf_mat_ravel(self, percentile=True) -> dict[str, int]:
        
        tn, fp, fn, tp = self._conf_mat(percentile).ravel()
        suffix = "ae" if percentile else "dt"

        return {
            f"tn_{suffix}": tn,
            f"fn_{suffix}": fn,
            f"tp_{suffix}": tp,
            f"fp_{suffix}": fp
        }
    
    @staticmethod
    def _merge_dicts(*dict_args) -> dict:
        result = {}

        for dictionary in dict_args:
            result.update(dictionary)

        return result

    def intrusion_detection_capability(self, percentile=True) -> float:
        
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

    @staticmethod
    def print_memory_usage() -> None:
    
        process = psutil.Process(os.getpid())
    
        ram = psutil.virtual_memory()
        mem_mb = process.memory_info().rss / (1024 * 1024) # Convert bytes to Megabytes
        cpu_usage = psutil.cpu_percent(interval=1)

        return {
            "CPU usage (%)": float("{:.2f}".format(cpu_usage)),
            "RAM usage (%)": float("{:.2f}".format(ram.percent)),
            "RAM usage (MB)": float("{:.2f}".format(mem_mb))
        }

    def fnr_fpr(self) -> dict[str, float]:
        
        cm_ae = self._conf_mat()
        cm_dt = self._conf_mat(False)

        print(f"\nConfusion Matrix Classify:")
        print(cm_dt) 

        # Use Confusion Matrix to calculate fnr and fpr for Percentile and DT 
        fpr_ae = cm_ae[0][1] / (cm_ae[0][0] + cm_ae[0][1]) 
        fnr_ae = cm_ae[1][0] / (cm_ae[1][1] + cm_ae[1][0]) 

        fpr_dt = cm_dt[0][1] / (cm_dt[0][0] + cm_dt[0][1]) 
        fnr_dt = cm_dt[1][0] / (cm_dt[1][1] + cm_dt[1][0]) 

        return {
            "FPR AE": float("{:.4f}".format(fpr_ae)),
            "FNR AE": float("{:.4f}".format(fnr_ae)),
            "FPR DT": float("{:.4f}".format(fpr_dt)), 
            "FNR DT": float("{:.4f}".format(fnr_dt))
        }

    def avg_pre(self) -> dict[str, float]:

        aps_ae = average_precision_score(self.y_true, self.errors_full)
        aps_dt = average_precision_score(self.y_true, self.y_proba) 

        return {
            "AP AE": float("{:.4f}".format(aps_ae)),
            "AP DT": float("{:.4f}".format(aps_dt))
        }

    def roc_auc(self) -> dict[str, float]:

        ras_ae = roc_auc_score(self.y_true, self.errors_full)
        ras_dt = roc_auc_score(self.y_true, self.y_proba) 

        return {
            "ROC AUC AE": float("{:.4f}".format(ras_ae)),
            "ROC AUC DT": float("{:.4f}".format(ras_dt))
        }
    
    def mcc(self) -> dict[str, float]:
        
        mcc_ae = matthews_corrcoef(self.y_true, self.y_pred_percentile)
        mcc_dt = matthews_corrcoef(self.y_true, self.y_pred) 
        
        return {
            "MCC AE": float("{:.4f}".format(mcc_ae)),
            "MCC DT": float("{:.4f}".format(mcc_dt))
        }
    
    def accuracy(self) -> dict[str, float]:

        acc_ae = accuracy_score(self.y_true, self.y_pred_percentile)
        acc_dt = accuracy_score(self.y_true, self.y_pred)

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
            "FDR DT": float("{:.4f}".format(fdr_dt))
        }
    
    def f1_sc(self) -> dict[str, float]:

        f1_ae = f1_score(self.y_true, self.y_pred_percentile)
        f1_dt = f1_score(self.y_true, self.y_pred) 

        return {
            "F1-Score AE": float("{:.4f}".format(f1_ae)),
            "F1-Score DT": float("{:.4f}".format(f1_dt))
        }

    def test_time(self) -> float:
        
        testing_time = self.et - self.st

        return testing_time
    
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
                "IDC AE": float("{:.4f}".format(self.intrusion_detection_capability())),
                "IDC DT": float("{:.4f}".format(self.intrusion_detection_capability(False))),
                "Evaluation Time": float(("{:.4f}".format(self.test_time()))),
            },
        )   

        return MetricRecord(metrics)