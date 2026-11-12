import os

import numpy as np
import psutil
from sklearn.metrics import confusion_matrix

class EvaluationMetrics():
    def intrusion_detection_capability(self, y_true, y_pred):
        
        """
        Calculation of Intrusion Detection Capability (C_ID) metric as per the paper:
        G. Gu, P. Fogla, D. Dagon, W. Lee, and B. Skoric, Measuring intrusion detection
        capability: an information-theoretic approach. Mar. 2006, Pages: 101. doi:
        10.1145/1128817.1128834
        """

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print("TN", tn, "FP", fp,"FN", fn,"TP", tp)
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

    def print_memory_usage(self):
    
        process = psutil.Process(os.getpid())
    
        # Convert bytes to Megabytes
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Current RAM usage: {mem_mb:.2f} MB")