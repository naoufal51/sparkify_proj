from joblib import load
import numpy as np
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score, auc,
                             f1_score, precision_score, recall_score, brier_score_loss)
import torch
from torch.nn import Module
from typing import Tuple, Dict, Any
from numpy import intp


class ModelEvaluatorOpt:
    """
    Class to evaluate a model.

    Attributes:
        model: Model to evaluate.

    """

    def __init__(self, model: Any):
        """
        Initialize the class.
        
        Args:
            model (Any): Model to evaluate.
        
        """
        self.model = model

    @staticmethod
    def load_model(path: str) -> Any:
        """
        Load a model from disk.
        """
        return load(path)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict churn for the test data.

        Args:
            X_test (np.ndarray): Test data

        Returns:
            y_pred (np.ndarray): Predictions

        """

        if isinstance(self.model, Module):
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_test)
            y_pred = torch.sigmoid(predictions).numpy()
        else:
            y_pred = self.model.predict_proba(X_test)[:, 1]
        return y_pred

    @staticmethod
    def optimize_threshold(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, intp]:
        """
        Optimize the threshold for the given predictions.
        It is based on the F2 score to account for the class imbalance.
        Also given that we are dealing with chun detection, we want to maximize the recall.

        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predictions

        Returns:
            best_threshold (float): Best threshold
            best_index (int): Index of the best threshold
        """

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        thresholds = np.append(thresholds, 1)
        f2_scores = (1 + 2**2) * (precision * recall) / \
            ((2**2 * precision) + recall)
        best_index = np.argmax(f2_scores)
        best_threshold = thresholds[best_index]
        return best_threshold, int(best_index)

    def calculate_metrics(self, y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Calculate the metrics for the given predictions.
        The metrics are: 
        - Accuracy
        - F1 score (weighted and not weighted)
        - Precision (weighted and not weighted)
        - Recall (weighted and not weighted)
        - Brier score loss ( Used for calibration)
        - Threshold

        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Model predictions

        Returns:
            metrics (Dict[str, float]): Dictionary with the computed metrics
            y_pred_binary (np.ndarray): Binary predictions

        """
        best_threshold, _ = self.optimize_threshold(y_test, y_pred)
        y_pred_binary = np.where(y_pred > best_threshold, 1, 0)
        brier = brier_score_loss(y_test, y_pred)
        return {
            "accuracy": float(np.mean(y_test == y_pred_binary)),
            "f1_weighted": float(f1_score(y_test, y_pred_binary, average='weighted')),
            "precision_weighted": float(precision_score(y_test, y_pred_binary, average='weighted')),
            "recall_weighted": float(recall_score(y_test, y_pred_binary, average='weighted')),
            "f1": float(f1_score(y_test, y_pred_binary)),
            "precision": float(precision_score(y_test, y_pred_binary)),
            "recall": float(recall_score(y_test, y_pred_binary)),
            "threshold": float(best_threshold),
            "brier_score": float(brier),
        }, y_pred_binary

    @staticmethod
    def calculate_auc(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Calculate the AUC for both the ROC and Precision-Recall curves.

        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predictions

        Returns:
            precision (np.ndarray): Precision values.
            recall (np.ndarray): Recall values.
            fpr (np.ndarray): False positive rate values.
            tpr (np.ndarray): True positive rate values.
            auc_roc (float): Area under the ROC curve.
            auc_pr (float): Area under the Precision-Recall curve.
        """
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_pr = float(auc(recall, precision))
        auc_roc = float(roc_auc_score(y_test, y_pred))
        return precision, recall, fpr, tpr, auc_roc, auc_pr
