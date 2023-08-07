import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay
from typing import Dict, Optional, Callable
from matplotlib.gridspec import GridSpec


class ModelVisualizer:
    """
    Class to visualize the results of the model evaluation.
    """

    @staticmethod
    def _create_save_plots(path: Optional[str], calibrate: bool, name: str, show_plot: bool, plot_function: Callable):
        """
        Create and save plots.
        
        Args:
            path (Optional[str]): Path to save the plots.
            calibrate (bool): If the model is calibrated.
            name (str): Name of the plot.
            show_plot (bool): If the plot is shown.
            plot_function (Callable): Function to create the plot.

        Returns:
            None
        """
        plot_function()
        if path:
            if calibrate:
                plt.savefig(f'{path}/{name}_calibrated.png')
            else:
                plt.savefig(f'{path}/{name}.png')
        if show_plot:
            plt.show()
        else:
            plt.close()
        

    @staticmethod
    def plot_curves(precision: np.ndarray, recall: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, pr_auc: float, path: Optional[str] = None, calibrate: bool = False, model_name: str = 'model', show_plot: bool = False):
        """
        Create a plot with the ROC and Precision-Recall curves.
        
        Args:
            precision (np.ndarray): Precision values.
            recall (np.ndarray): Recall values.
            fpr (np.ndarray): False positive rate values.
            tpr (np.ndarray): True positive rate values.
            roc_auc (float): Area under the ROC curve.
            pr_auc (float): Area under the Precision-Recall curve.
            path (Optional[str]): Path to save the plot.
            calibrate (bool): If the model is calibrated.
            model_name (str): Name of the model.
            show_plot (bool): If the plot is shown.

        Returns:
            None
        """

        def plot():
            # create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # plot precision-recall curve
            ax1.plot(recall, precision, label=f'Precision-Recall curve (area = {round(pr_auc, 2)})')
            ax1.set_xlabel('Recall')
            ax1.set_ylabel('Precision')
            ax1.set_title('Precision-Recall curve')
            ax1.legend(loc="lower left")

            # plot ROC curve
            ax2.plot(fpr, tpr, label=f'ROC curve (area = {round(roc_auc,2)})')
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_title('Receiver Operating Characteristic curve')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.legend(loc="lower right")


        ModelVisualizer._create_save_plots(path, calibrate, f'pr_roc_curves_{model_name}', show_plot=show_plot, plot_function=plot)

    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, n_bins: int = 10, path: Optional[str] = None, calibrate: bool = False, model_name: str = 'model', show_plot: bool = False):
        """
        Plot calibration curve to evaluate the calibration of a classifier.

        Args:
            y_true (np.ndarray): True labels.
            y_pred_probs (np.ndarray): Predicted probabilities.
            n_bins (int): Number of bins to discretize the predicted probabilities.
            path (Optional[str]): Path to save the plot.
            calibrate (bool): If the model is calibrated.
            model_name (str): Name of the model.
            show_plot (bool): If the plot is shown.

        Returns:
            None

        """

        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_probs, n_bins=n_bins)

        def plot():
            plt.figure(figsize=(10, 10))
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.plot(mean_predicted_value, fraction_of_positives, "s-")
            plt.ylabel("Fraction of positives")
            plt.xlabel("Mean predicted value")
            plt.ylim([-0.05, 1.05])
            plt.title("Calibration curve")

        ModelVisualizer._create_save_plots(path, calibrate, f'calibration_curve_{model_name}', show_plot=show_plot, plot_function=plot)

    @staticmethod
    def plot_confusion_matrix(y_test: np.ndarray, y_pred_binary: np.ndarray, path: Optional[str] = None, calibrate: bool = False, model_name: str = 'model', show_plot: bool = False):
        """
        Plot confusion matrix.

        Args:
            y_test (np.ndarray): True labels.
            y_pred_binary (np.ndarray): Predicted labels.
            path (Optional[str]): Path to save the plot.
            calibrate (bool): If the model is calibrated.
            model_name (str): Name of the model.
            show_plot (bool): If the plot is shown.

        Returns:
            None
        """
        cnf_matrix = confusion_matrix(y_test, y_pred_binary)
        _row_sums = cnf_matrix.sum(axis=1)[:, np.newaxis]
        _row_sums[_row_sums == 0] = 1 # avoid division by zero    
        cnf_matrix_norm = cnf_matrix.astype('float') / _row_sums

        def plot():
            plt.figure(figsize=(7, 7))
            sns.heatmap(cnf_matrix_norm, annot=True, fmt=".2f", cmap='Blues', square=True, cbar=False,
                        xticklabels=['Not churn', 'Churn'], yticklabels=['Not churn', 'Churn'])
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Confusion matrix')

        ModelVisualizer._create_save_plots(path, calibrate, f'confusion_matrix_{model_name}', show_plot=show_plot, plot_function=plot)

    @staticmethod
    def plot_summary_curves(metrics_dict:Dict, path: Optional[str] = None, calibrate: bool = False, show_plot: bool = False, suffix: str = ''):
        """
        Plot ROC and PR curves for all classifiers to compare them.
        
        Args:
            metrics_dict (Dict): Dictionary with the metrics to plot.
            path (Optional[str]): Path to save the plot.
            calibrate (bool): If True, the plot is saved with suffix '_calibrated'.
            show_plot (bool): If True, the plot is shown.
        Returns:
            None
        """

        def plot():
            # create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Plot ROC curve for all classifiers
            for classifier_name, metrics in metrics_dict.items():
                ax1.plot(metrics['fpr'], metrics['tpr'], label=f'{classifier_name} (AUC = {metrics["roc_auc"]:.2f})')
            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('Receiver Operating Characteristic')
            ax1.legend(loc="lower right")

            # Plot PR curve for all classifiers
            for classifier_name, metrics in metrics_dict.items():
                ax2.plot(metrics['recall'], metrics['precision'], label=f'{classifier_name} (AUC = {metrics["pr_auc"]:.2f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend(loc="upper right")
        if suffix != '':
            ModelVisualizer._create_save_plots(path, calibrate, f'summary_curves_{suffix}', plot_function=plot, show_plot=show_plot)
        else:
            ModelVisualizer._create_save_plots(path, calibrate, f'summary_curves', plot_function=plot, show_plot=show_plot)

    @staticmethod
    def plot_comp_calibration(y_test, y_pred,y_pred_calib, classifier_name, path: Optional[str] = None, calibrate: bool = False, show_plot: bool = False):
        """
        Plot calibration curve to evaluate the calibration of a classifier.

        Args:
            y_test (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels for uncalibrated models.
            y_pred_calib (np.ndarray): Predicted labels for calibrated models.
            classifier_name (str): Name of the classifier.
            path (Optional[str]): Path to save the plot.
            calibrate (bool): If the model is calibrated.
            show_plot (bool): If the plot is to be shown.

        Returns:
            None
        """
        # plot calibration curve
        print(classifier_name)

        def plot():
            fig = plt.figure(figsize=(10, 10))
            gs = GridSpec(2, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            CalibrationDisplay.from_predictions(y_test, y_pred, ax=ax1, n_bins=20, color='red', label=f'{classifier_name}_uncalibrated')
            CalibrationDisplay.from_predictions(y_test, y_pred_calib, ax=ax1, n_bins=20, color='blue', label=f'{classifier_name}_calibrated')
            ax1.legend()
            ax1.grid()
            ax1.set_title(f'Calibration curve for {classifier_name}')
            # plot histogram
            ax2.hist(y_pred, bins=20, alpha=0.5, color='red', label=f'{classifier_name}_uncalibrated')
            ax2.hist(y_pred_calib, bins=20, alpha=0.5, color='blue', label=f'{classifier_name}_calibrated')
            ax2.legend()
            ax2.grid()
            ax2.set_title(f'Prediction histogram for {classifier_name}')
            plt.tight_layout()


        ModelVisualizer._create_save_plots(path, calibrate, f'calibration_comparison_{classifier_name}', plot_function=plot, show_plot=show_plot)

