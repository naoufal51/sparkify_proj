import numpy as np
import shap
import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from typing import List, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd


class ModelExplainer:

    """
    Class to conduct model explanaiton through shapeley values.
    We use shap library to compute the shapley values and visualize them.
    """

    def __init__(self, model: Any, model_name: str, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str]):
        """
        Initialize the model explainer.

        Args:
            model (Any): Model to explain. (Pytorch model, sklearn model or xgboost model)
            X_train (np.ndarray): Training data.
            X_test (np.ndarray): Test data.
            feature_names (List[str]): Feature names.
        """

        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.tree_models = [RandomForestClassifier, DecisionTreeClassifier,
                            GradientBoostingClassifier, XGBClassifier, HistGradientBoostingClassifier]
        self.X_train_pd = pd.DataFrame(
            self.X_train, columns=self.feature_names)

        self.X_test_pd = pd.DataFrame(self.X_test, columns=self.feature_names)

        self.choose_explainer()

    def choose_explainer(self):
        """
        Choose the explainer based on the model type.

        Raises:
            Exception: If the model is not trained. 
        """
        if isinstance(self.model, Module):
            self.explainer = shap.DeepExplainer(self.model, self.X_train)
        elif any([isinstance(self.model, m) for m in self.tree_models]):
            self.explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, LogisticRegression):
            self.explainer = shap.KernelExplainer(lambda x: self.model.predict(pd.DataFrame(
                x, columns=self.feature_names)), shap.sample(self.X_train_pd, 100), feature_names=self.feature_names)

    def compute_shap_values(self, n_samples=100):
        """
        Compute the shapley values for the given model.

        Args:
            n_samples (int, optional): Number of samples to compute. Defaults to 100.

        Raises:
            Exception: If the explainer has not been chosen.
        """
        if not hasattr(self, 'explainer'):
            raise Exception("Please choose an explainer first.")

        try:
            if isinstance(self.model, Module):
                # if the model is a pytorch model, we need to convert the data to torch tensors
                self.shap_values = self.explainer.shap_values(
                    torch.tensor(self.X_train[:n_samples].astype(np.float32)))
            else:
                self.shap_values = self.explainer.shap_values(
                    self.X_train_pd[:n_samples])
        except Exception as e:
            raise Exception(
                "Please choose a valid explainer, is the model trained?", e)

        return self.shap_values, self.explainer.expected_value

    def plot_shap_values(self, n_samples=100, path=None):
        """
        Plot the shapley values for the given model.

        Args:
            n_samples (int, optional): Number of samples to plot. Defaults to 100.
            path ([type], optional): Path to save the plot. Defaults to None.

        Raises:
            Exception: If the shap values have not been computed.

        """
        if not hasattr(self, 'shap_values'):
            raise Exception("Please compute the shap values first.")

        if isinstance(self.model, Module):
            shap.summary_plot(self.shap_values, torch.tensor(self.X_test_pd[:n_samples].astype(
                np.float32)), feature_names=self.feature_names, show=False, plot_type='bar')
        else:
            shap.summary_plot(
                self.shap_values, self.X_test_pd[:n_samples], feature_names=self.feature_names, show=False, plot_type='bar')
        if path is not None:
            save_path = f'{path}/{self.model_name}_shap_values.png'
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

    def plot_shap_force(self, sample_index=0, output_index=0, path=None):
        """
        Plot the shapley values for the given model.

        Args:
            sample_index (int): Sample index to plot. Defaults to 0.
            output_index (int): Output index to plot. Defaults to 0.
            path (str): Path to save the plot. Defaults to None.


        Raises:
            Exception: If the shap values have not been computed.
        """
        if not hasattr(self, 'shap_values'):
            raise Exception("Please compute the shap values first.")

        # get the expected_value and shap_values for the specified output
        if np.issubdtype(type(self.explainer.expected_value), np.floating):
            expected_value = self.explainer.expected_value
            shap_values = self.shap_values
        else:
            expected_value = self.explainer.expected_value[output_index]
            shap_values = self.shap_values[output_index]

        if isinstance(self.model, Module):
            # if the model is a pytorch model, we need to convert the data to torch tensors
            shap.force_plot(expected_value, shap_values[sample_index], torch.tensor(
                self.X_test_pd.iloc[sample_index]), feature_names=self.feature_names, matplotlib=True, show=False)
        else:
            shap.force_plot(expected_value, shap_values[sample_index], self.X_test_pd.iloc[sample_index],
                            feature_names=self.feature_names, matplotlib=True, show=False)
        if path is not None:
            save_path = f'{path}/{self.model_name}_shap_force_{sample_index}_{output_index}.png'
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()
