import json
from .model_optimizer import ModelOptimizer
from .model_trainer import ModelTrainer, CalibratedModelTrainer
from .model_visualizer import ModelVisualizer
from .model_evaluator import ModelEvaluatorOpt
from sklearn.metrics import brier_score_loss, log_loss
from typing import List, Dict, Optional, Callable, Any
import os
import pickle
import datetime
import numpy as np
import logging
import pandas as pd
import sys
sys.path.append('../')  # Add the parent directory to the Python path


class ModelExperiment:
    """
    The class allows to experiment with different classifiers.    
    Args:
        classifiers (dict): Classifiers to evaluate.
        spaces (dict): Parameter spaces for each classifier.
        results (dict): Results dictionnary.
        seed (int): Random seed.
        workdir (str): Path to the directory where to store the results.
        max_evals (int): Maximum number of evaluations for each classifier.
    """

    def __init__(self, classifiers: Dict[str, Any], spaces: Dict[str, Any], results: Dict[str, Any] = None, seed: int = 42, workdir: str = '.', max_evals: int = 10):
        """
        ModelExperiment class constructor.
        
        Args:
            classifiers (dict): Classifiers to evaluate.
            spaces (dict): Parameter spaces for each classifier to be used for hyperopt.
            results (dict): dictionnary to store results.
            seed (int): Random seed for reproducibility.
            workdir (str): Path to the directory where to store the results.
            max_evals (int): Maximum number of evaluations for each classifier.
        
        """
        self.classifiers = classifiers
        self.spaces = spaces
        self.seed = seed
        if results is None:
            self.results = {'metrics': {}, 'parameters': {}}
        self.metrics = {}
        self.model_visualizer = ModelVisualizer()
        self.workdir = workdir
        self.models_path = f'{self.workdir}/models'
        self.artifacts_path = f'{self.workdir}/artifacts'
        self.max_evals = max_evals

    @staticmethod
    def timestamp() -> str:
        """
        Static method to generate a timestamp for the experiment.

        Returns:
            str: Timestamp.
        """
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def load_model(self, model_name: str):
        """
        Load model from pickle file given a file name.

        Args:
            model_name (str): Name of the model to load (obviously..)
        """

        model_name = f'{model_name}'

        try:
            with open(f'{self.models_path}/{model_name}.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        except (IOError, FileNotFoundError, pickle.UnpicklingError) as e:
            logging.error(f'Error loading model: {e}')

    def _create_directory(self, path: str):
        """
        Internal method to create a directory if it does not exist.

        Args:
            path (str): Path to the directory to create (models or artifacts)
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def create_directories(self):
        """
        Method to create models and artifacts directories.
        """
        self._create_directory(f'{self.models_path}')
        self._create_directory(f'{self.artifacts_path}')

    def save_model(self, model: Any, model_name: str):
        """
        Method for saving a trained model.

        Args:
            model (Any): Our trained model
            model_name (str): A name you choose to indentify the model with.
        """
        model_name = f'{model_name}'
        try:
            with open(f'{self.models_path}/{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        except (IOError, pickle.PicklingError) as e:
            logging.error(f'Error saving model: {e}')

    def save_results(self, calibration: bool = False):
        """
        Save metrics and the best parameters for each classifier.
        Calibration results are saved in a separate file.

        Args:
            calibration: Boolean to indicate if the results are for calibrated models.
        """

        if calibration:
            results_name = f'calibration_results.pkl'
        else:
            results_name = f'results.pkl'
        try:
            with open(f'{self.artifacts_path}/{results_name}', 'wb') as f:
                pickle.dump(self.results, f)
        except (IOError, pickle.PicklingError) as e:
            logging.error(f'Error saving results: {e}')

    # save result as json file
    def save_metrics_json(self, calibration: bool = False):
        """
        Save only the metrics for each classifier in a json file.
        Calibration results are saved in a separate file.

        Args:
            calibration: Boolean to indicate if the results are for calibrated models.
        """
        if calibration:
            metrics_name = f'calibration_metrics.json'
        else:
            metrics_name = f'metrics.json'
        try:
            with open(f'{self.artifacts_path}/{metrics_name}', 'w') as f:
                json.dump(self.metrics, f)
        except (IOError, pickle.PicklingError) as e:
            logging.error(f'Error saving metrics: {e}')

    def compute_metrics(self, classifier_name: str, classifier: Any, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        Plots the ROC and PR curves and the confusion matrix.
        Computes metrics for the classifier evaluation:
            - precision 
            - recall, 
            - fpr (False positive rate)
            - tpr (True positive rate)
            - roc_auc (receiver operating characteristic curve)
            - pr_auc (precision recall curve)
            - brier_score (Brier score is used for evaluating the model calibration)
            - log_loss ( log loss is used for model calibration)

        Args:
            classifier_name (str): Name of the classifier.
            classifier: Our classifier.
            X_test (pd.DataFrame): Feature data from the test set.
            y_test (pd.DataFrame): Label data from test set.
        """
        classifier_name_fig = f'{classifier_name}'
        model_evaluator = ModelEvaluatorOpt(classifier)
        y_pred = model_evaluator.predict(X_test)
        metrics, y_pred_binary = model_evaluator.calculate_metrics(
            y_test, y_pred)
        precision, recall, fpr, tpr, roc_auc, pr_auc = model_evaluator.calculate_auc(
            y_test, y_pred)
        # compute brier score and log loss
        brier_score = brier_score_loss(y_test, y_pred)
        log_loss_score = log_loss(y_test, y_pred)
        self.results['metrics'][classifier_name] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'precision': precision,
                                                    'recall': recall, 'pr_auc': pr_auc, 'brier_score': brier_score, 'log_loss': log_loss_score}
        self.model_visualizer.plot_curves(
            precision, recall, fpr, tpr, roc_auc, pr_auc, model_name=classifier_name_fig, path=self.artifacts_path)
        self.model_visualizer.plot_confusion_matrix(
            y_test, y_pred_binary, model_name=classifier_name_fig, path=self.artifacts_path)
        self.metrics[classifier_name] = metrics

    def _train_and_evaluate(self, classifier: Any, classifier_name: str, model_trainer: ModelTrainer, model_optimizer: ModelOptimizer, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        Internal method for training and evaluating a given classifier.

        Args:
            classifier (Any): Classifier to train and evaluate.
            classifier_name (str): Name of the classifier.
            model_trainer (ModelTrainer): ModelTrainer object to train the classifier.
            model_optimizer (ModelOptimizer): ModelOptimizer object to do hyperparamter optimization.
            X_test (pd.DataFrame): Feature data from the test set.
            y_test (pd.DataFrame): Label data from test set.        
        """
        best_params = model_optimizer.optimize(
            classifier, self.spaces[classifier_name])
        print(f'Best parameters for {classifier_name}: {best_params}')
        self.results['parameters'][classifier_name] = best_params
        trained_classifier = model_trainer.train(classifier, best_params)
        self.compute_metrics(
            classifier_name, trained_classifier, X_test, y_test)
        self.save_model(trained_classifier, classifier_name)

    def run(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the experiment for each classifier and hyperparameter space.

        Args:
            X_train (pd.Dataframe): Training data
            y_train (pd.DataFrame): Training labels
            X_test (pd.Dataframe): Test data
            y_test (pd.DataFrame): Test labels

        Returns:
            results (Dict): Dictionary with the results of the experiment
        """
        # convert y_train and y_test to series
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        self.create_directories()
        model_optimizer = ModelOptimizer(
            X_train, y_train, max_evals=self.max_evals)
        model_trainer = ModelTrainer(X_train, y_train, seed=self.seed)
        for classifier_name, classifier in self.classifiers.items():
            self._train_and_evaluate(
                classifier, classifier_name, model_trainer, model_optimizer, X_test, y_test)

        self.model_visualizer.plot_summary_curves(self.results['metrics'])
        self.save_results()
        self.save_metrics_json()
        return self.results

    def run_calibration_exp(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Compares the performance of a classifier with and without calibration.
        This method should be called after the run method.

        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Training labels
            X_test (pd.DataFrame): Test data
            y_test (pd.DataFrame): Test labels

        Returns:
            results (Dict): Dictionary with the results of the calibration experiment
        """
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        calib_model_trainer = CalibratedModelTrainer(
            X_train, y_train, seed=self.seed)
        for classifier_name, classifier in self.classifiers.items():
            trained_calib_classifier = calib_model_trainer.train(
                classifier, self.results['parameters'][classifier_name])
            self.compute_metrics(
                f'calibrated_{classifier_name}', trained_calib_classifier, X_test, y_test)
            self.save_model(trained_calib_classifier,
                            f'calibrated_{classifier_name}')

        self.model_visualizer.plot_summary_curves(
            self.results['metrics'], path=self.artifacts_path)
        self.save_results()
        self.save_metrics_json()
        return self.results

    def vizualize_calibration(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        We compare the performance of each model with its calibrated version by using line plots and histograms.

        Args:
            X_test (pd.DataFrame): Test data
            y_test (pd.DataFrame): Test labels
        """
        y_test = y_test.squeeze()
        for classifier_name in self.classifiers.keys():
            model = self.load_model(classifier_name)
            calib_model = self.load_model(f'calibrated_{classifier_name}')
            # predict probabilities
            y_pred = model.predict_proba(X_test)[:, 1]
            y_pred_calib = calib_model.predict_proba(X_test)[:, 1]
            ModelVisualizer.plot_comp_calibration(
                y_test, y_pred, y_pred_calib, path=self.artifacts_path, classifier_name=f'{classifier_name}')
