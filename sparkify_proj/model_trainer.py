import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any




class ModelTrainer:
    """
    Class to train a binary classifier for churn detection.

    Attributes:
        X: Dataframe with features
        y: DataFrame with labels
        seed: Seed for reproducibility
        weights: Weights for the classes
        params: Classifier Parameters
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, seed: int):
        """
        ModelTrainer class constructor.

        Args:
            X: Dataframe with feature data
            y: DataFrame with label data
            seed: random seed for reproducibility
    
        """
        self.X = X
        self.y = y.values
        self.weights = self.calculate_weights()
        self.params = {}
        self.seed = seed

    def calculate_weights(self) -> Dict[int, float]:
        """
        Compute the weights for the classes.
        This is useful for imbalanced classes.
        (Here we are using it for XGBClassifier)

        Returns:
            weights: Weights for the classes (binary classification)
        
        """
        class_weight_current = len(self.y) / (2.0 * np.bincount(self.y))
        weights = {0: class_weight_current[0], 1: class_weight_current[1]}
        return weights

    def train(self, classifier: Any, precomputed_params: Dict[str, Any] = None) -> Any:
        """
        Train the binary classifier.

        Args:
            classifier: Classifier to train
            precomputed_params: Precomputed parameters for the classifier

        Returns:
            classifier: Trained classifier
        """
        if precomputed_params:
            self.params = precomputed_params

        if isinstance(classifier, XGBClassifier):
            self.params['scale_pos_weight'] = self.weights[0]/self.weights[1]
        
        if 'class_weight' in classifier.get_params().keys():
            self.params['class_weight'] = 'balanced'

        # Set the seed
        if 'random_state' in self.params.keys():
            self.params['random_state'] = self.seed

        classifier.set_params(**self.params)
        classifier.fit(self.X, self.y)
        return classifier



class CalibratedModelTrainer(ModelTrainer):
    """
    Class to train a calibrated binary classifier for churn detection.

    Attributes:
        X: Dataframe with features
        y: DataFrame with labels
        seed: Seed for reproducibility
        weights: Weights for the classes

    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, seed: int):
        """
        CalibratedModelTrainer class constructor.

        Args:
            X (pd.DataFrame): Dataframe with feature data
            y (pd.DataFrame): DataFrame with label data
            seed (int): random seed for reproducibility
    
        """
        super().__init__(X, y, seed)
        self.weights = self.calculate_weights()

    def train(self, classifier: Any, precomputed_params: Dict[str, Any] = None) -> Any:
        """
        Train the binary classifier.

        Args:
            classifier: Classifier to train
            precomputed_params: Precomputed parameters for the classifier

        Returns:
            classifier: Trained classifier
        """
        if precomputed_params:
            self.params = precomputed_params

        if isinstance(classifier, XGBClassifier):
            self.params['scale_pos_weight'] = self.weights[0]/self.weights[1]
        elif 'class_weight' in self.params.keys():
            self.params['class_weight'] = 'balanced'

        # Set the seed if random_state is in attributes
        if 'random_state' in self.params.keys():
            self.params['random_state'] = self.seed

        classifier.set_params(**self.params)
        classifier.fit(self.X, self.y)
        calibrated_classifier = CalibratedClassifierCV(
            classifier, cv=5, method='sigmoid')
        calibrated_classifier.fit(self.X, self.y)
        return calibrated_classifier
