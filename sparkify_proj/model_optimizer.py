import pandas as pd
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from xgboost import XGBClassifier
from typing import Dict, Any, Callable


class ModelOptimizer:
    """
    Class to optimize the hyperparameters of a classifier using Hyperopt.

    Attributes:
        X (pd.Dataframe): Dataframe of features
        y (pd.Dataframe): Dataframe of target
        max_evals (int): Maximum number of evaluations for Hyperopt
    """

    def __init__(self, X, y, max_evals=50):
        self.X = X
        self.y = y
        self.max_evals = max_evals

    def f2_score(self, y_true, y_pred):
        """
        Computes the f2 score for the classifier.
        F2 score weighs recall higher than precision.

        Args:
            y_true: Ground truth
            y_pred: Predicted values

        Returns:
            f2_score: F2 score for the classifier.

        """
        return fbeta_score(y_true, y_pred, beta=2)

    def type_casting(self, params: Dict[str, Any]) -> Dict:
        """
        Converts float value to interger if it is equal to its integer value.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            params: Dictionary of hyperparameters with integer values.
        """
        return {key: int(value) if isinstance(value, float) and int(value) == value else value for key, value in params.items()}

    def calculate_weights(self) -> Dict[int, int]:
        """
        Compute the weights for the classes.
        This is useful for imbalanced classes.
        (Here we are using it for XGBClassifier)

        Returns:
            Dict [int, int] : Dictionnary of class weights.
        """
        class_weight_current = len(self.y) / (2.0 * np.bincount(self.y))
        return {0: class_weight_current[0], 1: class_weight_current[1]}

    def optimize(self, classifier: Any, space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Method to optimize the hyperparameters of a classifier.

        Args:
            classifier (Any): Classifier to optimize
            space (Dict[str, Any]): Space of hyperparameters to optimize

        Returns:
            best_params (Dict[str, Any]): Best hyperparameters generated by Hyperopt
        """

        trials = Trials()
        best = fmin(fn=self.get_objective_function(classifier),
                    space=space,
                    algo=tpe.suggest,
                    max_evals=self.max_evals,
                    trials=trials)

        best_params = space_eval(space, best)
        best_params = self.type_casting(best_params)
        return best_params

    def get_objective_function(self, classifier: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Wrapper method to create the objective function for Hyperopt.

        Args:
            classifier (Any): Classifier to optimize

        Returns:
            objective: Custom objective function for Hyperopt
        """

        def objective(params):
            params = self.type_casting(params)
            clf = classifier
            if isinstance(clf, XGBClassifier):
                weights = self.calculate_weights()
                params['scale_pos_weight'] = weights[0]/weights[1]
            elif 'class_weight' in params.keys():
                params['class_weight'] = 'balanced'
            clf.set_params(**params)
            score = cross_val_score(clf, self.X, self.y, cv=3, scoring=make_scorer(
                self.f2_score), n_jobs=-1).mean()
            return {'loss': -score, 'status': STATUS_OK}
        return objective
