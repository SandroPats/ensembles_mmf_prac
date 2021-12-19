import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.base_trees = []
        self.validation_score = []
        self.estimator_count = 0

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        elif self.feature_subsample_size > X.shape[1]:
            self.feature_subsample_size = X.shape[1]

        if X_val is not None and y_val is not None:
            self.validation_score = []
        else:
            self.validation_score = None

        for i in range(self.n_estimators):
            bootstrap_inds = np.random.randint(0, X.shape[0], X.shape[0])
            feature_set_idx = np.random.choice(X.shape[1],
                                               (self.feature_subsample_size),
                                               replace=False)
            X_train = X[bootstrap_inds][:, feature_set_idx]
            y_train = y[bootstrap_inds]
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         **self.trees_parameters)
            tree.fit(X_train, y_train)
            self.base_trees.append([tree, feature_set_idx])
            self.estimator_count += 1

            if self.validation_score is not None:
                mse = mean_squared_error(y_val, self.predict(X_val),
                                         squared=False)
                self.validation_score.append(mse)

        return self.validation_score

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        prediction_sum = 0
        for tree, feature_set in self.base_trees:
            prediction_sum += tree.predict(X[:, feature_set])

        return prediction_sum / self.estimator_count


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5,
        feature_subsample_size=None, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        def mse_for_coef(coef):
            return ((y - self.alg_sum -
                     self.current_pred*coef) ** 2).mean()

        self.base_trees = []
        self.validation_score = []
        self.estimator_count = 0
        self.alg_sum = np.zeros(X.shape[0])
        
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        if X_val is not None and y_val is not None:
            self.validation_score = []
        else:
            self.validation_score = None

        for i in range(self.n_estimators):
            feature_set_idx = np.random.choice(X.shape[1],
                                               (self.feature_subsample_size),
                                               replace=False)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         **self.trees_parameters)
            tree.fit(X[:, feature_set_idx], 2 * (y - self.alg_sum))
            self.current_pred = tree.predict(X[:, feature_set_idx])
            coef = minimize_scalar(mse_for_coef).x
            self.alg_sum += self.learning_rate * self.current_pred * coef
            self.base_trees.append([tree, feature_set_idx, coef])
            self.estimator_count += 1

            if self.validation_score is not None:
                mse = mean_squared_error(y_val, self.predict(X_val),
                                         squared=False)
                self.validation_score.append(mse)

        return self.validation_score

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        prediction_sum = 0
        for tree, feature_set, coef in self.base_trees:
            prediction_sum += (self.learning_rate *
                               coef *
                               tree.predict(X[:, feature_set]))

        return prediction_sum
