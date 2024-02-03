import numpy as np
import optuna
from sklearn.base import (BaseEstimator, RegressorMixin, check_array,
                          check_is_fitted, check_X_y)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from modules.config import Config


class PipelineFactory:
    def __init__(self, config: Config):
        self.config = config

    def create(self, params=None):
        raise NotImplementedError

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        return {
            "age_days_as_categorical": trial.suggest_categorical(
                "age_days_as_categorical", [True, False]
            ),
            "use_height_as_feature": trial.suggest_categorical(
                "use_height_as_feature", [True, False]
            ),
        }


class LinearRegressionPipelineFactory(PipelineFactory):
    def __init__(self, config):
        super().__init__(config)

    def create(self, params=None):
        params = params or {}

        if "random_state" in params:
            params.pop("random_state")

        if not self.config.age_days_as_categorical:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", LinearRegression(**params)),
                ]
            ).set_output(transform="pandas")
        else:
            return Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("scaler", StandardScaler()),
                    ("regressor", LinearRegression(**params)),
                ]
            ).set_output(transform="pandas")

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
                "positive": trial.suggest_categorical("positive", [True, False]),
            }
        )

        return params


class SVRPipelineFactory(PipelineFactory):
    def __init__(self, config):
        super().__init__(config)

    def create(self, params=None):
        if "random_state" in params:
            params.pop("random_state")

        if not self.config.age_days_as_categorical:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", SVR(**params)),
                ]
            ).set_output(transform="pandas")
        else:
            return Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("scaler", StandardScaler()),
                    ("regressor", SVR(**params)),
                ]
            ).set_output(transform="pandas")

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf"]
                ),
                "degree": trial.suggest_int("degree", 1, 5),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "C": trial.suggest_loguniform("C", 1e-3, 1e3),
                "epsilon": trial.suggest_loguniform("epsilon", 1e-3, 1e3),
            }
        )

        return params


class RFRegressionPipelineFactory(PipelineFactory):
    def __init__(self, config):
        super().__init__(config)

    def create(self, params=None):
        params = params or {}

        if not self.config.age_days_as_categorical:
            return Pipeline(
                [
                    ("regressor", RandomForestRegressor(**params)),
                ]
            ).set_output(transform="pandas")
        else:
            return Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("regressor", RandomForestRegressor(**params)),
                ]
            ).set_output(transform="pandas")

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 32),
                "max_features": trial.suggest_categorical("max_features", [None]),
            }
        )

        return params


class XGBRegressionPipelineFactory(PipelineFactory):
    def __init__(self, config):
        super().__init__(config)

    def mape_objective(self, y_true, y_pred):
        grad = np.sign(y_pred - y_true) / (y_true * len(y_true))
        hess = np.zeros_like(y_true)
        return grad, hess

    def smape_objective(self, y_true, y_pred):
        grad = np.sign(y_pred - y_true) / np.abs(y_true + y_pred)
        grad -= np.abs(y_true - y_pred) / np.square(y_true + y_pred)
        grad /= len(y_true)

        hess = 2 * (10000.0 * (y_pred == y_true)) / np.abs(y_true + y_pred)
        hess += 2.0 * np.abs(y_true - y_pred) / np.abs(y_true + y_pred) ** 3
        hess += 2.0 * np.sign(y_true - y_pred) / np.square(y_true + y_pred)
        hess /= len(y_true)

        return grad, hess

    def create(self, params=None):
        params = params or {}

        if not self.config.age_days_as_categorical:
            return Pipeline(
                [
                    ("regressor", XGBRegressor(obj=self.smape_objective, **params)),
                ]
            ).set_output(transform="pandas")
        else:
            return Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("regressor", XGBRegressor(obj=self.smape_objective, **params)),
                ]
            ).set_output(transform="pandas")

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e0),
            }
        )

        return params


class NNRegressionPipelineFactory(PipelineFactory):
    def __init__(self, config):
        super().__init__(config)

    def create(self, params=None):
        params = params or {}

        if not self.config.age_days_as_categorical:
            return Pipeline(
                [
                    ("scaler", MinMaxScaler()),
                    ("regressor", MLPRegressor(**params)),
                ]
            ).set_output(transform="pandas")
        else:
            return Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("scaler", StandardScaler()),
                    ("regressor", MLPRegressor(**params)),
                ]
            ).set_output(transform="pandas")

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", [(32,), (64,)]
                ),
                "activation": trial.suggest_categorical("activation", ["relu"]),
                "solver": trial.suggest_categorical("solver", ["adam"]),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 0.0001, 0.1
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 100),
                "beta_1": trial.suggest_loguniform("beta_1", 1e-3, 1e0),
                "beta_2": trial.suggest_loguniform("beta_2", 1e-3, 1e0),
            }
        )

        return params


class SGDRegressionPipelineFactory(PipelineFactory):
    def __init__(self, config):
        super().__init__(config)

    def create(self, params=None):
        params = params or {}

        if not self.config.age_days_as_categorical:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", SGDRegressor(**params)),
                ]
            ).set_output(transform="pandas")
        else:
            return Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("scaler", StandardScaler()),
                    ("regressor", SGDRegressor(**params)),
                ]
            ).set_output(transform="pandas")

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "loss": trial.suggest_categorical("loss", ["squared_error"]),
                "penalty": trial.suggest_categorical(
                    "penalty", ["l2", "l1", "elasticnet"]
                ),
                "alpha": trial.suggest_loguniform("alpha", 1e-3, 1e0),
                "l1_ratio": trial.suggest_loguniform("l1_ratio", 1e-3, 1e0),
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
            }
        )

        return params


class AdjustedEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, multiplier=1.0):
        self.estimator = estimator
        self.multiplier = multiplier
        self._is_fitted = False

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.estimator.fit(X, y)
        self._is_fitted = True
        return self
    
    def __sklearn_is_fitted__(self):
        return self._is_fitted

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.estimator.predict(X) * self.multiplier


class XGBoostLinregPipelineFactory(PipelineFactory):
    def __init__(self, config):
        super().__init__(config)

    def create(self, params=None):
        if params is None:
            params, xgb_params, sgd_params = {}, {}, {}
        else:
            xgb_params = {
                k.replace("xgb_", ""): v
                for k, v in params.items()
                if k.startswith("xgb_")
            }
            sgd_params = {
                k.replace("sgd_", ""): v
                for k, v in params.items()
                if k.startswith("sgd_")
            }
            params = {
                k: v
                for k, v in params.items()
                if not k.startswith("xgb_") and not k.startswith("sgd_")
            }
            if 'weights_xgb' in params:
                weights_xgb = params.pop('weights_xgb')
                params['weights'] = [weights_xgb, 1.0 - weights_xgb]

        if not self.config.age_days_as_categorical:
            xgb = Pipeline(
                [
                    ("regressor", XGBRegressor(**xgb_params)),
                ]
            ).set_output(transform="pandas")

            sgd = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", SGDRegressor(**sgd_params)),
                ]
            ).set_output(transform="pandas")
        else:
            xgb = Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("regressor", XGBRegressor(**xgb_params)),
                ]
            ).set_output(transform="pandas")

            sgd = Pipeline(
                [
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    ("scaler", StandardScaler()),
                    ("regressor", SGDRegressor(**sgd_params)),
                ]
            ).set_output(transform="pandas")

        multiplier = params.pop("multiplier", 1.0)
        params.pop("random_state", None)
        model = AdjustedEstimator(
            estimator=VotingRegressor(estimators=[("xgb", xgb), ("sgd", sgd)], **params),
            multiplier=multiplier,
        )

        return model

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "weights_xgb": trial.suggest_categorical(
                    "weights_xgb", np.arange(0.7, 0.95, 0.05).round(2).tolist()
                ),
                "multiplier": trial.suggest_categorical(
                    "multiplier", np.arange(1.0, 1.31, 0.01).round(2).tolist()
                ),
                "xgb_n_estimators": trial.suggest_int("xgb_n_estimators", 10, 100),
                "xgb_max_depth": trial.suggest_int("xgb_max_depth", 2, 6),
                "xgb_learning_rate": trial.suggest_loguniform(
                    "xgb_learning_rate", 1e-3, 1e0
                ),
                "sgd_loss": trial.suggest_categorical("sgd_loss", ["squared_error"]),
                "sgd_penalty": trial.suggest_categorical(
                    "sgd_penalty", ["l2", "l1", "elasticnet"]
                ),
                "sgd_alpha": trial.suggest_loguniform("sgd_alpha", 1e-3, 1e0),
                "sgd_l1_ratio": trial.suggest_loguniform("sgd_l1_ratio", 1e-3, 1e0),
                "sgd_fit_intercept": trial.suggest_categorical(
                    "sgd_fit_intercept", [True, False]
                ),
            }
        )

        return params


def get_pipeline_factory(algo):
    pipeline_factories = {
        "lr": LinearRegressionPipelineFactory,
        "svm": SVRPipelineFactory,
        "rf": RFRegressionPipelineFactory,
        "xgb": XGBRegressionPipelineFactory,
        "nn": NNRegressionPipelineFactory,
        "sgd": SGDRegressionPipelineFactory,
        "xgblr": XGBoostLinregPipelineFactory,
    }

    if algo not in pipeline_factories:
        raise ValueError(f"Invalid algo: {algo}")

    return pipeline_factories[algo]
