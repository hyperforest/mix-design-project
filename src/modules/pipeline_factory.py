import numpy as np
import optuna
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    check_array,
    check_is_fitted,
    check_X_y,
)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from modules.config import Config

from .objectives import get_xgb_objective


def _insert_onehot_encoder(pipeline, categorical=True, wrap=True):
    if categorical:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        pipeline.insert(1, ("onehot", ohe))

    if wrap:
        return Pipeline(pipeline).set_output(transform="pandas")
    else:
        return pipeline


def get_linreg_pipeline(params=None, categorical=False):
    params = params or {}
    pipeline = [("scaler", StandardScaler()), ("regressor", LinearRegression(**params))]
    return _insert_onehot_encoder(pipeline, categorical)


def get_svr_pipeline(params=None, categorical=False):
    params = params or {}
    pipeline = [("scaler", StandardScaler()), ("regressor", SVR(**params))]
    return _insert_onehot_encoder(pipeline, categorical)


def get_rf_pipeline(params=None, categorical=False):
    params = params or {}
    pipeline = [("regressor", RandomForestRegressor(**params))]
    return _insert_onehot_encoder(pipeline, categorical)


def get_xgb_pipeline(params=None, categorical=False, objective="smape"):
    params = params or {}
    objective = get_xgb_objective(objective)
    pipeline = [
        (
            "regressor",
            XGBRegressor(obj=objective, enable_categorical=categorical, **params),
        )
    ]
    return Pipeline(pipeline).set_output(transform="pandas")


def get_nn_pipeline(params=None, categorical=False):
    params = params or {}
    pipeline = [("scaler", MinMaxScaler()), ("regressor", MLPRegressor(**params))]
    return _insert_onehot_encoder(pipeline, categorical)


def get_sgdr_pipeline(params=None, categorical=False):
    params = params or {}
    pipeline = [("scaler", StandardScaler()), ("regressor", SGDRegressor(**params))]
    return _insert_onehot_encoder(pipeline, categorical)


class PipelineFactory:
    def __init__(self, config: Config):
        self.config = config

    def create(self, params=None):
        if params is None:
            return {}
        return params.copy()

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
        params_ = super().create(params)

        if "random_state" in params_:
            params_.pop("random_state")

        return get_linreg_pipeline(
            params_, categorical=self.config.age_days_as_categorical
        )

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
        params_ = super().create(params)
        return get_svr_pipeline(
            params_, categorical=self.config.age_days_as_categorical
        )

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
        params_ = super().create(params)
        return get_rf_pipeline(params_, categorical=self.config.age_days_as_categorical)

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
    def __init__(self, config, objective=None):
        super().__init__(config)
        self.objective = objective

    def create(self, params=None):
        params_ = super().create(params)
        return get_xgb_pipeline(
            params_,
            categorical=self.config.age_days_as_categorical,
            objective=self.objective
        )

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
        params_ = super().create(params)
        return get_nn_pipeline(params_, categorical=self.config.age_days_as_categorical)

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
        params_ = super().create(params)
        return get_sgdr_pipeline(
            params_, categorical=self.config.age_days_as_categorical
        )

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)
        params.update(
            {
                "loss": trial.suggest_categorical("loss", ["squared_error"]),
                "penalty": trial.suggest_categorical("penalty", ["elasticnet"]),
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
    def __init__(self, config, xgb_objective=None):
        super().__init__(config)
        self.xgb_objective = xgb_objective

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
            if "weights_xgb" in params:
                weights_xgb = params.pop("weights_xgb")
                params["weights"] = [weights_xgb, 1.0 - weights_xgb]

        xgb_params.pop("age_days_as_categorical", None)
        xgb_params.pop("use_height_as_feature", None)

        sgd_params.pop("age_days_as_categorical", None)
        sgd_params.pop("use_height_as_feature", None)

        xgb = get_xgb_pipeline(
            xgb_params,
            categorical=self.config.age_days_as_categorical,
            objective=self.xgb_objective
        )
        sgd = get_sgdr_pipeline(
            sgd_params, categorical=self.config.age_days_as_categorical
        )

        multiplier = params.pop("multiplier", 1.0)
        params.pop("random_state", None)

        estimator = VotingRegressor(estimators=[("xgb", xgb), ("sgd", sgd)], **params)
        model = AdjustedEstimator(estimator=estimator, multiplier=multiplier)
        return model

    @classmethod
    def get_params(cls, trial: optuna.Trial):
        params = super().get_params(trial)

        params.update(
            {
                "weights_xgb": trial.suggest_categorical(
                    "weights_xgb", np.arange(0.8, 0.95, 0.05).round(2).tolist()
                ),
                "multiplier": trial.suggest_categorical(
                    "multiplier", np.arange(1.0, 1.31, 0.01).round(2).tolist()
                ),
            }
        )

        xgb_params = XGBRegressionPipelineFactory.get_params(trial)
        sgd_params = SGDRegressionPipelineFactory.get_params(trial)

        for k, v in xgb_params.items():
            params[f"xgb_{k}"] = v

        for k, v in sgd_params.items():
            params[f"sgd_{k}"] = v

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
