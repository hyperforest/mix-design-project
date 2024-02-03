import mlflow
import optuna
import pandas as pd

from modules.config import Config
from modules.pipeline_factory import PipelineFactory
from modules.training import train


def tune(objective, config: Config):
    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(
        study_name=config.experiment_name,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=config.n_trials)

    return study


def get_objective_func(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    pipeline_factory: PipelineFactory,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
):
    def objective(trial: optuna.Trial):
        params = pipeline_factory.get_params(trial)
        params["random_state"] = config.random_state

        mlflow.set_experiment(config.experiment_name)
        metrics, model = train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            params=params,
            pipeline_factory=pipeline_factory,
            mlflow_log=True
        )

        return metrics[f"val_{config.metrics}"]

    return objective
