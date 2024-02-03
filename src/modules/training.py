import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from modules.config import Config
from modules.pipeline_factory import PipelineFactory


def evaluate(model, X, y):
    y_pred = model.predict(X)
    ape = np.abs(y_pred - y) / y
    sape = np.abs(y_pred - y) / np.abs(y_pred + y)
    num_underestimate = np.sum(y_pred < y)
    return {
        "avg_ape": np.mean(ape),
        "max_ape": np.max(ape),
        "p90_ape": np.percentile(ape, 90),
        "p95_ape": np.percentile(ape, 95),
        "avg_sape": np.mean(sape),
        "max_sape": np.max(sape),
        "p90_sape": np.percentile(sape, 90),
        "p95_sape": np.percentile(sape, 95),
        "num_underestimate": num_underestimate
    }


def agg_metrics(cv_metrics):
    return (
        pd.DataFrame(cv_metrics)
        .describe(percentiles=[0.95, 0.90])
        .loc[["mean", "90%", "95%", "max"]]
        .rename(index={"mean": "avg", "90%": "p90", "95%": "p95"})
        .stack()
        .reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "value"})
        .assign(name=lambda x: x["a"] + "_" + x["b"])
        .drop(["a", "b"], axis=1)
        .set_index("name")
        .loc[:, "value"]
        .to_dict()
    )


def cross_validate(
    pipeline: Pipeline,
    cv: RepeatedKFold,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: "pd.DataFrame | None" = None,
    y_test: "pd.Series | None" = None,
):
    metrics_train, metrics_val = [], []

    for i, (train_index, val_index) in enumerate(cv.split(X_train, y_train)):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        model = clone(pipeline)
        model.fit(X_train_cv, y_train_cv)
        metrics_train.append(evaluate(model, X_train_cv, y_train_cv))
        metrics_val.append(evaluate(model, X_val_cv, y_val_cv))

    metrics = {
        "train": agg_metrics(metrics_train),
        "val": agg_metrics(metrics_val),
    }

    if X_test is not None and y_test is not None:
        model = clone(pipeline)
        model.fit(X_train, y_train)
        metrics_test = evaluate(model, X_test, y_test)
        metrics["test"] = metrics_test

        return metrics, model

    return metrics


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config,
    pipeline_factory: PipelineFactory,
    params: dict | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    mlflow_log: bool = False,
):
    if config.age_days_as_categorical:
        X_train["age_days"] = X_train["age_days"].astype(str)
        if X_test is not None:
            X_test["age_days"] = X_test["age_days"].astype(str)

    features_ = config.features[:]
    if not config.use_height_as_feature:
        features_.remove("height")

    X_train_ = X_train[features_]
    if X_test is not None:
        X_test_ = X_test[features_]
    else:
        X_test_ = None  

    cv = RepeatedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )

    params = params or {}
    if "age_days_as_categorical" in params:
        age_days_as_categorical = params.pop(  # noqa: F841
            "age_days_as_categorical", None
        )
    if "use_height_as_feature" in params:
        use_height_as_feature = params.pop("use_height_as_feature", None)  # noqa: F841

    pipeline = pipeline_factory.create(params=params)

    if X_test is not None:
        metrics, model = cross_validate(
            pipeline=pipeline,
            cv=cv,
            X_train=X_train_,
            y_train=y_train,
            X_test=X_test_,
            y_test=y_test,
        )
    else:
        metrics = cross_validate(
            pipeline=pipeline, cv=cv, X_train=X_train_, y_train=y_train
        )
        model.fit(X_train_, y_train)

    # rename metrics
    metrics = [{f"{k}_{k2}": v2 for k2, v2 in v.items()} for k, v in metrics.items()]
    metrics = {k: v for d in metrics for k, v in d.items()}

    if mlflow_log:
        mlflow.set_experiment(config.experiment_name)
        with mlflow.start_run() as run:  # noqa: F841
            logged_params = params.copy()
            logged_params.update(config.__dict__)

            mlflow.log_params(logged_params)
            mlflow.log_metrics(metrics)

            if config.log_model:
                mlflow.sklearn.log_model(model, "model")

    return metrics, model
