import argparse
import json
import warnings
from itertools import combinations

from modules.config import Config
from modules.data import load_data
from modules.pipeline_factory import get_pipeline_factory
from modules.training import train
from modules.tuning import get_objective_func, tune

warnings.filterwarnings("ignore")


def override_config(config, args):
    for k, v in args.__dict__.items():
        if v is not None:
            setattr(config, k, v)

    return config


def main():
    # handling training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--log_model", default=False)
    args = parser.parse_args()

    # load config
    with open(args.config_path) as f:
        config = Config(**json.load(f))

    config = override_config(config, args)

    # load data
    df_train, df_test = load_data(
        config.data_path, config.scheme_path, random_state=config.random_state
    )
    X_train, y_train = df_train.drop("weight", axis=1), df_train.weight
    X_test, y_test = df_test.drop("weight", axis=1), df_test.weight

    print(df_train.shape, df_test.shape)

    # feature selection
    best_metrics, best_comb = float("inf"), None
    all_feats = config.features[:]
    for r in range(2, len(all_feats) + 1):
        combs = list(combinations(all_feats, r))
        for i, comb in enumerate(combs):
            config.features = list(comb)

            # start tuning
            pipeline_factory = get_pipeline_factory(config.algo)(config)
            if config.n_trials == 0:
                params = {"positive": True} if config.algo == "lr" else None

                metrics, model = train(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    params=params,
                    config=config,
                    pipeline_factory=pipeline_factory,
                    mlflow_log=True,
                )

                monitored = metrics["val_" + config.metrics]
                if monitored < best_metrics:
                    best_metrics = monitored
                    best_comb = comb

                print(
                    f"[{i + 1}/{len(combs)}] - metrics: {monitored:.5f} - " + \
                    f"best metrics: {best_metrics:.5f} - best comb: {best_comb}"
                )
            else:
                objective = get_objective_func(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    config=config,
                    pipeline_factory=pipeline_factory,
                )
                study = tune(objective, config)  # noqa: F841


if __name__ == "__main__":
    result = main()
