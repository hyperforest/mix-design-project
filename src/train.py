import argparse
import json
import warnings

from modules.config import Config
from modules.data import load_data
from modules.pipeline_factory import get_pipeline_factory
from modules.tuning import get_objective_func, tune

warnings.filterwarnings("ignore")


def override_config(config, args):
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    if args.n_trials:
        config.n_trials = args.n_trials

    if args.log_model:
        config.log_model = args.log_model

    if args.algo:
        config.algo = args.algo

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

    # start tuning
    pipeline_factory = get_pipeline_factory(config.algo)(config)
    objective = get_objective_func(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        config=config,
        pipeline_factory=pipeline_factory,
    )
    study = tune(objective, config)

    return study


if __name__ == "__main__":
    study = main()
