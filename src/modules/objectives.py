import numpy as np


def mape_objective(y_true, y_pred):
    grad = np.sign(y_pred - y_true) / (y_true * len(y_true))
    hess = np.zeros_like(y_true)
    return grad, hess


def smape_objective(y_true, y_pred):
    grad = np.sign(y_pred - y_true) / np.abs(y_true + y_pred)
    grad -= np.abs(y_true - y_pred) / np.square(y_true + y_pred)
    grad /= len(y_true)

    hess = 2 * (10000.0 * (y_pred == y_true)) / np.abs(y_true + y_pred)
    hess += 2.0 * np.abs(y_true - y_pred) / np.abs(y_true + y_pred) ** 3
    hess += 2.0 * np.sign(y_true - y_pred) / np.square(y_true + y_pred)
    hess /= len(y_true)

    return grad, hess


def get_xgb_objective(objective_name: str | None = None):
    if objective_name is None:
        return 'reg:squarederror'

    obj_dict = {
        "mape": mape_objective,
        "smape": smape_objective
    }

    return obj_dict[objective_name]
