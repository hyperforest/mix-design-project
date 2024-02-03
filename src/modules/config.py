from dataclasses import dataclass, field


@dataclass
class Config:
    experiment_name: str | None = None
    data_path: str = "../datasets/dataset.csv"
    scheme_path: str = "../datasets/scheme.csv"
    algo: str = "lr"
    random_state: int | None = None
    use_height_as_feature: bool = True
    age_days_as_categorical: bool = False
    n_splits: int = 5
    n_repeats: int = 10
    n_trials: int = 3
    log_model: bool = True
    metrics: str = "max_max_ape"
    features: list = field(
        default_factory=[
            "age_days",
            "diameter",
            "height",
            "fas",
            "water",
            "cement",
            "slump",
            "sikacim_kg",
            "fine_aggregate_kg",
            "coarse_aggregate_kg",
            "area",
        ]
    )
