from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    raw_data_dir: Path
    status_file: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    raw_data_path: Path
    transformed_data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    model_target_dir: Path
    train_x_data_path: Path
    test_x_data_path: Path
    train_y_data_path: Path
    test_y_data_path: Path
    model_name: str
    subsample: float
    reg_lambda: float
    reg_alpha: float
    num_leaves: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    colsample_bytree: float
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_results_dir: Path
    test_x_data_path: Path
    test_y_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
