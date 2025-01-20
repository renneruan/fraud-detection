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


# @dataclass(frozen=True)
# class ModelTrainerConfig:
#     root_dir: Path
#     train_data_path: Path
#     test_data_path: Path
#     model_name: str
#     alpha: float
#     l1_ratio: float
#     target_column: str


# @dataclass(frozen=True)
# class ModelEvaluationConfig:
#     root_dir: Path
#     test_data_path: Path
#     model_path: Path
#     all_params: dict
#     metric_file_name: Path
#     target_column: str
#     mlflow_uri: str
