from fraud_detection.constants import *
from fraud_detection.utils import read_yaml, create_directories
from fraud_detection.entity.config_entity import (
    # DataIngestionConfig,
    DataValidationConfig,
    # DataTransformationConfig,
    # ModelTrainerConfig,
    # ModelEvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            raw_data_dir=config.raw_data_dir,
            status_file=config.status_file,
            all_schema=schema,
        )

        return data_validation_config

    # def get_data_transformation_config(self) -> DataTransformationConfig:
    #     config = self.config.data_transformation

    #     create_directories([config.root_dir])

    #     data_transformation_config = DataTransformationConfig(
    #         root_dir=config.root_dir,
    #         data_path=config.data_path,
    #     )

    #     return data_transformation_config

    # def get_model_trainer_config(self) -> ModelTrainerConfig:
    #     config = self.config.model_trainer
    #     params = self.params.ElasticNet
    #     schema = self.schema.TARGET_COLUMN

    #     create_directories([config.root_dir])

    #     model_trainer_config = ModelTrainerConfig(
    #         root_dir=config.root_dir,
    #         train_data_path=config.train_data_path,
    #         test_data_path=config.test_data_path,
    #         model_name=config.model_name,
    #         alpha=params.alpha,
    #         l1_ratio=params.l1_ratio,
    #         target_column=schema.name,
    #     )

    #     return model_trainer_config

    # def get_model_evaluation_config(self) -> ModelEvaluationConfig:
    #     config = self.config.model_evaluation
    #     params = self.params.ElasticNet
    #     schema = self.schema.TARGET_COLUMN

    #     create_directories([config.root_dir])

    #     model_evaluation_config = ModelEvaluationConfig(
    #         root_dir=config.root_dir,
    #         test_data_path=config.test_data_path,
    #         model_path=config.model_path,
    #         all_params=params,
    #         metric_file_name=config.metric_file_name,
    #         target_column=schema.name,
    #         mlflow_uri="https://dagshub.com/entbappy/End-to-end-Machine-Learning-Project-with-MLflow.mlflow",
    #     )

    #     return model_evaluation_config
