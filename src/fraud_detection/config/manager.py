from dotenv import load_dotenv
import os
from fraud_detection.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from fraud_detection.utils.commons import read_yaml, create_directories
from fraud_detection.entity.config_entity import (
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
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

        create_directories([config.root_path])

        data_validation_config = DataValidationConfig(
            root_path=config.root_path,
            raw_data_dir=config.raw_data_dir,
            status_file=config.status_file,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.transformed_data_path])

        data_transformation_config = DataTransformationConfig(
            transformed_data_path=config.transformed_data_path,
            raw_data_path=config.raw_data_path,
            target_column=config.target_column,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.LGBMClassifier
        schema = self.schema.TARGET_COLUMN

        create_directories([config.model_target_dir])

        model_trainer_config = ModelTrainerConfig(
            model_target_dir=config.model_target_dir,
            train_x_data_path=config.train_x_data_path,
            train_y_data_path=config.train_y_data_path,
            test_x_data_path=config.test_x_data_path,
            test_y_data_path=config.test_y_data_path,
            model_name=config.model_name,
            subsample=params.subsample,
            reg_lambda=params.reg_lambda,
            reg_alpha=params.reg_alpha,
            num_leaves=params.num_leaves,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            colsample_bytree=params.colsample_bytree,
            target_column=schema.fraude,
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.LGBMClassifier
        schema = self.schema.TARGET_COLUMN

        create_directories([config.model_results_dir])

        load_dotenv()

        model_evaluation_config = ModelEvaluationConfig(
            model_results_dir=config.model_results_dir,
            test_x_data_path=config.test_x_data_path,
            test_y_data_path=config.test_y_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.fraude,
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )

        return model_evaluation_config
