"""
Módulo responsável por gerenciar a leitura dos arquivos YAML de configuração.

Ele cria os diretórios necessários e fornece funções para que as classes de
configuração sejam utilizadas pelos respectivos componentes.
"""

import os
from dotenv import load_dotenv
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
    """
    Classe responsável por carregar e gerenciar configurações a partir de
     arquivos YAML.

    Lê os arquivos de configuração, cria diretórios e fornece métodos
     para que os componentes do pipeline acessem suas configurações.
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        """
        Inicializa a ConfigurationManager.

        Parâmetros:
            config_filepath (str): Path para arquivo de configuração principal.
            params_filepath (str): Path para arquivo de parâmetros do modelo.
            schema_filepath (str): Path para arquivo de esquema dos dados.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Obtém a configuração para a etapa de validação dos dados.

        Retorna:
            DataValidationConfig: Objeto com configurações para validação.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_path])

        return DataValidationConfig(
            root_path=config.root_path,
            raw_data_path=config.raw_data_path,
            status_file=config.status_file,
            all_schema=schema,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Obtém a configuração para a etapa de transformação dos dados.

        Retorna:
            DataTransformationConfig: Objeto de configurações da transformação.
        """
        config = self.config.data_transformation

        create_directories([config.transformed_data_path])

        return DataTransformationConfig(
            transformed_data_path=config.transformed_data_path,
            raw_data_path=config.raw_data_path,
            target_column=config.target_column,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Obtém a configuração para a etapa de treinamento do modelo.

        Retorna:
            ModelTrainerConfig: Objeto contendo os parâmetros do modelo
             e os caminhos dos dados de treino/teste.
        """
        config = self.config.model_trainer
        params = self.params.LGBMClassifier
        schema = self.schema.TARGET_COLUMN

        create_directories([config.model_target_path])

        return ModelTrainerConfig(
            model_target_path=config.model_target_path,
            train_x_data_path=config.train_x_data_path,
            train_y_data_path=config.train_y_data_path,
            test_x_data_path=config.test_x_data_path,
            test_y_data_path=config.test_y_data_path,
            model_name=config.model_name,
            subsample=params.subsample,
            reg_lambda=params.reg_lambda,
            num_leaves=params.num_leaves,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            colsample_bytree=params.colsample_bytree,
            scale_pos_weight=params.scale_pos_weight,
            target_column=schema.fraude,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Obtém a configuração para a etapa de avaliação do modelo.

        Retorna:
            ModelEvaluationConfig: Objeto contendo os parâmetros para registro
             métricas e caminhos de dados de teste e modelo.
        """
        config = self.config.model_evaluation
        params = self.params.LGBMClassifier
        schema = self.schema.TARGET_COLUMN

        create_directories([config.model_results_path])

        load_dotenv()

        return ModelEvaluationConfig(
            model_results_path=config.model_results_path,
            test_x_data_path=config.test_x_data_path,
            test_y_data_path=config.test_y_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.fraude,
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )
