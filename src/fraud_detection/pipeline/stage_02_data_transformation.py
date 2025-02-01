"""
Arquivo contendo etapa do pipeline para Transformação dos Dados.

Serve para acoplar as configurações lidas do arquivo yaml no componente.
Pode ser executado independentemente
"""

from fraud_detection.config.manager import ConfigurationManager
from fraud_detection.components.data_transformation import DataTransformation
from fraud_detection import logger


STAGE_NAME = "Data Transformation"


def DataTransformationTrainingPipeline():
    """
    Função para repassar configuração para etapa de Transformação dos dados
    Invoca método de pré-processamento dos dados para o modelo ser treinado.
    """
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.preprocessing_pipeline()


if __name__ == "__main__":
    try:
        logger.info("[INICIO DE ETAPA] %s", STAGE_NAME)

        DataTransformationTrainingPipeline()

        logger.info("[FIM DE ETAPA] %s completo.\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
