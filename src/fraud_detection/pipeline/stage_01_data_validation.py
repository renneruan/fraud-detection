"""
Arquivo contendo etapa do pipeline para Validação dos Dados de entrada.

Serve para acoplar as configurações lidas do arquivo yaml no componente.
Pode ser executado independentemente.
"""

from fraud_detection.config.manager import ConfigurationManager
from fraud_detection.components.data_validation import DataValidation
from fraud_detection import logger


STAGE_NAME = "Data Validation"


def DataValidationTrainingPipeline():
    """
    Função para repassar configuração para etapa de Validação dos dados
    Invoca método do componente de Validação para validar todas as colunas.
    """

    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_all_columns()


if __name__ == "__main__":
    try:
        logger.info("[INICIO DE ETAPA] %s", STAGE_NAME)

        DataValidationTrainingPipeline()

        logger.info("[FIM DE ETAPA] %s completo.\n\n", STAGE_NAME)
    except Exception as e:
        logger.exception(e)
        raise e
