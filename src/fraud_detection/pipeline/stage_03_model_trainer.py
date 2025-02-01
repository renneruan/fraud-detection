"""
Arquivo contendo etapa do pipeline para Treinamento dos Dados.

Serve para acoplar as configurações lidas do arquivo yaml no componente.
Pode ser executado independentemente.
"""

from fraud_detection.config.manager import ConfigurationManager
from fraud_detection.components.model_trainer import ModelTrainer
from fraud_detection import logger


STAGE_NAME = "Model Trainer"


def ModelTrainerTrainingPipeline():
    """
    Função para repassar configuração para etapa de Treinamento dos dados
    Invoca método de treino do modelo com os dados processados.
    """
    config = ConfigurationManager()
    model_trainer_config = config.get_model_trainer_config()
    model_trainer_config = ModelTrainer(config=model_trainer_config)
    model_trainer_config.train()


if __name__ == "__main__":
    try:
        logger.info("[INICIO DE ETAPA] %s", STAGE_NAME)

        ModelTrainerTrainingPipeline()

        logger.info("[FIM DE ETAPA] %s completo.\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
