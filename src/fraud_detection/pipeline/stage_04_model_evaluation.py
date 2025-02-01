"""
Arquivo contendo etapa do pipeline para Avaliação do modelo.

Serve para acoplar as configurações lidas do arquivo yaml no componente.
Pode ser executado independentemente.
"""

from fraud_detection.config.manager import ConfigurationManager
from fraud_detection.components.model_evaluation import ModelEvaluation
from fraud_detection import logger

STAGE_NAME = "Model Evaluation"


def ModelEvaluationTrainingPipeline():
    """
    Função para repassar configuração para etapa de Avaliação dos dados
    Invoca método de avaliação do modelo após o treinamento.
    Também salva as métricas na plataforma MLFlow
    """
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()
    model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
    model_evaluation_config.start_mlflow()


if __name__ == "__main__":
    try:
        logger.info("[INICIO DE ETAPA] %s", STAGE_NAME)

        ModelEvaluationTrainingPipeline()

        logger.info("[FIM DE ETAPA] %s completo.\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
