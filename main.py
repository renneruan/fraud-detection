"""
Este script é responsável por orquestrar as etapas do treinamento do modelo.
O treinamento é dividido em 4 etapas:
1. Data Validation: Validação dos dados de entrada para verificar colunas
2. Data Transformation: Transformação dos dados de acordo com EDA realizado
3. Model Trainer: Treinamento do modelo com hiperparâmetros otimizados
4. Model Evaluation: Avaliação do modelo nos dados de teste
"""

from fraud_detection import logger

from fraud_detection.pipeline.stage_01_data_validation import (
    DataValidationTrainingPipeline,
)

from fraud_detection.pipeline.stage_02_data_transformation import (
    DataTransformationTrainingPipeline,
)

from fraud_detection.pipeline.stage_03_model_trainer import (
    ModelTrainerTrainingPipeline,
)

from fraud_detection.pipeline.stage_04_model_evaluation import (
    ModelEvaluationTrainingPipeline,
)


# Utiliza estratégia de programação funcional para iterar sobre etapas
stages = {
    "Data Validation": DataValidationTrainingPipeline,
    "Data Transformation": DataTransformationTrainingPipeline,
    "Model Trainer": ModelTrainerTrainingPipeline,
    "Model Evaluation": ModelEvaluationTrainingPipeline,
}


for stage_name, stage_function in stages.items():
    try:
        logger.info("[INICIO DE ETAPA] %s", stage_name)
        data_validation = stage_function()

        # Cada arquivo de pipeline precisa possuir o método main
        data_validation.main()
        logger.info("[FIM DE ETAPA], %s\n\n", stage_name)
        logger.info("---------------------------------------------")
    except Exception as e:
        logger.error("Erro na etapa %s", stage_name)
        logger.exception(e)
        raise e

logger.info("Treinamento finalizado com sucesso!")
