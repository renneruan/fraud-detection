"""
Módulo do pipeline utilizado para predição.

Pensado inicialmente para processamento de apenas um dados recebido.

A partir do dado recebido carrega o modelo, aplica o pré-processamento
adequado aos dados de entrada e devolve a probabilidade da classe.
"""

import joblib
from sklearn.exceptions import NotFittedError
from fraud_detection.constants import MODEL_PATH, PIPELINE_PATH
from fraud_detection import logger


class PredictionPipeline:
    """
    Class contendo pipeline de predição, submete os dados de entrada ao mesmo
    pipeline de transformação ajustado na etapa de pré-processamento dos dados
    de treino.
    """

    def __init__(self):
        self.pipeline = joblib.load(PIPELINE_PATH)
        self.model = joblib.load(MODEL_PATH)

    def transform_input_data(self, data):
        """
        Aplica o pipeline de pré-processamento nos dados a serem
        usados para predição. O pipeline possui informações de Imputers e
        Encodings salvos a partir apenas dos dados de treino.

        Args:
        - data (pd.DataFrame): DataFrame de linha única contendo dados a serem
            submetidos ao modelo.

        Returns:
        - pd.DataFrame: Dados transformados para aplicação ao modelo treinado.
        """
        try:
            transformed_data = self.pipeline.transform(data)
            print(transformed_data)
            return transformed_data
        except NotFittedError as e:
            logger.exception(
                "Pipeline não foi ajustado aos dados de treino. %s", e
            )
            return None

    def predict(self, data):
        """
        Função que realiza a predição dos dados, também invoca o método para
        pré-processar os dados inseridos.

        Args:
        - data (pd.DataFrame): Dados de entrada para predição.

        Returns:
        - tuple:
            - prediction int: Classe a qual os dados foram classificados
            - prediction_proba list: Lista de probabilidades das classes
        """
        prediction = self.model.predict(data)
        prediction_proba = self.model.predict_proba(data)
        return (prediction[0], prediction_proba[0][1])
