"""
Este módulo realiza a avaliação do modelo de Machine Learning treinado.

Irá coletar as métricas e salvá-las utilizando MLFlow.

Classes:
    ModelEvaluation: Classe para obtenção e registro de métricas.

Dependências:
    - pandas
    - sklearn
    - urlib
    - mlflow
    - joblib
    - pathlib
    - fraud_detection.utils.commons.save_json
    - fraud_detection.entity.config_entity.ModelEvaluationConfig
"""

from urllib.parse import urlparse
from pathlib import Path
import joblib

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import mlflow
import mlflow.sklearn
from fraud_detection.entity.config_entity import ModelEvaluationConfig
from fraud_detection.utils.commons import save_json


class ModelEvaluation:
    """
    Classe responsável por coletar métricas e salvá-las em ambiente
    do MLFlow.

    Irá avaliar os valores de AUC, Precision, Recall e F1 Score para
    o modelo treinado.

    Args:
        ModelEvaluationConfig (dataclass): Classe com valores de configuração.
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_metrics(self, actual, pred):
        """
        Função para retornar métricas do modelo a partir dos dados preditos
        comparando com os valores reais.

        Args:
            actual (Series): Série contendo valores reais das classes de saída.
            pred (Series): Série contendo os valores preditos das classes.

        Returns:
            tuple:
                - auc (float): Métrica de Área sob a cuva
                - precision (float): Métrica de precisão
                - recall (float): Métrica de revocação
                - f1 (float): Métrica de F1 Score
        """
        auc = roc_auc_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return auc, precision, recall, f1

    def start_mlflow(self):
        """
        Método para inicializar MLFlow e enviar métricas do modelo treinado.
        """

        # Lê os dados de teste
        X_test = pd.read_csv(self.config.test_x_data_path)
        y_test = pd.read_csv(self.config.test_y_data_path)

        # Carrega o arquivo do modelo já treinado
        model = joblib.load(self.config.model_path)

        # Configurações respectivas ao login do MLFlow
        # Para este projeto foi utilizado o DagsHub
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Cria valores de predição
            predicted_values = model.predict(X_test)

            (auc, precision, recall, f1) = self.evaluate_metrics(
                y_test, predicted_values
            )

            scores = {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            # Salva as métricas em arquivo texto e envia para MLFlow
            save_json(path=Path(self.config.metric_file_name), data=scores)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(scores)

            # Verifica se há o servidor alcançavel do MLFlow
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="LGBMClassifierModel"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
