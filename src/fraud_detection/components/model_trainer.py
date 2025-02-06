"""
Este módulo realiza o treinamento do modelo LightGBM utilizando os parâmetros
repassados como configuração ao pipeline.

Classes:
    ModelTrainer: Classe para treinamento do modelo.

Dependências:
    - os
    - pandas
    - lightgbm
    - joblib
    - fraud_detection.logger
    - fraud_detection.entity.config_entity.ModelTrainerConfig
"""

import os
import pandas as pd
import joblib
from lightgbm import LGBMClassifier

from fraud_detection import logger
from fraud_detection.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    """
    Classe para treinamento do modelo LightGBM.

    Args:
        ModelTrainerConfig (dataclass): Configurações do treinamento.
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """
        Método para treinamento do classificado LGBM.
        Treina o modelo com os dados transformados, e os parâmetros
        e salva em arquivo joblib.
        """

        # Lê os arquivos de treino, espera-se que estes já estejam
        # transformados e possuindo apenas valores numéricos.
        X_train = pd.read_csv(self.config.train_x_data_path)
        y_train = pd.read_csv(self.config.train_y_data_path)

        lgbm = LGBMClassifier(
            subsample=self.config.subsample,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,
            num_leaves=self.config.num_leaves,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            colsample_bytree=self.config.colsample_bytree,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=42,
        )

        # Utilizando ravel para garantir y como array 1D
        model = lgbm.fit(X_train, y_train.values.ravel())
        logger.info("Modelo treinado")
        logger.info(model)

        joblib.dump(
            model,
            os.path.join(
                self.config.model_target_path, self.config.model_name
            ),
        )
