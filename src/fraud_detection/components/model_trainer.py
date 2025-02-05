from lightgbm import LGBMClassifier
from fraud_detection.entity.config_entity import ModelTrainerConfig
import pandas as pd
import os
from fraud_detection import logger
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = pd.read_csv(self.config.train_x_data_path)
        y_train = pd.read_csv(self.config.train_y_data_path)

        lgbm = LGBMClassifier(
            is_unbalance=True,
            subsample=self.config.subsample,
            reg_lambda=self.config.reg_lambda,
            reg_alpha=self.config.reg_alpha,
            num_leaves=self.config.num_leaves,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            colsample_bytree=self.config.colsample_bytree,
            random_state=42,
        )
        model = lgbm.fit(X_train, y_train)
        logger.info("Model trained")
        logger.info(model)

        joblib.dump(
            model,
            os.path.join(
                self.config.model_target_path, self.config.model_name
            ),
        )
