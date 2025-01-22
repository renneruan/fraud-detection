import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import joblib
from fraud_detection.entity.config_entity import ModelEvaluationConfig
from fraud_detection.utils.commons import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_metrics(self, actual, pred):
        auc = roc_auc_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return auc, precision, recall, f1

    def start_mlflow(self):

        X_test = pd.read_csv(self.config.test_x_data_path)
        y_test = pd.read_csv(self.config.test_y_data_path)

        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_values = model.predict(X_test)

            (auc, precision, recall, f1) = self.evaluate_metrics(
                y_test, predicted_values
            )

            scores = {"auc": auc, "precision": precision, "recall": recall, "f1": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metrics(scores)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="LGBMClassifierModel"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
