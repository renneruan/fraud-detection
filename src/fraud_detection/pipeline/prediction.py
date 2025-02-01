"""
Módulo do pipeline utilizado para predição.
A partir do dado recebido carrega o modelo, aplica o pré-processamento
adequado aos dados de entrada e devolve a probabilidade da classe.
"""

import joblib
from pathlib import Path

MODEL_PATH = "artifacts/model_output/model.joblib"


class PredictionPipeline:
    "a"

    def __init__(self):
        self.model = joblib.load(Path(MODEL_PATH))

    def predict(self, data):
        "b"
        prediction = self.model.predict(data)
        prediction_proba = self.model.predict_proba(data)
        return (prediction[0], prediction_proba[0][1])

    def predict2(self, data):
        "c"
        prediction = self.model.predict(data)
        prediction_proba = self.model.predict_proba(data)
        return (prediction[0], prediction_proba[0][1])
