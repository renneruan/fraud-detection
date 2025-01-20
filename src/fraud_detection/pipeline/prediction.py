import joblib
from pathlib import Path

MODEL_PATH = "artifacts/model_output/model.joblib"


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path(MODEL_PATH))

    def predict(self, data):
        prediction = self.model.predict(data)
        prediction_proba = self.model.predict_proba(data)
        return (prediction[0], prediction_proba[0][1])
