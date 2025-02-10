"""
Módulo com funções de endpoint a serem servidas pela API Flask.
"""

from flask import Flask, render_template, request
import pandas as pd
from fraud_detection.pipeline.prediction import PredictionPipeline
from fraud_detection import logger


app = Flask(__name__)


@app.route("/", methods=["GET"])
def homePage():
    """Home page a ser renderizada"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def index():
    """Endpoint de predição para verificação de fraude"""
    try:
        score_1 = int(request.form["score_1"])
        score_2 = float(request.form["score_2"])
        score_3 = float(request.form["score_3"])
        score_4 = float(request.form["score_4"])
        score_5 = float(request.form["score_5"])
        score_6 = float(request.form["score_6"])
        score_7 = int(request.form["score_7"])
        score_8 = float(request.form["score_8"])
        score_9 = float(request.form["score_9"])
        score_10 = float(request.form["score_10"])
        valor_compra = float(request.form["valor_compra"])
        pais = str(request.form["pais"])
        produto = str(request.form["produto"])
        categoria_produto = str(request.form["categoria_produto"])
        data_compra = str(request.form["data_compra"])
        entrega_doc_1 = str(request.form["entrega_doc_2"])
        entrega_doc_2 = str(request.form["entrega_doc_2"])
        entrega_doc_3 = str(request.form["entrega_doc_3"])

        data = [
            score_1,
            score_2,
            score_3,
            score_4,
            score_5,
            score_6,
            score_7,
            score_8,
            score_9,
            score_10,
            valor_compra,
            pais,
            produto,
            categoria_produto,
            data_compra,
            entrega_doc_1,
            entrega_doc_2,
            entrega_doc_3,
        ]
        logger.info("predict data:")
        logger.info(data)

        data = pd.DataFrame([data])
        data.columns = [
            "score_4",
            "score_7",
            "score_2",
            "score_5",
            "score_9",
            "score_10",
            "categoria_produto",
            "entrega_doc_1",
            "entrega_doc_2",
            "entrega_doc_3",
            "hora_compra",
            "dia_compra",
            "score_1_2",
            "score_1_3",
            "score_1_4",
            "continente_AS",
            "continente_EU",
            "continente_NA",
            "continente_OC",
            "continente_SA",
            "log_score_3",
            "log_valor_compra",
            "cbrt_score_6",
        ]

        print(data)
        obj = PredictionPipeline()
        data = obj.transform_input_data(data)

        print(data)
        predict, predict_proba = obj.predict(data)

        results = {predict, predict_proba}
        logger.info("predict results:")
        logger.info(results)

        resultado = f"{results}"
        return resultado

    except ValueError as e:
        print(f"Não foi possível: {e}")
        return "falha"


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port = 8080, debug=True)
    app.run(host="0.0.0.0", port=8080)
