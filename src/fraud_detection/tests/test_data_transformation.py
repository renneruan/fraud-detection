import pandas as pd
from fraud_detection.components.data_transformation import (
    DropColumns,
    DocumentsProcessor,
    DateProcessor,
    OneHotEncoderProcessor,
    ImputeValuesProcessor,
)


def test_drop_columns():
    data = pd.DataFrame(
        {
            "score_1": [10, 20, 30],
            "score_fraude_modelo": [0.1, 0.2, 0.3],
            "produto": ["A", "B", "C"],
            "score_8": [1, 2, 3],
        }
    )
    processor = DropColumns()
    transformed_data = processor.transform(data)

    assert "score_fraude_modelo" not in transformed_data.columns
    assert "produto" not in transformed_data.columns
    assert "score_8" not in transformed_data.columns
    assert "score_1" in transformed_data.columns


def test_documents_processor():
    data = pd.DataFrame(
        {
            "entrega_doc_1": [1, 0, 0],
            "entrega_doc_2": [None, "Y", "N"],
            "entrega_doc_3": ["N", None, "Y"],
            "score_1": [1, 2, 3],
        }
    )
    processor = DocumentsProcessor()
    transformed_data = processor.transform(data)

    assert transformed_data["entrega_doc_1"].dtype == int

    assert transformed_data["entrega_doc_1"].tolist() == [1, 0, 0]
    assert transformed_data["entrega_doc_2"].tolist() == [0, 1, 0]
    assert transformed_data["entrega_doc_3"].tolist() == [0, 0, 1]
    assert transformed_data["score_1"].tolist() == [
        1,
        2,
        3,
    ]


def test_date_processor():
    data = pd.DataFrame(
        {
            "data_compra": [
                "2025-01-01 08:30:00",
                "2025-01-02 14:15:00",
                "2025-01-03 23:45:00",
                "2025-01-03 03:45:00",
            ]
        }
    )
    processor = DateProcessor()
    transformed_data = processor.transform(data)

    assert "hora_compra" in transformed_data.columns
    assert "dia_compra" in transformed_data.columns
    assert "turno_compra" in transformed_data.columns

    assert transformed_data["hora_compra"].to_list() == [8, 14, 23, 3]
    assert transformed_data["dia_compra"].to_list() == [2, 3, 4, 4]

    assert transformed_data["turno_compra"].to_list() == [0, 1, 2, 3]

    assert "data_compra" not in transformed_data.columns


def test_one_hot_encoder_processor():
    data = pd.DataFrame(
        {
            "score_1": ["A", "B", "A"],
            "continente": ["América", "Europa", "Ásia"],
        }
    )
    processor = OneHotEncoderProcessor()
    processor.fit(data)
    transformed_data = processor.transform(data)

    assert all(
        col in transformed_data.columns
        for col in processor.encoder.get_feature_names_out()
    )
    assert "score_1" not in transformed_data.columns
    assert "continente" not in transformed_data.columns


# def test_impute_values_processor():
#     data = pd.DataFrame(
#         {
#             "score_4": [1, None, 3],
#             "score_7": [None, 2, 3],
#             "score_2": [1.5, None, 2.5],
#             "valor_compra": [100.0, None, 200.0],
#         }
#     )
#     processor = ImputeValuesProcessor()
#     processor.fit(data)
#     transformed_data = processor.transform(data)

#     assert (
#         transformed_data.isnull().sum().sum() == 0
#     )
