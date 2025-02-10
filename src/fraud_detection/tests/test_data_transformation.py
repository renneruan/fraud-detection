"""
Módulo de teste para etapa de pré-processamento dos dados.

Armazena os testes unitários referentes a todos os passos do pipeline criado.
"""

import pytest
import numpy as np
import pandas as pd
from fraud_detection.components.data_transformation import (
    DropColumns,
    DocumentsProcessor,
    DateProcessor,
    NonFrequentAggregator,
    OneHotEncoderProcessor,
    ImputeValuesProcessor,
    TargetEncoderTransformer,
    TransformColumns,
)


def test_drop_columns():
    """Teste para verificar exclusão de colunas, DropColumns"""

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

    assert isinstance(
        transformed_data, pd.DataFrame
    ), "Não foi retornado um DataFrame"
    assert (
        "score_fraude_modelo" not in transformed_data.columns
        and "produto" not in transformed_data.columns
        and "score_8" not in transformed_data.columns
    ), "Colunas não excluídas como esperado."

    # Não deve excluir score
    assert (
        "score_1" in transformed_data.columns
    ), "Coluna excluída de forma indevida"


def test_documents_processor():
    """
    Teste para verificar transformação de colunas de documentos
    DocumentsProcessor
    """

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

    assert isinstance(
        transformed_data, pd.DataFrame
    ), "Não foi retornado um DataFrame"

    assert (
        transformed_data["entrega_doc_1"].dtype == int
    ), "Documentos não transformados para numérico"

    assert (
        transformed_data["entrega_doc_1"].tolist() == [1, 0, 0]
        and transformed_data["entrega_doc_2"].tolist() == [0, 1, 0]
        and transformed_data["entrega_doc_3"].tolist() == [0, 0, 1]
    ), "Colunas de documentos não convertidas corretamente."

    assert transformed_data["score_1"].tolist() == [
        1,
        2,
        3,
    ], "Coluna indevidamente alterada."


def test_date_processor():
    """
    Testes para pré-processamento de coluna de Data
    DateProcessor
    """
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

    assert isinstance(
        transformed_data, pd.DataFrame
    ), "Não retornado um DataFrame"

    assert (
        "hora_compra" in transformed_data.columns
        and "dia_compra" in transformed_data.columns
        and "turno_compra" in transformed_data.columns
    ), "Colunas derivadas não criadas corretamente"

    assert (
        transformed_data["hora_compra"].to_list() == [8, 14, 23, 3]
        and transformed_data["dia_compra"].to_list() == [2, 3, 4, 4]
        and transformed_data["turno_compra"].to_list() == [0, 1, 2, 3]
    ), "Valores das novas colunas não calculados corretamente"

    assert (
        "data_compra" not in transformed_data.columns
    ), "Coluna de data original não excluída"


def test_one_hot_encoder_processor():
    """
    Teste para o ajuste do encoder de categorias.
    OneHotEncoderProcessor
    """

    data_train = pd.DataFrame(
        {
            "score_1": [1, 2, 3, 4],
            "continente": ["NA", "EU", "EU", "SA"],
        }
    )

    data_test = pd.DataFrame(
        {
            "score_1": [1, 2, 2, 6],
            "continente": ["NA", "OC", "OC", "SA"],
        }
    )
    processor = OneHotEncoderProcessor()

    with pytest.raises(
        ValueError, match="O encoder não foi ajustado previamente"
    ):
        processor.transform(data_train)

    processor = OneHotEncoderProcessor()
    processor.fit(data_train)

    assert processor.fitted is True, "Encoder não alterado corretamente"
    assert (
        len(processor.encoder.categories_) == 2
    ), "Categorias não ajustadas ao encoder"

    transformed_data = processor.transform(data_test)

    assert isinstance(
        transformed_data, pd.DataFrame
    ), "Não retornou um DataFrame"

    expected_columns = processor.encoder.get_feature_names_out(
        ["score_1", "continente"]
    )

    assert all(
        col in transformed_data.columns for col in expected_columns
    ), "Colunas novas geradas pelo Encoder não estão presentes na saída"
    assert transformed_data.shape[1] == len(
        expected_columns
    ), "Tamanho de colunas geradas pelo encoder incorreto"

    assert (
        "score_1" not in transformed_data.columns
        and "continente" not in transformed_data.columns
    ), "Colunas originais ainda presentes nos dados codificados."

    assert transformed_data["score_1_1"].to_list() == [
        1,
        0,
        0,
        0,
    ] and transformed_data["continente_NA"].to_list() == [
        1,
        0,
        0,
        0,
    ], "Valores codificados incorretos"

    assert (
        "score_1_3" in transformed_data.columns
        and "continente_EU" in transformed_data.columns
    ), "Colunas de codificação não criadas corretamente."

    assert (
        not (transformed_data["score_1_4"] != 0).any()
        and not (transformed_data["continente_EU"] != 0).any()
    ), "Colunas presentes no treino mas não no teste não codificadas."

    assert (
        "score_1_6" not in transformed_data.columns
        and "continente_OC" not in transformed_data.columns
    ), "Colunas presentes no teste codificadas incorretamente"


def test_impute_values_processor():
    """
    Teste para etapa do pré-processamento de impute de valores vazios
    ImputeValuesProcessor
    """

    columns = {
        "discrete": ["score_4", "score_7"],
        "continuous": [
            "score_2",
            "score_3",
            "score_5",
            "score_6",
            "score_9",
            "score_10",
            "valor_compra",
        ],
    }

    train_values = {
        "discrete": [1, 2, 2, 3, 4],
        "continuous": [
            10,
            10.5,
            20.3,
            15.2,
            23.5,
        ],
    }

    test_values = {
        "discrete": [np.nan, 1, 2, np.nan, 3],
        "continuous": [
            np.nan,
            10.5,
            20.3,
            15.2,
            np.nan,
        ],
    }

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for category, col_list in columns.items():
        for col in col_list:
            train_data[col] = train_values[category]
            test_data[col] = test_values[category]

    train_data["non_numeric_column"] = "A"
    test_data["non_numeric_column"] = "B"

    processor = ImputeValuesProcessor()
    processor.fit(train_data)

    transformed = processor.transform(test_data)

    assert (
        not transformed.isnull().any().any()
    ), "Ainda há valores nulos após Imputer aplicado"

    assert transformed["score_4"].to_list() == [2, 1, 2, 2, 3] and transformed[
        "score_2"
    ].to_list() == [
        15.2,
        10.5,
        20.3,
        15.2,
        15.2,
    ], "Valores de preenchimento de nulo diferentes do esperado"

    assert (
        "non_numeric_column" in transformed.columns
    ), "Coluna alterada indevidamente"


def test_transform_columns():
    """Teste para etapa de transformações numéricas (log)"""
    data = pd.DataFrame(
        {
            "score_1": [10, 20, 30, 40, 50],
            "score_3": [1, 2, 3, 4, 5],
            "valor_compra": [100, 200, 300, 400, 500],
        }
    )

    processor = TransformColumns()

    transformed = processor.transform(data)
    assert (
        "log_score_3" in transformed.columns
        and "log_valor_compra" in transformed.columns
        and "score_1" in transformed.columns
    ), "Colunas após transformação diferentes do esperado."

    assert (
        "score_3" not in transformed.columns
        and "valor_compra" not in transformed.columns
    ), "Colunas originais não excluídas"

    expected_log_score_3 = np.log1p(data["score_3"])

    assert (
        transformed["log_score_3"].to_list() == expected_log_score_3.to_list()
        and transformed["score_1"].to_list() == data["score_1"].to_list()
    ), "Valores de log não aplicadados corretamente"


def test_non_frequent_aggregator():
    """
    Testes para etapa de agregar valores não frequentes
    NonFrequentAggregator
    """

    data = pd.DataFrame(
        {
            "categoria_produto": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "D",
                "D",
                "D",
                "E",
                "E",
                "F",
            ],
            "valor": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        }
    )

    aggregator = NonFrequentAggregator()
    aggregator.fit(data)

    transformed = aggregator.transform(data)
    assert isinstance(transformed, pd.DataFrame)

    assert (
        "categoria_produto_reduzida" in transformed.columns
    ), "A coluna reduzida não foi criada corretamente"

    assert (
        "categoria_produto" not in transformed.columns
    ), "A coluna original não foi removida"

    # Verificar categorias agregadas
    # A categoria D é a única com mais de 2 ocorrências
    assert (
        transformed.loc[
            data["categoria_produto"] == "D", "categoria_produto_reduzida"
        ].nunique()
        == 1
    ), "Categoria que não deveria ser agregada foi modificada."

    assert list(
        transformed["categoria_produto_reduzida"].value_counts().index
    ) == [
        "Outros",
        "D",
    ], "Colunas após agregamento incorretas."
    assert list(
        transformed["categoria_produto_reduzida"].value_counts().values
    ) == [
        8,
        3,
    ], "Valores de quantias agregadas incorretos"

    assert (
        "valor" in transformed.columns
    ), "Coluna 'valor' foi removida incorretamente"


def test_target_encoder_transformer():
    """
    Testes para o TargetEncoder de colunas categóricas.
    """

    data_train = pd.DataFrame(
        {
            "categoria_produto_reduzida": [
                "A",
                "B",
                "A",
                "C",
                "B",
                "C",
                "C",
                "A",
            ],
            "valor": [100, 200, 150, 300, 250, 400, 350, 125],
        }
    )
    y_train = pd.Series([1, 0, 1, 0, 0, 1, 1, 1])

    data_test = pd.DataFrame(
        {
            "categoria_produto_reduzida": [
                "A",
                "B",
                "C",
                "D",
            ],
            "valor": [130, 210, 310, 400],
        }
    )

    encoder = TargetEncoderTransformer()
    encoder.fit(data_train, y_train)

    transformed = encoder.transform(data_test)

    assert (
        transformed["categoria_produto_reduzida"].dtype == np.float64
    ), "A coluna transformada deve ser numérica"

    unique_values = transformed["categoria_produto_reduzida"].unique()
    assert len(unique_values) > 1, "O encoding não foi aplicado corretamente"

    assert not np.isnan(
        transformed.loc[
            data_test["categoria_produto_reduzida"] == "D",
            "categoria_produto_reduzida",
        ]
    ).all(), "Categoria nova não codificada"

    expected_value_A = y_train[
        data_train["categoria_produto_reduzida"] == "A"
    ].mean()
    actual_value_A = transformed.loc[
        data_test["categoria_produto_reduzida"] == "A",
        "categoria_produto_reduzida",
    ].iloc[0]
    assert np.isclose(
        actual_value_A, expected_value_A
    ), "Valor incorreto obtido"

    assert (
        "valor" in transformed.columns
    ), "Coluna 'valor' foi removida incorretamente"
