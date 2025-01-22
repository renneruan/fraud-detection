"""
Este módulo realiza tarefas de transformação de dados para serem aplicados ao
 modelo de detecção.

Fornece funcionalidades para pré-processar e transformar os dados ao formato
 adequado. Ele inclui diversos transformadores para lidar com a imputação de
 valores ausentes, feature engineering e a divisão de conjuntos de dados.

Classes:
    CustomProcessor: Classe base customizada para criar processadores de
                     transformação.
    DropColumn: Remove colunas específicas do conjunto de dados.
    DocumentsProcessor: Processa colunas de documentos, imputando valores
                         vazios e convertendo binários em inteiros.
    CountryProcessor: Transforma a coluna de país para continentes.
    OneHotEncoderProcessor: Realiza codificação one-hot em colunas categóricas.
    DateProcessor: Realiza engenharia de features para coluna de data, criando
                    variáveis como hora e dia da semana.
    ImputeValuesProcessor: Trata valores ausentes em colunas numéricas.
    TransformColumns: Aplica transformações matemáticas (log, raiz cúbica).
    DataTransformation: Encapsula todo o pipeline de transformação de dados,
                         incluindo a divisão em treino e teste, aplicando
                         transformações e salvando os dados transformados.

Dependências:
    - pandas
    - numpy
    - sklearn
    - pycountry_convert
    - fraud_detection.logger
    - fraud_detection.entity.config_entity.DataTransformationConfig
"""

import os

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

from fraud_detection import logger
from fraud_detection.entity.config_entity import DataTransformationConfig

import pycountry_convert as pc


class CustomProcessor(BaseEstimator, TransformerMixin):
    """
    Classe para abstrair processadores base da biblioteca
    scikit-learn, permite salvar colunas a qual o processador será aplicado.

    Args:
    - cols (list): Lista de colunas a serem processadas.
    """

    def __init__(self, cols=None):
        self.cols = [] if cols is None else cols

    def fit(self, X, y=None):
        """
        Necessário para aderência ao pipeline de transformação, ele aplica
          método fit a cada etapa.
        """
        return self


class DropColumn(CustomProcessor):
    """Processador para remoção de colunas"""

    def transform(self, X):
        """
        Remove colunas que foram repassadas na criação do processador

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados sem possuir colunas desejadas.
        """
        return X.drop(self.cols, axis=1)


class DocumentsProcessor(CustomProcessor):
    """Processador direcionado para colunas de documentos"""

    def transform(self, X):
        """
        Trata as colunas de documento, preenchendo valores ausentes com o
        valor padrão "N" e posteriormente converte os valores para binário.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados com colunas de documentos processadas.
        """

        documents_cols = ["entrega_doc_1", "entrega_doc_2", "entrega_doc_3"]
        X_copy = X.copy()
        X_copy[documents_cols] = X_copy[documents_cols].fillna("N")

        # astype irá converter o tipo da coluna para inteiro (0 ou 1)
        X_copy[documents_cols] = (X_copy[documents_cols] == "Y").astype(int)

        return X_copy


class CountryProcessor(CustomProcessor):
    """Processador direcionado para coluna de país"""

    def transform(self, X):
        """
        Realiza o feature engineering na coluna de informação de país.
        Transformando a coluna em valores de continente.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados com nova coluna de continente.
        """

        X_copy = X.copy()
        # Insere valores ausentes baseado no valor mais frequente
        X_copy["pais"] = X_copy["pais"].fillna(X_copy["pais"].mode()[0])

        # Utiliza biblioteca pycountry_convert para transformar os códigos
        # de países em continentes.
        X_copy["continente"] = X_copy["pais"].apply(
            pc.country_alpha2_to_continent_code
        )

        # Remove a coluna de país para evitar duplicidade de informação
        X_copy = X_copy.drop("pais", axis=1)

        return X_copy


class DateProcessor(CustomProcessor):
    """Processador direcionado para coluna de data"""

    def transform(self, X):
        """
        Realiza o feature engineering para a coluna de data.
        Cria features de hora da compra e dia da semana.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados com novas colunas.
        """
        date_column = "data_compra"
        X_copy = X.copy()
        date = pd.to_datetime(X_copy[date_column])

        X_copy["hora_compra"] = date.dt.hour
        X_copy["dia_compra"] = date.dt.dayofweek

        # Remove coluna de data para evitar informação duplicada
        # A coluna de data removida também impede que o modelo deprecie
        # para datas mais recentes sendo inseridas.
        X_copy = X_copy.drop(date_column, axis=1)

        return X_copy


class OneHotEncoderProcessor(CustomProcessor):
    def transform(self, X):
        X_encoded = pd.get_dummies(
            X, columns=self.cols, drop_first=True, dtype=int
        )
        return X_encoded


class ImputeValuesProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, discrete_cols=None, continuous_cols=None):
        self.discrete_cols = discrete_cols
        self.continuous_cols = continuous_cols
        self.numerical_imputer = ColumnTransformer(
            transformers=[
                (
                    "discrete",
                    SimpleImputer(strategy="most_frequent"),
                    self.discrete_cols,
                ),
                (
                    "continuous",
                    SimpleImputer(strategy="mean"),
                    self.continuous_cols,
                ),
            ],
            remainder="passthrough",
        )

    def fit(self, X):
        self.numerical_imputer.fit(X)
        return self

    def transform(self, X):

        X_transformed = self.numerical_imputer.transform(X)
        X_transformed = pd.DataFrame(
            X_transformed, columns=self._get_column_names(X)
        )
        return X_transformed

    def _get_column_names(self, X):

        transformed_columns = (
            self.discrete_cols
            + self.continuous_cols
            + [
                col
                for col in X.columns
                if col not in self.discrete_cols + self.continuous_cols
            ]
        )
        return transformed_columns


class TransformColumns(CustomProcessor):
    def __init__(self, log_cols=None, cbrt_col=None):
        self.log_cols = log_cols
        self.cbrt_col = cbrt_col

    def transform(self, X):
        X_new = X.copy()
        for col in self.log_cols:
            X_new["log_" + col] = np.log1p(X_new[col].astype(float))
        X_new["cbrt_" + self.cbrt_col] = np.cbrt(
            X_new[self.cbrt_col].astype(float)
        )

        X_new = X_new.drop(self.log_cols + [self.cbrt_col], axis=1)

        return X_new


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def split_data(self, target_column):
        data = pd.read_csv(self.config.raw_data_path)

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Dividindo os dados na proporção (0.8, 0.2).
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("Dados divididos em treino e teste")

        splits = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        # Loop para salvar os arquivos
        for name, splitted_data in splits.items():
            splitted_data.to_csv(
                os.path.join(self.config.transformed_data_path, f"{name}.csv"),
                index=False,
            )

            logger.info(splitted_data.shape)

        return X_train, X_test, y_train, y_test

    def apply_target_encoder(self, X_train, X_test, y_train, column):
        category_train_df = X_train[column].to_frame(name=column)
        category_test_df = X_test[column].to_frame(name=column)

        target_encoder = TargetEncoder()

        X_train[column] = target_encoder.fit_transform(
            category_train_df, y_train
        )
        X_test[column] = target_encoder.transform(category_test_df)

        return X_train, X_test

    def convert_to_numeric(self, X):
        X_copy = X.copy()
        print(X_copy.select_dtypes(include=["object"]).columns)
        for col in X_copy.select_dtypes(include=["object"]).columns:
            try:
                X_copy[col] = pd.to_numeric(X_copy[col], errors="coerce")
            except ValueError as e:
                print(e)

        return X_copy

    def preprocessing_pipeline(self):
        X_train, X_test, y_train, _ = self.split_data(
            self.config.target_column
        )

        discrete_columns = ["score_4", "score_7"]
        continuous_columns = [
            "score_2",
            "score_3",
            "score_5",
            "score_6",
            "score_9",
            "score_10",
            "valor_compra",
        ]

        to_drop_columns = ["score_fraude_modelo", "produto", "score_8"]

        pipeline = Pipeline(
            [
                ("dropper", DropColumn(to_drop_columns)),
                (
                    "imputer",
                    ImputeValuesProcessor(
                        discrete_cols=discrete_columns,
                        continuous_cols=continuous_columns,
                    ),
                ),
                ("docs", DocumentsProcessor()),
                ("country", CountryProcessor()),
                ("date", DateProcessor()),
                ("encoder", OneHotEncoderProcessor(["score_1", "continente"])),
                (
                    "transform",
                    TransformColumns(["score_3", "valor_compra"], "score_6"),
                ),
            ]
        )

        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        (X_train_transformed, X_test_transformed) = self.apply_target_encoder(
            X_train, X_test, y_train, "categoria_produto"
        )

        X_train_transformed = self.convert_to_numeric(X_train_transformed)
        X_test_transformed = self.convert_to_numeric(X_test_transformed)

        splitted_transforms = {
            "X_train_transformed": X_train_transformed,
            "X_test_transformed": X_test_transformed,
        }

        for name, split_transform in splitted_transforms.items():
            split_transform.to_csv(
                os.path.join(self.config.transformed_data_path, f"{name}.csv"),
                index=False,
            )

            logger.info(
                "Dados divididos salvos em: %s.csv, de tamanho: %s",
                name,
                split_transform.shape,
            )
