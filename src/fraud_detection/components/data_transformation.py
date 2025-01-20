import os

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from fraud_detection.entity.config_entity import DataTransformationConfig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from fraud_detection import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

import pycountry_convert as pc


class CustomProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, X, y=None):
        return self


class DropColumn(CustomProcessor):
    def transform(self, X):
        return X.drop(self.cols, axis=1)


class DocumentsProcessor(CustomProcessor):
    def transform(self, X):
        X_new = X.copy()
        X_new[self.cols] = X_new[self.cols].fillna("N")
        X_new[self.cols] = (X_new[self.cols] == "Y").astype(int)

        return X_new


class CountryProcessor(CustomProcessor):
    def transform(self, X):
        X_new = X.copy()
        X_new["pais"] = X_new["pais"].fillna(X_new["pais"].mode()[0])

        X_new["continente"] = X_new["pais"].apply(
            lambda x: pc.country_alpha2_to_continent_code(x)
        )

        X_new = X_new.drop("pais", axis=1)

        return X_new


class OneHotEncoderProcessor(CustomProcessor):
    def transform(self, X):
        X_encoded = pd.get_dummies(X, columns=self.cols, drop_first=True, dtype=int)
        return X_encoded


class DateProcessor(CustomProcessor):
    def __init__(self, date_col, hour_col, day_col):
        self.date_col = date_col
        self.hour_col = hour_col
        self.day_col = day_col

    def transform(self, X):
        X_new = X.copy()
        date = pd.to_datetime(X_new[self.date_col])

        X_new[self.hour_col] = date.dt.hour
        X_new[self.day_col] = date.dt.dayofweek

        X_new = X_new.drop(self.date_col, axis=1)

        return X_new


class ImputeValuesProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, discrete_cols=[], continuous_cols=[]):
        self.discrete_cols = discrete_cols
        self.continuous_cols = continuous_cols
        self.numerical_imputer = ColumnTransformer(
            transformers=[
                (
                    "discrete",
                    SimpleImputer(strategy="most_frequent"),
                    self.discrete_cols,
                ),
                ("continuous", SimpleImputer(strategy="mean"), self.continuous_cols),
            ],
            remainder="passthrough",
        )

    def fit(self, X, y=None):
        self.numerical_imputer.fit(X)
        return self

    def transform(self, X):

        X_transformed = self.numerical_imputer.transform(X)
        X_transformed = pd.DataFrame(X_transformed, columns=self._get_column_names(X))
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
    def __init__(self, log_cols=[], cbrt_col=None):
        self.log_cols = log_cols
        self.cbrt_col = cbrt_col

    def transform(self, X):
        X_new = X.copy()
        for col in self.log_cols:
            X_new["log_" + col] = np.log1p(X_new[col].astype(float))
        X_new["cbrt_" + self.cbrt_col] = np.cbrt(X_new[self.cbrt_col].astype(float))

        X_new = X_new.drop(self.log_cols + [self.cbrt_col], axis=1)

        return X_new


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def split_data(self):
        data = pd.read_csv(self.config.raw_data_path)

        X = data.drop("fraude", axis=1)
        y = data["fraude"]

        # Split the data into training and test sets. (0.75, 0.25) split.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("Splited data into training and test sets")

        splits = {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }

        # Loop para salvar os arquivos
        for name, splitted_data in splits.items():
            splitted_data.to_csv(
                os.path.join(self.config.transformed_data_path, f"{name}.csv"),
                index=False,
            )

            logger.info(splitted_data.shape)
            print(splitted_data.shape)

    def apply_target_encoder(self):
        category_train_df = self.X_train_transformed["categoria_produto"].to_frame(
            name="categoria_produto"
        )
        category_test_df = self.X_test_transformed["categoria_produto"].to_frame(
            name="categoria_produto"
        )

        target_encoder = TargetEncoder()

        self.X_train_transformed["categoria_produto"] = target_encoder.fit_transform(
            category_train_df, self.y_train
        )
        self.X_test_transformed["categoria_produto"] = target_encoder.transform(
            category_test_df
        )

    def convert_to_numeric(self, input_data):
        new_data = input_data.copy()
        print(new_data.select_dtypes(include=["object"]).columns)
        for col in new_data.select_dtypes(include=["object"]).columns:
            try:
                new_data[col] = pd.to_numeric(new_data[col], errors="coerce")
            except ValueError as e:
                print(e)
                pass

        return new_data

    def preprocessing_pipeline(self):
        self.split_data()

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

        documents_columns = ["entrega_doc_1", "entrega_doc_2", "entrega_doc_3"]

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
                ("docs", DocumentsProcessor(documents_columns)),
                ("country", CountryProcessor()),
                ("date", DateProcessor("data_compra", "hora_compra", "dia_compra")),
                ("encoder", OneHotEncoderProcessor(["score_1", "continente"])),
                (
                    "transform",
                    TransformColumns(["score_3", "valor_compra"], "score_6"),
                ),
            ]
        )

        self.X_train_transformed = pipeline.fit_transform(self.X_train)
        self.X_test_transformed = pipeline.transform(self.X_test)

        self.apply_target_encoder()

        self.X_train_transformed = self.convert_to_numeric(self.X_train_transformed)
        self.X_test_transformed = self.convert_to_numeric(self.X_test_transformed)

        splitted_transforms = {
            "X_train_transformed": self.X_train_transformed,
            "X_test_transformed": self.X_test_transformed,
        }

        for name, split_transform in splitted_transforms.items():
            split_transform.to_csv(
                os.path.join(self.config.transformed_data_path, f"{name}.csv"),
                index=False,
            )

            logger.info(split_transform.shape)
            print(split_transform.shape)
