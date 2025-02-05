"""
Este módulo realiza tarefas de transformação de dados para serem aplicados ao
 modelo de detecção.

Fornece funcionalidades para pré-processar e transformar os dados ao formato
 adequado. Ele inclui diversos transformadores para lidar com a imputação de
 valores ausentes, feature engineering e a divisão de conjuntos de dados.

Classes:
    CustomProcessor: Classe base customizada para criar processadores de
                     transformação.
    DropColumns: Remove colunas específicas do conjunto de dados.
    DocumentsProcessor: Processa colunas de documentos, imputando valores
                         vazios e convertendo binários em inteiros.
    CountryProcessor: Transforma a coluna de país para continentes.
    OneHotEncoderProcessor: Realiza codificação one-hot em colunas categóricas.
    DateProcessor: Realiza engenharia de features para coluna de data, criando
                    variáveis como hora e dia da semana.
    ImputeValuesProcessor: Trata valores ausentes em colunas numéricas.
    TransformColumns: Aplica transformações matemáticas (log, raiz cúbica).
    NonFrequentAggregator: Transforma dados de categoria de produto não
                            frequentes em uma única classificação Outros.
    TargetEncoderTransformer: Aplica TargetEncoder a coluna de categoria de
                               produtos.
    DataTransformation: Encapsula todo o pipeline de transformação de dados,
                         incluindo a divisão em treino e teste, aplicando
                         transformações e salvando os dados transformados.


Dependências:
    - pandas
    - numpy
    - sklearn
    - joblib
    - pycountry_convert
    - fraud_detection.logger
    - fraud_detection.entity.config_entity.DataTransformationConfig
"""

import os

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, OneHotEncoder

import pycountry_convert as pc
from fraud_detection import logger
from fraud_detection.entity.config_entity import DataTransformationConfig


# Classe que servirá para o pai dos transformadores customizados
class CustomProcessor(BaseEstimator, TransformerMixin):
    """
    Classe para abstrair processadores base da biblioteca
    scikit-learn, sobrescreve o método fit para um retorno arbitrário.

    Utilizamos para garantir a aplicabilidade da transformação ao processo
    do fit_transform do pipeline, porém em que as alterações serão ditas
    apenas pela função transform, ou seja não serão armazenadas informações
    no fit para evitar data leakege, as etapas que necessitam este
    armazenamento não utilizarão este placeholder.
    """

    def __init__(self):
        pass

    def fit(self, _, _y=None):
        """
        Sobrescrita do fit para compatibilidade com classes
        pais do scikit-learn, as transformações serão aplicadas com a
        função transform.
        """
        return self


class DropColumns(CustomProcessor):
    """
    Função do pipeline de transformação para exclusão de colunas.
    As colunas foram selecionadas a partir da análise estatística dos dados
    ou não possuem informações práticas para o modelo, ou apresentam
    distribuições não aderentes como uniforme ou de com alta cardinalidade
    de categorias.

    Colunas a serem excluídas: "score_fraude_modelo", "produto", "score_8"
    """

    def __init__(self):
        self.drop_columns = ["score_fraude_modelo", "produto", "score_8"]

    def transform(self, X):
        """
        Retorna os dados de entrada sem a presença das colunas declaradas
        na inicialização da classe.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados sem as colunas desejadas.
        """
        return X.drop(self.drop_columns, axis=1)


class DocumentsProcessor(CustomProcessor):
    """
    Processador direcionado para transformações nas colunas de documentos.

    Colunas transformadas: "entrega_doc_1", "entrega_doc_2", "entrega_doc_3"
    """

    def __init__(self):
        self.document_columns = [
            "entrega_doc_1",
            "entrega_doc_2",
            "entrega_doc_3",
        ]

    def transform(self, X):
        """
        Trata as colunas de documento, preenchendo valores ausentes com o
        valor padrão "N" e posteriormente converte os valores para binário.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados com colunas de documentos processadas.
        """
        X_new = X.copy()
        X_new[self.document_columns] = (
            X_new[self.document_columns].astype(str).fillna("N")
        )
        X_new[self.document_columns] = (
            X_new[self.document_columns] == "Y"
        ).astype(int)

        return X_new


class CountryProcessor(CustomProcessor):
    """Processador direcionado para transformação da coluna país."""

    def __init__(self):
        self.country_column = "pais"

    def transform(self, X):
        """
        Realiza o feature engineering na coluna de país da compra.
        Transformando a coluna em uma nova contendo o continente.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados com nova coluna de continente.
        """

        X_new = X.copy()
        # Insere valores ausentes baseado no valor mais frequente
        X_new[self.country_column] = X_new[self.country_column].fillna(
            X_new[self.country_column].mode()[0]
        )

        # Utiliza biblioteca pycountry_convert para transformar os códigos
        # de países em continentes.
        X_new["continente"] = X_new[self.country_column].apply(
            pc.country_alpha2_to_continent_code
        )
        # Remove a coluna de país para evitar duplicidade de informação
        X_new = X_new.drop(self.country_column, axis=1)

        return X_new


class DateProcessor(CustomProcessor):
    """
    Processador direcionado para transformações da coluna de data da compra.
    """

    def __init__(self):
        self.date_column = "data_compra"

    def _hour_to_period(self, hour):
        """
        Função para ser mapeada aos dados, criando a coluna turno.

        Args:
        - hour (int): Valor inteiro da hora da compra.

        Returns:
        - int: Inteiro representando o turno da compra.
        """
        if 6 <= hour < 12:
            return 0  # "Manhã"
        if 12 <= hour < 18:
            return 1  # "Tarde"
        if 18 <= hour < 24:
            return 2  # "Noite"

        return 3  # "Madrugada"

    def transform(self, X):
        """
        Realiza o feature engineering para a coluna de data.
        Cria features de hora da compra, dia da semana da compra e turno.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados com novas colunas relacionadas a data.
        """
        X_new = X.copy()
        date = pd.to_datetime(X_new[self.date_column])

        X_new["hora_compra"] = date.dt.hour
        X_new["dia_compra"] = date.dt.dayofweek
        X_new["turno_compra"] = X_new["hora_compra"].apply(
            self._hour_to_period
        )

        # Remove coluna de data para evitar informação duplicada
        # A coluna de data removida também impede que o modelo deprecie
        # para datas mais recentes sendo inseridas.
        X_new = X_new.drop(self.date_column, axis=1)

        return X_new


class OneHotEncoderProcessor(CustomProcessor):
    """
    Classe de transformação para aplicar OneHotEncoder a colunas categóricas
    de baixa cardinalidade.

    Utiliza OneHotEncoder provindo do pacote scikit-learn.

    Lida com dados não desconhecidos com a tratativa "ignore"

    Colunas que serão aplicadas: "score_1", "continente"
    """

    def __init__(self):
        self.columns_to_encode = ["score_1", "continente"]
        self.encoder = OneHotEncoder(
            sparse_output=False, dtype=int, handle_unknown="ignore"
        )
        self.fitted = False

    def fit(self, X, _y=None):
        """
        Realiza o fit do encoder com os dados de treino.

        Args:
        - X (pd.DataFrame): Dados de treino de entrada.

        Returns:
        - OneHotEncoderProcessor: Processador com dados ajustados.
        """
        self.encoder.fit(X[self.columns_to_encode])
        self.fitted = True
        return self

    def transform(self, X):
        """
        Aplica o OneHotEncoding utilizando o processador previamente ajustado.

        Args:
        - X (pd.DataFrame): Conjunto de dados originais.

        Returns:
        - pd.DataFrame: Dados com novas colunas de one hot encoding.
        """
        if not self.fitted:
            raise ValueError(
                "O encoder não foi ajustado previamente (Fit necessário)."
            )

        X_new = X.copy()

        # Cria novas colunas com Hot Encodings a partir dos dados recebidos
        encoded_columns = self.encoder.transform(X_new[self.columns_to_encode])

        # Transforma as colunas de resultado em um DataFrame
        encoded_df = pd.DataFrame(
            encoded_columns,
            columns=self.encoder.get_feature_names_out(self.columns_to_encode),
            index=X_new.index,
        )

        # Remove as colunas originais dos dados de entrada
        X_new = X_new.drop(columns=self.columns_to_encode)

        # Concatena os dados originais as colunas de hot encoding criadas
        X_encoded = pd.concat([X_new, encoded_df], axis=1)
        return X_encoded


class ImputeValuesProcessor(CustomProcessor):
    """
    Classe para aplicar transformações nas colunas numéricas existentes.
    Aplica as classes discretas e contínuas aos respectivos Imputers,
    aplicando-os a um ColumnTransformer.

    Utiliza a moda para colunas numéricas discretas e mediana para colunas
    contínuas.

    Colunas discretas: "score_4" e "score_7"
    Colunas contínuas: "score_2", "score_3", "score_5", "score_6", "score_9",
     "score_10" e "valor_compra"
    """

    def __init__(self):
        self.discrete_columns = ["score_4", "score_7"]
        self.continuous_columns = [
            "score_2",
            "score_3",
            "score_5",
            "score_6",
            "score_9",
            "score_10",
            "valor_compra",
        ]

        # Utiliza dois imputers para os tipos numéricos diferentes
        # Aplica ambos a um ColumnTransformer
        self.numerical_imputer = ColumnTransformer(
            transformers=[
                (
                    "discrete",
                    SimpleImputer(strategy="most_frequent"),
                    self.discrete_columns,
                ),
                (
                    "continuous",
                    SimpleImputer(strategy="median"),
                    self.continuous_columns,
                ),
            ],
            # Configuração que evita o transformador de excluir os dados
            # não utilizados para a resposta.
            remainder="passthrough",
        )

    def fit(self, X, _y=None):
        """
        Realiza o fit do imputer numérico com os dados de treino.

        Args:
        - X (pd.DataFrame): Dados de treino de entrada.

        Returns:
        - ImputeValuesProcessor: Processador com dados ajustados.
        """
        self.numerical_imputer.fit(X)
        return self

    def transform(self, X):
        """
        Aplica os dados de entrada ao imputer numérico previamente ajustados
        aos dados de treino. Aplica novamente os labels as colunas para seguir
         o pipeline corretamente.

        Args:
        - X (pd.DataFrame): Dados de entraga a serem transformados.

        Returns:
        - pd.DataFrame: Dados com as colunas numéricas transformadas.

        """

        X_transformed = self.numerical_imputer.transform(X)
        X_transformed = pd.DataFrame(
            X_transformed, columns=self._get_column_names(X)
        )
        return X_transformed

    def _get_column_names(self, X):
        """
        Função para concatenar o nome das colunas originais com as colunas
        transformadas, uma vez que o resultado da transformação retorna em
        arrays não indexados.

        Args:
        - X (pd.DataFrame): Conjunto de dados com colunas originais.

        Returns:
        - list(str): Lista de nomes das colunas concatenadas.

        """

        transformed_columns = (
            self.discrete_columns
            + self.continuous_columns
            + [
                col
                for col in X.columns
                if col not in self.discrete_columns + self.continuous_columns
            ]
        )
        return transformed_columns


class TransformColumns(CustomProcessor):
    """
    Classe de processador para aplicar transformação log
    as colunas aplicáveis.

    Colunas: "score_3" e "valor_compra"
    """

    def __init__(self):
        self.log_columns = ["score_3", "valor_compra"]

    def transform(self, X):
        """
        Aplica transformação log as colunas ditas na inicialização.
        Insere as transformações em novas colunas e retira as originais.

        Args:
        - X (pd.DataFrame): Dados recebidos a serem transformados.

        Returns:
        - pd.DataFrame: Dados com as novas colunas logarítmicas.
        """
        X_new = X.copy()
        for col in self.log_columns:
            X_new[f"log_{col}"] = np.log1p(X_new[col].astype(float))

        X_new = X_new.drop(self.log_columns, axis=1)

        return X_new


class NonFrequentAggregator(CustomProcessor):
    """
    Classe de processador para agregas valores não frequentes de categorias
    de produto.
    """

    def __init__(self):
        self.column = "categoria_produto"
        self.valid_categories = None
        self.threshold = 2  # Limiar de no mínimo 3 arquivos para uma categoria

    def fit(self, X, _y=None):
        """
        Método de ajuste irá criar o array de categorias válidas.
        Transformações posteriores irão ser realizadas utilizando este array.

        Args:
        - X (pd.DataFrame): Dados de entrada

        Returns:
        - NonFrequentAggregator: Processador ajustado.
        """
        frequent_categories = X[self.column].value_counts()
        self.valid_categories = frequent_categories[
            frequent_categories > self.threshold
        ].index
        return self

    def transform(self, X):
        """
        Recebe os dados que irão consultar o vetor de categorias
        frequentes e agrupar os que não estão com mais de 2 ocorrências.

        Args:
        - X (pd.DataFrame): Dados a serem ajustados.

        Returns:
        - pd.DataFrame: Dados com a nova coluna agregada.
        """
        X_new = X.copy()

        new_column_name = self.column + "_reduzida"

        X_new[new_column_name] = X_new[self.column]
        X_new.loc[
            ~X_new[self.column].isin(self.valid_categories),
            new_column_name,
        ] = "Outros"

        X_new = X_new.drop(self.column, axis=1)
        return X_new


class TargetEncoderTransformer(CustomProcessor):
    """
    Transformer para aplicar Target Encoding a uma coluna categórica
    de alta cardinalidade.

    Coluna a ser aplicada: "categoria_produto_reduzida"
    """

    def __init__(self):
        self.column = "categoria_produto_reduzida"
        self.encoder = TargetEncoder(shuffle=False)

    def fit(self, X, y=None):
        """
        Ajusta o TargetEncoder usando os dados de treino.

        Args:
        - X (pd.DataFrame): Conjunto de dados de treino.
        - y (pd.Series): Coluna de classes de saída.

        Returns:
        - self: O próprio objeto transformador.
        """
        self.encoder.fit(X[[self.column]], y)
        return self

    def transform(self, X):
        """
        Transforma os dados usando o TargetEncoder previamente ajustado.

        Args:
        - X (pd.DataFrame): Conjunto de dados a ser transformado.

        Retorna:
        - pd.DataFrame: Conjunto de dados transformados.
        """
        X_transformed = X.copy()
        X_transformed[self.column] = self.encoder.transform(X[[self.column]])
        return X_transformed


class DataTransformation:
    """
    Classe que irá agregar e chamar todas as funções de processamento,
    recebendo as configurações e caminhos de arquivos de entrada e saída.

    Args:
    - DataTransformationConfig (dataclass): Classe de valores de configuração.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _save_splits(self, splits):
        """
        Método privado para dividir os pedaços de dados após divisão.

        Args:
        - splits (dict): Dicionário contendo nome e dados divididos.
        """
        for name, split_transform in splits.items():
            split_transform.to_csv(
                os.path.join(self.config.transformed_data_path, f"{name}.csv"),
                index=False,
            )

            logger.info(
                "Dados divididos salvos em: %s.csv, de tamanho: %s",
                name,
                split_transform.shape,
            )

    def _remove_outliers(self, data):
        """
        Função para remoção de outliers dos dados de entrada.

        A partir do EDA realizado temos os limiares:
          Valor máximo de 10 para "score_5"
          Valor máximo de 483 para "score_6"
        """
        data_sem_outliers = data[data["score_5"] < 10]
        data_sem_outliers = data_sem_outliers[
            data_sem_outliers["score_6"] < 483
        ]

        return data_sem_outliers

    def _split_data(self, data, target_column):
        """
        Função para realizar a divisão dos dados em treino e teste.

        Proporção de 20% para dados de teste.

        Args:
        - target_column (str): Coluna alvo contendo classes.

        Returns:
        - tuple: Uma tupla contendo quatro conjuntos de dados:
            - X_train (pd.DataFrame): Conjunto de treino.
            - X_test (pd.DataFrame): Conjunto de teste.
            - y_train (pd.DataFrame): Rótulos do conjunto de treino.
            - y_test (pd.DataFrame): Rótulos do conjunto de teste
        """
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
        self._save_splits(splits)

        return X_train, X_test, y_train, y_test

    def convert_to_numeric(self, X):
        """
        Função para converser colunas de objeto para numérico.
        Será utilizado após as transformações, garantindo uma correta tipagem
        dos dados de entrada do modelo.

        Args:
        - X (pd.DataFrame): Dados a serem convertidos em numérico.

        Returns:
        - pd.DataFrame: Dados após a conversão numérica.
        """
        X_new = X.copy()
        for col in X_new.select_dtypes(include=["object"]).columns:
            try:
                X_new[col] = pd.to_numeric(X_new[col], errors="coerce")
            except ValueError as e:
                print(e)

        return X_new

    def preprocessing_pipeline(self):
        """
        Método para realizar chamada do processamento dos dados.

        Irá criar o pipeline de processamento utilizando as classes de
        processadores presentes neste módulo.

        Remove outliers, divide os dados, processa e salva os resultados.
        """
        data = pd.read_csv(self.config.raw_data_path)

        # Remove outliers antes da divisão de treino e teste
        data_without_outliers = self._remove_outliers(data)

        X_train, X_test, y_train, _ = self._split_data(
            data_without_outliers, self.config.target_column
        )

        pipeline = Pipeline(
            [
                ("dropper", DropColumns()),
                ("imputer", ImputeValuesProcessor()),
                ("docs", DocumentsProcessor()),
                ("country", CountryProcessor()),
                ("date", DateProcessor()),
                ("encoder", OneHotEncoderProcessor()),
                ("transform", TransformColumns()),
                ("column_aggregator", NonFrequentAggregator()),
                ("target_encoder", TargetEncoderTransformer()),
            ]
        )

        # Aplica pipeline de processamento para as colunas
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        X_test_transformed = pipeline.transform(X_test)

        # Garante que as colunas transformadas sejam numéricas para modelo
        X_train_transformed = self.convert_to_numeric(X_train_transformed)
        X_test_transformed = self.convert_to_numeric(X_test_transformed)

        splitted_transforms = {
            "X_train_transformed": X_train_transformed,
            "X_test_transformed": X_test_transformed,
        }

        # Salva os dados transformados para carregamento no modelo
        self._save_splits(splitted_transforms)

        # Salva pipeline de processamento para que este seja aplicável
        # aos dados de predição utilizando transform.
        joblib.dump(
            pipeline, f"{self.config.transformed_data_path}/pipeline.joblib"
        )
