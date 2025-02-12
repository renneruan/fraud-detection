"""
Módulo para armazenar o formato das configurações recebidas.

Serve para relacionar o que é lido no arquivo yaml para o formato a
ser traduzido em python.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Armazena o modelo de configuração para validação dos dados.

    Args:
        root_path (Path): O diretório que será salvo o arquivo com resultado\
                          da validação.
        raw_data_path (Path): Diretório contendo os dados iniciais brutos.
        status_file (str): Arquivo que será salvo contendo o resultado.
        all_schema (dict): Colunas que devem estar nos dados de entrada.
    """

    root_path: Path
    raw_data_path: Path
    status_file: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Armazena o modelo de configuração para transformação dos dados.

    Args:
        raw_data_path (Path): Diretório contendo os dados iniciais brutos.
        transformed_data_path (Path): Diretório que os dados transformados\
                                      serão salvos.
        target_column (str): Coluna alvo para ser usada na transformação.
    """

    raw_data_path: Path
    transformed_data_path: Path
    target_column: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Armazena o modelo de configuração para o treinamento do modelo.
    Irá carregar os hiperparâmetros inerentes ao modelo.

    Args:
        model_target_path (Path): Caminho para salvar o modelo treinado.
        train_x_data_path (Path): Caminho para os dados de treino (features).
        test_x_data_path (Path): Caminho para os dados de teste (features).
        train_y_data_path (Path): Caminho para os rótulos de treino.
        test_y_data_path (Path): Caminho para os rótulos de teste.
        model_name (str): Nome do modelo.

        Hiperparâmetros do LightGBM (consulte a documentação do scikit-learn):
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

        subsample (float): Fração de amostras usadas por árvore.
        reg_alpha (float): Regularização L1.
        reg_lambda (float): Regularização L2.
        num_leaves (int): Número máximo de folhas por árvore.
        n_estimators (int): Número de árvores no modelo.
        max_depth (int): Profundidade máxima das árvores.
        learning_rate (float): Taxa de aprendizado.
        colsample_bytree (float): Fração de colunas usadas por árvore.
        scale_pos_weight (float): Proporção para lidar com desbalanceamento\
                                  de classe
        target_column (str): Nome da coluna alvo para predição.
    """

    model_target_path: Path
    train_x_data_path: Path
    test_x_data_path: Path
    train_y_data_path: Path
    test_y_data_path: Path
    model_name: str
    subsample: float
    reg_lambda: float
    reg_alpha: float
    num_leaves: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    colsample_bytree: float
    scale_pos_weight: float
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Armazena o padrão de configurações para a etapa de avaliação do modelo.

    Args:
        model_results_path (Path): Caminho onde os resultados da avaliação
         serão armazenados.
        test_x_data_path (Path): Caminho para os dados de teste (features).
        test_y_data_path (Path): Caminho para os rótulos de teste.
        model_path (Path): Caminho para o modelo treinado a ser avaliado.
        all_params (dict): Dicionário contendo os hiperparâmetros usados no \
                           modelo.
        metric_file_name (Path): Nome do arquivo onde as métricas de \
                                 avaliação serão salvas.
        target_column (str): Nome da coluna alvo para predição.
        mlflow_uri (str): URI do MLflow para rastreamento dos experimentos.
    """

    model_results_path: Path
    test_x_data_path: Path
    test_y_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
