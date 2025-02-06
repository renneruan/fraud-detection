"""
Módulo para armazenamento de constantes dos caminhos de configuração

Armazena valores de:

- CONFIG_FILE_PATH: Caminho do arquivo de configuração.
- PARAMS_FILE_PATH: Caminho para arquivo de parâmetros do modelo.
- SCHEMA_FILE_PATH: Caminho para schema dos dados de entrada.
- PIPELINE_PATH: Caminho para arquivo do pipeline de pré-processamento.
- MODEL_PATH: Caminho para arquivo do modelo treinado.
"""

from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("config/schema.yaml")

PIPELINE_PATH = Path("artifacts/data_transformation/pipeline.pkl")
MODEL_PATH = Path("artifacts/model_output/model.joblib")
