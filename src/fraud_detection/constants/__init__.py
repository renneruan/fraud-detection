"""
Módulo para armazenamento de constantes dos caminhos de configuração
"""

from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("config/schema.yaml")

PIPELINE_PATH = Path("artifacts/data_transformation/pipeline.pkl")
MODEL_PATH = Path("artifacts/model_output/model.joblib")
