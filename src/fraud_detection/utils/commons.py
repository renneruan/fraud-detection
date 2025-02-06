"""
Módulo de funções a serem utilizadas de forma comum entre os módulos.

Possui funções de leituras e criação de arquivos e diretórios.

Funções
-------
- read_yaml: Lê um arquivo yaml e retorna um objeto ConfigBox.
- create_directories: Cria diretórios de acordo com o caminhos passado.
- save_json: Salva um arquivo json no caminho especificado.
"""

import os
import json
from pathlib import Path

import yaml
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError

from fraud_detection import logger


# Ensure annotations garante que o parâmetro passado é do tipo esperado
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Lê um arquivo yaml e retorna um objeto ConfigBox com
    as informações lidas.

    Será utilizado para ler arquivos de configuração em formato yaml.

    Utilizar o ConfigBox é útil para transformarmos o arquivo yaml
    para um estrutura que possa ser repassada para os módulos Python.

    Args:
        path_to_yaml (str): Caminho para o arquivo.

    Raises:
        ValueError: Se o arquivo estiver vazio
        Exception: Qualquer outra exceção

    Returns:
        ConfigBox: Informações do arquivo em formato ConfigBox
    """
    try:
        with open(path_to_yaml, encoding="UTF-8") as yaml_file:
            # Safe load é utilizado para evitar execução de código malicioso
            content = yaml.safe_load(yaml_file)
            logger.info(
                "Arquivo yaml: %s carregado com sucesso.", path_to_yaml
            )
            return ConfigBox(content)
    except BoxValueError as exc:
        raise ValueError("Arquivo yaml está vazio.") from exc
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Cria diretórios de acordo com a lista passada.

    Args:
        path_to_directories (list): Lista de caminhos para os diretórios.
        verbose (bool, optional): Booleano para decidir imprimir logs ou não.
          Valor padrão: True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info("Diretório criado: %s", path)


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Salva um arquivo json no caminho especificado.
    Dados de dicionário passados serão armazenados no JSON.

    Args:
        path (str): Caminho para salvar o arquivo JSON.
        data (dict): Dados a serem salvos no arquivo JSON.
    """

    with open(path, "w", encoding="UTF-8") as f:
        json.dump(data, f, indent=4)

    logger.info("Dados salvos em arquivo JSON: %s", path)
