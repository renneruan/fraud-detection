from fraud_detection.entity.config_entity import DataValidationConfig
import datetime
import pandas as pd


class DataValidation:
    """
    Classe que irá realizar a validação para as colunas dos dados de entrada.

    Args:
    - DataValidationConfig (dataclass): Classe de valores de configuração.
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Função para validar colunas dos dados de entrada, verifica
        se elas seguem o esquema de configuração repassado.

        Returns:
        - bool: Booleano contendo status do sucesso da validação.
        """
        try:
            validation_status = None

            data = pd.read_csv(self.config.raw_data_path)
            all_cols = list(data.columns)

            # Carrega o arquivo de schema e cria o arquivo
            # de status da verificação.
            all_schema = self.config.all_schema.keys()
            with open(self.config.status_file, "w") as f:
                # O arquivo de status pode ser utilizado em orquestradores
                f.write(
                    datetime.datetime.now().strftime(
                        "%d-%b-%Y (%H:%M:%S.%f)\n"
                    )
                )

            # Verifica se as colunas dos dados recebido são aderentes ao schema
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                else:
                    validation_status = True

                with open(self.config.status_file, "a") as f:
                    f.write(f"{col} validation status: {validation_status}\n")

            return validation_status

        except Exception as e:
            raise e
