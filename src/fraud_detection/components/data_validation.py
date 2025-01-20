from fraud_detection.entity.config_entity import DataValidationConfig
import datetime
import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.raw_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()
            with open(self.config.status_file, "w") as f:
                f.write(datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)\n"))

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.status_file, "a") as f:
                        f.write(f"{col} validation status: {validation_status}\n")
                else:
                    validation_status = True
                    with open(self.config.status_file, "a") as f:
                        f.write(f"{col} validation status: {validation_status}\n")

            return validation_status

        except Exception as e:
            raise e
