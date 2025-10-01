import os
import sys
import yaml
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataValidationConfig, DataIngestionConfig
from src.entity.artifact_entity import  DataValidationArtifact

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        self.data_validation_config = data_validation_config

    def validate_column_names(self, df: pd.DataFrame, schema: dict, dataset_name) -> bool:
        expected_columns = schema['schemas'][dataset_name]['columns'].keys()
        actual_columns = df.columns.tolist()
        if set(expected_columns) != set(actual_columns):
            logging.info(f"Column names mismatch in {dataset_name}. Expected: {expected_columns}, Actual: {actual_columns}")
            return False
        logging.info(f"Column names match for {dataset_name}.")
        return True

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("starting data validation")

            imbalance_data_df = pd.read_csv("data/imb_data.csv")
            raw_data_df = pd.read_csv("data/raw_data.csv")

            logging.info("data loaded successfully")

            os.makedirs(self.data_validation_config.DATA_VALIDATION_ARTIFACT_DIR, exist_ok=True)

            with open(self.data_validation_config.DATA_VALIDATION_SCHEMA_FILE_NAME, 'r') as file:
                schema = yaml.safe_load(file)

            logging.info("schema file loaded successfully")

            imbalance_valid = self.validate_column_names(imbalance_data_df, schema, 'imbalance_data')
            raw_valid = self.validate_column_names(raw_data_df, schema, 'raw_data')

            logging.info(f"Imbalance data validation status: {imbalance_valid}")
            logging.info(f"Raw data validation status: {raw_valid}")

            if not (imbalance_valid and raw_valid):
                with open(self.data_validation_config.DATA_VALIDATION_REPORT_FILE_PATH, 'w') as report_file:
                    report = {'STATUS': False}
                    yaml.dump(report, report_file)
            else:
                with open(self.data_validation_config.DATA_VALIDATION_REPORT_FILE_PATH, 'w') as report_file:
                    report = {'STATUS': True}
                    yaml.dump(report, report_file)

            data_validation_artifact = DataValidationArtifact(
                data_validation_report_file_path=self.data_validation_config.DATA_VALIDATION_REPORT_FILE_PATH
            )
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_validation_config = DataValidationConfig()

    data_validation = DataValidation(data_validation_config=data_validation_config)
    data_validation.initiate_data_validation()