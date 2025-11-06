import os
import sys
import pandas as pd
from dotenv import load_dotenv

from src.constant import *
from src.logger import logging
from src.exception import CustomException

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.configuration.mongodb_conn import MongoDBFetcher

load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.CONNECTION_URL = os.getenv("MONGO_DB_CONN")
        os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACT_DIR, exist_ok=True)

    def load_data(self):            
        try:
            logging.info("Entering the load_data function of DataIngestion class")
            mongo_fetcher = MongoDBFetcher(
                uri=self.CONNECTION_URL,
                db_name=DB_NAME,
                collection_name=COLLECTION_NAME
            )

            dataframes = mongo_fetcher.fetch_both()
            raw_data = dataframes.get("raw_data")
            imb_data = dataframes.get("imb_data")

            logging.info("Data fetched from MongoDB successfully")
            raw_data.drop(columns=['_id'], inplace=True, axis=1)
            imb_data.drop(columns=['_id'], inplace=True, axis=1)

            raw_data.to_csv(self.data_ingestion_config.RAW_ARTIFACT_DIR, index=False)
            imb_data.to_csv(self.data_ingestion_config.IMBALANCE_ARTIFACT_DIR, index=False)
            


            logging.info(f"imbalance data saved at: {self.data_ingestion_config.IMBALANCE_ARTIFACT_DIR}")
            logging.info(f"raw data saved at: {self.data_ingestion_config.RAW_ARTIFACT_DIR}")

            return self.data_ingestion_config.RAW_ARTIFACT_DIR, self.data_ingestion_config.IMBALANCE_ARTIFACT_DIR
        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            raw_data_path, imbalance_data_path = self.load_data()
            logging.info("Data loaded successfully")
        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    
    data_ingest_config = DataIngestionConfig()
    obj = DataIngestion(data_ingestion_config=data_ingest_config)
    obj.initiate_data_ingestion()