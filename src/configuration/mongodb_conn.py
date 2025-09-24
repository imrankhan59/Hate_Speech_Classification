from pymongo import MongoClient
import pandas as pd
from src.logger import logging


class MongoDBFetcher:
    def __init__(self, uri: str, db_name: str, collection_name: str):
        logging.info("Entered the MongoDBFetcher class")
        """
        Initialize MongoDB connection.

        :param uri: MongoDB connection string
        :param db_name: Database name
        :param collection_name: Collection name
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def fetch_dataframe(self, dataset_name: str) -> pd.DataFrame:
        logging.info("Fetching dataframe from MongoDB")
        """
        Fetch a specific dataframe by dataset_name.

        Assumes documents are stored with a field "dataset" identifying them.
        Example document structure:
        {
            "dataset": "raw_data",
            "data": [{col1: val1, col2: val2, ...}, ...]
        }

        :param dataset_name: name of dataset ("raw_data" or "imb_data")
        :return: Pandas DataFrame
        """
        doc = self.collection.find_one({"dataset": dataset_name},{"_id": 0})

        if not doc or "data" not in doc:
            return pd.DataFrame()

        df = pd.DataFrame(doc["data"])
        logging.info(f"Fetched {len(df)} records for dataset: {dataset_name}")
        return df

    def fetch_both(self) -> dict:
        logging.info("connected to MongoDB")
        """
        Fetch both raw_data and imb_data at once.

        :return: Dictionary with both DataFrames
        """
        datasets = {}
        for name in ["raw_data", "imb_data"]:
            datasets[name] = self.fetch_dataframe(name)
        logging.info("Fetched both datasets successfully")
        return datasets

if __name__ == "__main__":
    pass