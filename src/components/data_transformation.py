import os
import re
import yaml
import sys
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

from src.entity.config_entity import DataTransformationConfig, DataValidationConfig
from src.entity.artifact_entity import  DataValidationArtifact, DataTransformationArtifact
from src.components.data_validation import DataValidation


stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
stopword = set(stopwords.words('english'))

class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    def imbalance_data_cleaning(self, imbalance_data_file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"reading imbalance data from file paths {imbalance_data_file_path}")
            imbalance_data = pd.read_csv(imbalance_data_file_path)
            logging.info("data read successfully")

            imbalance_data.drop(columns=self.data_transformation_config.ID, axis=self.data_transformation_config.AXIS, 
                                                 inplace=self.data_transformation_config.INPLACE)
            
            #logging.info(f"dropped ID column from imbalance data : data shape {imbalance_data.columns}")
            logging.info(f"exited the imbalance data cleaning method with data shape {imbalance_data.columns}")
            return imbalance_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def raw_data_cleaning(self, raw_data_file_path: str) -> pd.DataFrame:
        try:
            logging.info("reading raw data from file paths")
            raw_data = pd.read_csv(raw_data_file_path)
            logging.info(f"raw data read successfully columns are {raw_data.columns}")
            raw_data.drop(self.data_transformation_config.DROP_COLUMN, 
                          axis = self.data_transformation_config.AXIS, inplace = self.data_transformation_config.INPLACE)
            
            logging.info(f"dropped columns {self.data_transformation_config.DROP_COLUMN} from raw data : data shape {raw_data.columns}")
            logging.info(f"removed null values from raw data : data shape {raw_data.columns}") 

            #raw_data[self.data_transformation_config.CLASS].replace({0:1}, inplace=True)
            #raw_data[self.data_transformation_config.CLASS].replace({2:0}, inplace=True)

            raw_data[self.data_transformation_config.CLASS] = raw_data[self.data_transformation_config.CLASS].replace({0:1})
            raw_data[self.data_transformation_config.CLASS] = raw_data[self.data_transformation_config.CLASS].replace({2:0})

            logging.info(f"raw_data classes {raw_data[self.data_transformation_config.CLASS].unique()}")

            raw_data.rename(columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL}, inplace = True)

            logging.info(f"raw_data columns are now changes to {raw_data.columns}")

            return raw_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def text_cleaning(self, words: str) -> str:
        try:
            if pd.isna(words):
                return ""
            words = str(words).lower()
            words = re.sub('', '', words)
            words = re.sub(r'\d+', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [w for w in words.split(' ') if w not in stopword]
            words = " ".join(words)
            words = [stemmer.stem(w) for w in words.split(' ')]
            words = " ".join(words)
            return words
        except Exception as e:
                raise CustomException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("initiate_data_transformation method started")
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACT_DIR, exist_ok=True)
            
            data_validation_report_file = self.data_validation_artifact.data_validation_report_file_path
            with open(data_validation_report_file, 'r') as file:
                report = yaml.safe_load(file)
            if report['STATUS'] != True:
                logging.info("Data validation failed. Cannot proceed with data transformation.")
                raise Exception("Data validation failed. Cannot proceed with data transformation.")
            logging.info(f"Data validation status {report['STATUS']}")

            logging.info("Data validation passed. Proceeding with data transformation.")
            imbalance_data_file_path = "data/imb_data.csv"
            raw_data_file_path = "data/raw_data.csv"

            imbalance_data = self.imbalance_data_cleaning(imbalance_data_file_path)
            raw_data = self.raw_data_cleaning(raw_data_file_path)

            logging.info("______________________________________________________________")

            logging.info(f"raw_data columns are {raw_data.columns}")

            logging.info("cleaned imbalance and raw data successfully")

            df = pd.concat([imbalance_data, raw_data])

            logging.info(f"new dataframe columns are \n {df.columns}")

            df['tweet']=df['tweet'].fillna('')

            logging.info(f"DataFrame : {df.head(1)}")
            
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.text_cleaning)

            logging.info(f"df columns after applying text_cleaning method {df.columns} ")
            logging.info(f"df look like \n {df.head}")

            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index = False)

            logging.info(f"transformed data saved at {self.data_transformation_config.TRANSFORMED_FILE_PATH}")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_file_path = self.data_transformation_config.TRANSFORMED_FILE_PATH,
            )
            logging.info(f"concatenated imbalance and raw data : data shape {df.shape}")
            return data_transformation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    data_validation_config = DataValidationConfig()
    data_transformation_config = DataTransformationConfig()

    data_validation = DataValidation(data_validation_config=data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()

    data_transformation = DataTransformation(data_validation_artifact = data_validation_artifact, data_transformation_config = data_transformation_config)
    data_transformation.initiate_data_transformation()