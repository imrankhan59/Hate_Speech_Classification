import os
import sys
import pandas as pd
import pickle
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

from src.constant import *
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split 

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.entity.config_entity import ModelTrainerConfig, DataTransformationConfig, DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.ml.model import ModelArchitecture
from src.utils.utils import setup_mlflow, read_params

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.params = read_params()

    def spliting_data(self, csv_path):
        try:
            logging.info("Entered the Splitting_Data function")
            logging.info("Reading the data")

            df = pd.read_csv(csv_path, index_col = False)

            logging.info(f"spliting data into X and Y  \n {df.head(3)}")
            logging.info(f"Dataframe column are {df.columns}")
            logging.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

            x = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train_test_split on the data")
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = self.params['data']['test_size'], random_state = self.params['data']['random_state'])

            logging.info(":::-------------------------------------------------------------->")

            logging.info(f"shape of X_train {X_train.shape}  X_test {X_test.shape}")
            logging.info(f"shape of Y_train {Y_train.shape} Y_test {Y_test.shape}")
            logging.info("Exit the splitting_data function")
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            raise CustomException(e, sys)
        

    def tokanizing(self, X_train):
        try:
            logging.info("Entered tokenization method applying on the data:")
            tokenizer = Tokenizer(num_words = self.params['model']['max_words'] , oov_token="<OOV>")
            X_train = X_train.fillna("").astype(str)
            tokenizer.fit_on_texts(X_train)
            train_sequences = tokenizer.texts_to_sequences(X_train)

            logging.info(f"Converting text to sequences {train_sequences}")

            train_padded = pad_sequences(train_sequences, maxlen = self.params['model']['max_len'])

            return train_padded, tokenizer
        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name : initiate_model_trainer
        Description : This function initiate a model trainer steps

        Output      : Returns model trainer artifact
        On Failure  : write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the initiate_model_trainer")
            if os.path.exists(self.model_trainer_config.TRAINED_MODEL_PATH):
                logging.info(f"Model already exists at {self.model_trainer_config.TRAINED_MODEL_PATH}. Skipping training.")
                model_trainer_artifacts = ModelTrainerArtifact(
                train_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH    
            )
                return model_trainer_artifacts

        # Proceed with training
            logging.info("Model not found. Starting training process.")

            logging.info("calling splitting_data method")
            X_train, X_test, Y_train, Y_test = self.spliting_data(self.data_transformation_artifact.transformed_file_path)

            logging.info(f"data values :: {X_train.head(1)} \n")
            logging.info(f"shape of the X_train :: {X_train.shape} and Y_train :: {Y_train.shape} \n")
            logging.info(f"shape of X_test :: {X_test.shape} and  Y_test :: {Y_test.shape}")
            
            logging.info("calling tokanizing method")
            logging.info(f"X_train columns are {X_train.name}")
            padded_sequence , tokenizer = self.tokanizing(X_train)

            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok = True)

            model_architectue = ModelArchitecture()
            model = model_architectue.get_model()

            logging.info("Starting MLflow run...")
            setup_mlflow()

            with mlflow.start_run(run_name = "model trainer" , nested=True) as run:
                run_id = run.info.run_id
                logging.info(f"MLflow run_id: {run_id}")

            # Log hyperparameters
                mlflow.log_param("test_size", self.params['data']['test_size'])
                mlflow.log_param("random_state", self.params['data']['random_state'])
                mlflow.log_param("epochs", self.params['model']['epochs'])
                mlflow.log_param("batch_size", self.params['model']['batch_size'])
                mlflow.log_param("max_words", self.params['model']['max_words'])
                mlflow.log_param("max_len", self.params['model']['max_len'])

                logging.info("Entered into model training")

                history = model.fit(padded_sequence, Y_train, epochs = self.params['model']['epochs'], batch_size = self.params['model']['batch_size'], validation_split = self.params['model']['validation_split'], verbose = 1)
                logging.info("Model training finished:")

                mlflow.keras.log_model(model, artifact_path="model")

                last_run_id_file = os.path.join(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, "last_run_id.txt")
                with open(last_run_id_file, "w") as f:
                    f.write(run_id)
                mlflow.log_artifact(last_run_id_file, artifact_path="artifacts")

                for epoch in range(self.params['model']['epochs']):
                    mlflow.log_metric("train_loss", history.history["loss"][epoch])
                    mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch])
                    mlflow.log_metric("val_loss", history.history["val_loss"][epoch])
                    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch])

                with open('tokenizer.pickle', 'wb') as handle:
                    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)

                logging.info("Saving the model:")
                model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

                logging.info("saving X_test and Y_test")
                X_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
                Y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

                X_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            logging.info("Mlflow logging completed")

            model_trainer_artifacts = ModelTrainerArtifact(
                train_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH    
            )

            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_validation_config = DataValidationConfig()
    data_transformation_config = DataTransformationConfig()
    model_trainer_config = ModelTrainerConfig()

    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()

    data_transformation = DataTransformation(data_ingestion_artifact = data_ingestion_artifact, data_validation_artifact = data_validation_artifact, data_transformation_config = data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()

    model_trainer = ModelTrainer(data_transformation_artifact = data_transformation_artifact, model_trainer_config = model_trainer_config)
    model_trainer.initiate_model_trainer()