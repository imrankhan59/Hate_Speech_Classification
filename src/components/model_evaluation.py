import os
import sys 
import json
import pickle
import numpy as np
import pandas as pd
import keras
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelEvaluationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelEvaluationArtifact, ModelTrainerArtifact

from src.constant import *
from src.ml import model
from src.utils.utils import setup_mlflow, read_params

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from sklearn.metrics import confusion_matrix
from keras.utils import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifact):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.params = read_params()

        
    def evaluation(self, model, tokenizer):
        try:
            logging.info("Entering the evaluate method of ModelEvaluation class")
            print(self.model_trainer_artifacts.x_test_path)

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path)

            x_test = x_test[TWEET]
            y_test = y_test[LABEL]

            x_test = x_test.fillna("").astype(str)

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_padded = pad_sequences(test_sequences, maxlen = self.params['model']['max_len'])

            loss, accuracy = model.evaluate(test_sequences_padded, y_test)

            logging.info(f"the test accuracy is {accuracy} and Loss {loss}")

            lstm_prediction = model.predict(test_sequences_padded)

            res = []

            for pred in lstm_prediction:
                if pred[0] < 0.5:
                    res.append(0)
                else:
                    res.append(1)

            conf_matrix = confusion_matrix(y_test, res)
            logging.info(f"confusion matrix is {confusion_matrix(y_test, res)}\n")

            precision = precision_score(y_test, res)
            recall = recall_score(y_test, res)
            f1 = f1_score(y_test, res)
            auc = roc_auc_score(y_test, lstm_prediction)

            logging.info(f"Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc} \n")
            logging.info(f"\nClassification Report:\n{classification_report(y_test, res)}")

            return {
            "accuracy": accuracy,
            "loss": loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
            }

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_model_evaluation(self):
        try:
            logging.info("Entering initiate_Model_Evaluation method of ModelEvaluation Class")
            logging.info("Loading current trained Model")
            logging.info("Starting MLflow run...")
            setup_mlflow()

            with mlflow.start_run(run_name = "model evauation" , nested=True) as run:

                #trained_model = keras.models.load_model(self.model_trainer_artifacts.train_model_path)
                os.makedirs(self.model_evaluation_config.MODEL_EVALUATION_DIR, exist_ok=True)

                with open("artifacts/ModelTrainerArtifacts/last_run_id.txt", "r") as f:
                        run_id = f.read().strip()

                logging.info(f"Loading model from mlflow model_uri {run_id}")

                logged_model_uri = f"runs:/{run_id}/model"
                trained_model = mlflow.keras.load_model(logged_model_uri)

                with open('tokenizer.pickle', 'rb') as f:
                    tokenizer = pickle.load(f)

                trained_metrices = self.evaluation(trained_model, tokenizer)
                with open(model_evaluation_config.METRICS_FILE_PATH, "w") as f:
                    json.dump(trained_metrices, f, indent=4)

                mlflow.log_metrics(trained_metrices)
                logging.info(f"Trained Model accuracy {trained_metrices}")

                logging.info("Registering trained model to MLflow Model Registry")
                model_details = mlflow.register_model(logged_model_uri, "LSTM")

                client = MlflowClient()

                try:
                    prod_model_uri = f"models:/{PROD_MODEL_NAME}@production"
                    prod_model = mlflow.keras.load_model(prod_model_uri)
                    logging.info("Found existing Production model.")
                except Exception:
                    logging.info("No Production model registered yet.")
                    prod_model = None

                # Case 1: No production model yet
                if prod_model is None:
                    logging.info("Promoting trained model to Production since no Production model exists.")
                    client.set_registered_model_alias(
                        name=NEW_MODEL_NAME,
                        alias="production",
                        version=model_details.version
                )

                    is_model_accepted = True

            # ---------------- Case 2: Production model exists ----------------
                else:
                    prod_metrics = self.evaluation(prod_model, tokenizer)
                    logging.info(f"Production Model metrics: {prod_metrics}")

                    if trained_metrices["accuracy"] > prod_metrics["accuracy"]:
                        logging.info("Trained model outperforms Production. Promoting to Production.")
                        client.set_registered_model_alias(
                            name=NEW_MODEL_NAME,
                            alias="production",
                            version=model_details.version
                        )
                        is_model_accepted = True
 
                    else:
                        logging.info("Production model is better. Moving trained model to Staging.")
                        client.set_registered_model_alias(
                            name=NEW_MODEL_NAME,
                            alias = "stagging",
                            version=model_details.version
                        )
                        is_model_accepted = False   


                        client.transition_model_version_stage(
                        name=NEW_MODEL_NAME,
                        version=model_details.version,
                        stage="Archived"
                        )  
                        logging.info(f"Archived model version {model_details.version} for {NEW_MODEL_NAME}.")                                

                model_evaluation_artifacts = ModelEvaluationArtifact(is_model_accepted = is_model_accepted)
                return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)    
        

if __name__ == "__main__":

    data_ingestion_config = DataIngestionConfig()
    data_validation_config = DataValidationConfig()
    data_transformation_config = DataTransformationConfig()
    model_trainer_config = ModelTrainerConfig()
    model_evaluation_config = ModelEvaluationConfig()

    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()

    data_transformation = DataTransformation(data_ingestion_artifact = data_ingestion_artifact, data_validation_artifact = data_validation_artifact, data_transformation_config = data_transformation_config)
    data_transformation_artifact = data_transformation.initiate_data_transformation()

    model_trainer = ModelTrainer(data_transformation_artifact = data_transformation_artifact, model_trainer_config = model_trainer_config)
    model_trainer_artifact = model_trainer.initiate_model_trainer()

    model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config, model_trainer_artifacts=model_trainer_artifact)
    model_evaluation_artifact = model_evaluation.initiate_model_evaluation()