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

from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataValidationConfig, DataTransformationConfig, ModelEvaluationConfig, ModelTrainerConfig
from src.entity.artifact_entity import  ModelEvaluationArtifact, ModelTrainerArtifact

from src.constant import *
from src.utils.utils import setup_mlflow, read_params

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
            logging.info("Entered the evaluation method of ModelEvaluation class")
            print(self.model_trainer_artifacts.x_test_path)

            logging.info("Reading test data")
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path)

            x_test = x_test[TWEET]
            y_test = y_test[LABEL]

            x_test = x_test.fillna("").astype(str)

            logging.info("Tokenizing and padding test data")
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_padded = pad_sequences(test_sequences, maxlen = self.params['model']['max_len'])

            logging.info("Evaluating the model on test data")
            loss, accuracy = model.evaluate(test_sequences_padded, y_test)
            logging.info(f"test data accuracy is {accuracy} and Loss {loss}")

            logging.info("Generating predictions on test data")
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
            logging.info(f"Precision: {precision}")

            recall = recall_score(y_test, res)
            logging.info(f"Recall: {recall}")

            f1 = f1_score(y_test, res)
            logging.info(f"F1 Score: {f1}")

            auc = roc_auc_score(y_test, lstm_prediction)
            logging.info(f"AUC: {auc}")


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
            logging.info("*"*70)
            logging.info("$$$$$$$$$$$$ MODEL EVALUATION STAGE $$$$$$$$$$$$")
            logging.info("Entered initiate_Model_Evaluation method of ModelEvaluation Class")
            logging.info("Starting MLflow run...")

            setup_mlflow()
            with mlflow.start_run(run_name="model_evaluation", nested=True) as run:

                os.makedirs(self.model_evaluation_config.MODEL_EVALUATION_DIR, exist_ok=True)

                logging.info("Reading run_id of trained model")
                with open("artifacts/ModelTrainerArtifacts/last_run_id.txt", "r") as f:
                    run_id = f.read().strip()

                logging.info(f"Loading model from mlflow model_uri {run_id}")
                logged_model_uri = f"runs:/{run_id}/model"
                trained_model = mlflow.keras.load_model(logged_model_uri)

                logging.info("Loading tokenizer")
                with open('tokenizer.pickle', 'rb') as f:
                    tokenizer = pickle.load(f)

                logging.info("Evaluating trained model")
                trained_metrics = self.evaluation(trained_model, tokenizer)

                logging.info("Saving trained model metrics")
                with open(self.model_evaluation_config.METRICS_FILE_PATH, "w") as f:
                    json.dump(trained_metrics, f, indent=4)

                mlflow.log_metrics(trained_metrics)
                logging.info(f"Trained Model metrics: {trained_metrics}")

                # Register model
                logging.info("Registering trained model to MLflow Model Registry")
                model_details = mlflow.register_model(logged_model_uri, "LSTM")
                client = MlflowClient()

                try:
                    prod_model_uri = "models:/LSTM/Production"
                    prod_model = mlflow.keras.load_model(prod_model_uri)
                    logging.info("Found existing Production model.")
                    prod_metrics = self.evaluation(prod_model, tokenizer)
                    logging.info(f"Production Model metrics: {prod_metrics}")
                except Exception:
                    logging.info("No Production model exists yet.")
                    prod_model = None
                    prod_metrics = None

                is_model_accepted = False

                if (prod_model is None) or (trained_metrics["accuracy"] > prod_metrics.get("accuracy", 0)):
                    logging.info("Trained model is better or no Production model exists. Registering and promoting to Production.")
                    # Register model
                    model_details = mlflow.register_model(logged_model_uri, "LSTM")

                    client.transition_model_version_stage(
                        name="LSTM",
                        version=model_details.version,
                        stage="Production",
                        archive_existing_versions=True
                    )
                    is_model_accepted = True
                else:
                    logging.info("Production model is better. Registering trained model to Staging.")
                    model_details = mlflow.register_model(logged_model_uri, "LSTM")
                    client.transition_model_version_stage(
                        name="LSTM",
                        version=model_details.version,
                        stage="Staging"
                    )

                logging.info(f"Model accepted for Production: {is_model_accepted}")
                return is_model_accepted, trained_metrics

        except Exception as e:
            raise CustomException(e, sys)
    
        
if __name__ == "__main__":
    model_evaluation_config = ModelEvaluationConfig()
    model_trainer_artifact = ModelTrainerArtifact(
        train_model_path = "artifacts/ModelTrainerArtifacts/trained_model.h5",
        x_test_path = "artifacts/ModelTrainerArtifacts/x_test.csv",
        y_test_path = "artifacts/ModelTrainerArtifacts/y_test.csv"    
    )

    model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config, model_trainer_artifacts=model_trainer_artifact)
    model_evaluation_artifact = model_evaluation.initiate_model_evaluation()