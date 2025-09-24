import os
from pathlib import Path
from src.constant import *
from dataclasses import dataclass, field

@dataclass
class DataIngestionConfig:
    DATA_INGESTION_ARTIFACT_DIR: str = os.path.join(os.getcwd(), DATA_INGESTION_ARTIFACTS)
    IMBALANCE_ARTIFACT_DIR: str = os.path.join(DATA_INGESTION_ARTIFACT_DIR, IMB_DATA_FILENAME)
    RAW_ARTIFACT_DIR: str = os.path.join(DATA_INGESTION_ARTIFACT_DIR, RAW_DATA_FILENAME)


@dataclass
class DataValidationConfig:
    DATA_VALIDATION_ARTIFACT_DIR: str = os.path.join(os.getcwd(), ARTIFACT_DIR, DATA_VALIDATION_ARTIFACTS)
    DATA_VALIDATION_REPORT_FILE_PATH: str = os.path.join(DATA_VALIDATION_ARTIFACT_DIR, DATA_VALIDATION_REPORT_FILE_NAME)
    DATA_VALIDATION_SCHEMA_FILE_NAME: str = DATA_VALIDATION_SCHEMA_FILE_NAME

        
@dataclass
class DataTransformationConfig:
    DATA_TRANSFORMATION_ARTIFACT_DIR: str = os.path.join(os.getcwd(), ARTIFACT_DIR, DATA_TRANSFORMATION_ARTIFACTS)
    TRANSFORMED_FILE_PATH: str = os.path.join(DATA_TRANSFORMATION_ARTIFACT_DIR, TRANSFORMED_FILE_NAME)
    ID: str = ID
    AXIS: int = AXIS
    #DROP_COLUMN: list = DROP_COLUMN
    DROP_COLUMN: list = field(default_factory=lambda: DROP_COLUMN)
    INPLACE: bool = INPLACE
    CLASS: str = CLASS
    LABEL: str = LABEL
    TWEET: str = TWEET


@dataclass
class ModelTrainerConfig:
    MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACT_DIR, MODEL_TRAINER_ARTIFACTS)
    TRAINED_MODEL_PATH = os.path.join(MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_NAME)
    X_TEST_DATA_PATH = os.path.join(MODEL_TRAINER_ARTIFACTS_DIR, X_TEST_FILE_NAME)
    Y_TEST_DATA_PATH = os.path.join(MODEL_TRAINER_ARTIFACTS_DIR, Y_TEST_FILE_NAME)
    X_TRAIN_DATA_PATH = os.path.join(MODEL_TRAINER_ARTIFACTS_DIR, X_TRAIN_FILE_NAME)


@dataclass 
class ModelEvaluationConfig:
    MODEL_EVALUATION_DIR= os.path.join(os.getcwd(), ARTIFACT_DIR, MODEL_EVALUATION_ARTIFACT_DIR)
    #BEST_MODEL_PATH_DIR= os.path.join(MODEL_EVALUATION_ARTIFACT_DIR, BEST_MODEL_DIR)
    METRICS_FILE_PATH = os.path.join(MODEL_EVALUATION_DIR, METRICS)


@dataclass
class ModelPusherConfig:
    TRAINED_MODEL_PATH= os.path.join(os.getcwd(), ARTIFACT_DIR, MODEL_TRAINER_ARTIFACTS)
    TRAINED_MODEL_NAME = TRAINED_MODEL_NAME