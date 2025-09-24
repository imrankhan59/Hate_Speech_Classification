# Common Constants
ARTIFACT_DIR = "artifacts"
LABEL = "label"  # column name
TWEET = "tweet"  # column name

# Data Ingestion Constants
DATA_INGESTION_ARTIFACTS = "data"
DB_NAME = "NLP"
COLLECTION_NAME = "NLP_DATA"
IMB_DATA_FILENAME = "imb_data.csv"
RAW_DATA_FILENAME = "raw_data.csv"


# Data Validation Constants
DATA_VALIDATION_ARTIFACTS = "DataValidationArtifacts"
DATA_VALIDATION_REPORT_FILE_NAME = "data_validation_report.yaml"
DATA_VALIDATION_SCHEMA_FILE_NAME = "schema.yaml"

# Data Transformation Constants
DATA_TRANSFORMATION_ARTIFACTS = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME = "transformed_data.csv"
DATA_DIR = "data"
ID = "id"
AXIS = 1
INPLACE = True
DROP_COLUMN = ['count', 'hate_speech', 'offensive_language', 'neither']
CLASS = "class"

# Model Trainer

MODEL_TRAINER_ARTIFACTS = "ModelTrainerArtifacts"
TRAINED_MODEL_DIR = "trained_model"
TRAINED_MODEL_NAME = "model.h5"
X_TEST_FILE_NAME = "x_test.csv"
Y_TEST_FILE_NAME = "y_test.csv"
X_TRAIN_FILE_NAME = "x_train.csv"





MODEL_EVALUATION_ARTIFACT_DIR = "ModelEvaluationArtifacts"
METRICS = "metrics.json"

PROD_MODEL_NAME = "LSTM"
NEW_MODEL_NAME = "LSTM"