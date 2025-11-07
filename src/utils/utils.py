import logging
import mlflow
import yaml
import os
import dagshub
from dotenv import load_dotenv

load_dotenv()

def setup_mlflow():
    # Use DAGsHub URI from environment if set, otherwise fallback to local mlruns
    dagshub.auth.add_app_token(os.getenv("DAGS_HUB_TOKEN"))
    mlflow.set_tracking_uri("https://dagshub.com/imrankhan59/Hate_Speech_Classification.mlflow")
    dagshub.init(repo_owner='imrankhan59', repo_name='Hate_Speech_Classification', mlflow=True)
    
    # Set your experiment
    mlflow.set_experiment("LSTM_Hates_Speech_Classification")
    logging.info("MLflow setup complete with DAGsHub tracking URI.")



def read_params(config_path: str = "params.yaml") -> dict:
    """
    Reads parameters from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file (default: 'params.yaml')

    Returns:
        dict: Parsed parameters
    """
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    return params



if __name__ == "__main__":
    pass