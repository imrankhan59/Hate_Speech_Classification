import mlflow
import yaml
import os


def setup_mlflow():
    # Use DAGsHub URI from environment if set, otherwise fallback to local mlruns
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set your experiment
    mlflow.set_experiment("LSTM_Hate_Speech_Classification")



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