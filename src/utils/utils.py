import mlflow
import yaml


def setup_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("Hate_Speech_Classification Pipline")



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