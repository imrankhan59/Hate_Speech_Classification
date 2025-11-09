import logging
import mlflow
import yaml
import os
import pandas as pd
import dagshub
from dotenv import load_dotenv
import re
import string
from nltk.corpus import stopwords
import nltk

stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
stopword = set(stopwords.words('english'))

load_dotenv()

def setup_mlflow():
    # Use DAGsHub URI from environment if set, otherwise fallback to local mlruns
    dagshub.auth.add_app_token(os.getenv("DAGS_HUB_TOKEN"))
    mlflow.set_tracking_uri("https://dagshub.com/imrankhan59/Hate_Speech_Classification.mlflow")
    dagshub.init(repo_owner='imrankhan59', repo_name='Hate_Speech_Classification', mlflow=True)
    
    # Set your experiment
    mlflow.set_experiment("LSTM_Hates_Speech_Classification")
    logging.info("MLflow setup complete with DAGsHub tracking URI.")

        # ✅ Debug log for GitHub Actions
    print("✅ MLflow tracking URI:", mlflow.get_tracking_uri())
    print("✅ MLflow experiment:", mlflow.get_experiment_by_name("LSTM_Hates_Speech_Classification"))



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


def clean_text(words: str) -> str:
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
            logging.error(f"Error in cleaning text: {e}")


if __name__ == "__main__":
    pass