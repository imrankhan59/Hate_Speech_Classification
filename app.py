from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Annotated
import pickle
import mlflow
from src.utils.utils import read_params
from src.utils.text_cleaning import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.logger import logging

mlflow.set_tracking_uri("sqlite:///mlruns_db/mlflow.db")

app = FastAPI(title="Sentiment Analysis API", version="1.0")

@app.on_event("startup")
def load_model():
    global tokenizer, prod_model, MAX_LEN
    try:
        with open('tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        model_uri = "models:/LSTM/Production"
        prod_model = mlflow.keras.load_model(model_uri)
        params = read_params()
        MAX_LEN = params['model']['max_len']
        logging.info("Model and tokenizer loaded successfully at startup.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError("Startup failed - model not loaded.")

class TextInput(BaseModel):
    text: Annotated[str, Field(..., example="This is a sample text for prediction.")]

class PredictionResponse(BaseModel):
    cleaned_text: str
    sentiment: str
    raw_score: Annotated[float, Field(..., example=0.85)]


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(data: TextInput):
    try:
        cleaned_text = clean_text(data.text)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(seq, maxlen=MAX_LEN)
        prediction = prod_model.predict(padded_seq)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check logs for details."
        )
    
    predicted_label = int((prediction > 0.5).astype("int32")[0][0])
    sentiment = "Positive" if predicted_label == 1 else "Negative"

    return PredictionResponse(
    cleaned_text=cleaned_text,
    sentiment=sentiment,
    raw_score=float(prediction[0][0])
    )








