from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Activation
from tensorflow.keras.optimizers import RMSprop
from src.utils.utils import read_params
from src.constant import *

class ModelArchitecture:
    def __init__(self):
        self.params = read_params()

    def get_model(self):
        model = Sequential()
        model.add(Embedding(self.params["model"]["max_words"], 128))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
        model.add(LSTM(64, dropout = 0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation = 'sigmoid'))
        model.summary()
        model.compile(loss = "binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])

        return model
    
if __name__ == "__main__":
    pass