from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pickle
import re


app = FastAPI()

# Load model
model = tf.keras.models.load_model("stress_model.h5")


with open('tokenizer.json') as f:
    data = json.load(f)

tokenizer = tokenizer_from_json(json.dumps(data))

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

max_len = 50

# Input format
class TextIn(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

@app.post("/predict")
def predict_sentiment(data: TextIn):
    # Preprocess
    text = clean_text(data.text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    # Predict
    pred = model.predict(padded)[0]  # e.g., [0.1, 0.8, 0.1]
    pred_class_index = np.argmax(pred)
    pred_class = le.inverse_transform([pred_class_index])[0]

    return {
        "predicted_class": pred_class,
        "confidence": float(np.max(pred)),
        "all_confidences": pred.tolist()
    }
