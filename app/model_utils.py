from joblib import load
import pandas as pd

model = load("app/model.joblib")
label_encoder = load("app/label_encoder.joblib")

columns = [
    "age", "job", "marital", "education", "default", "balance", 
    "housing", "loan", "contact", "duration", "campaign", 
    "pdays", "previous", "poutcome"
]

def predict(features):
    df = pd.DataFrame([features], columns=columns)
    result = model.predict(df)[0]
    return int(label_encoder.inverse_transform([result])[0] == "yes")  # Return 1 if 'yes', 0 if 'no'
