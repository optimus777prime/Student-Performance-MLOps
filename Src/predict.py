from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

model = joblib.load(MODELS_DIR / "student_model.pkl")
encoders = joblib.load(MODELS_DIR / "encoders.pkl")

student = {
    "gender": "female",
    "race/ethnicity": "group B",
    "parental level of education": "bachelor's degree",
    "lunch": "standard",
    "test preparation course": "completed",
    "math score": 70,
    "reading score": 75,
    "writing score": 80,
}

student_df = pd.DataFrame([student])

for column, encoder in encoders.items():
    if column != "result":
        student_df[column] = encoder.transform(student_df[column])

prediction = model.predict(student_df)
result = encoders["result"].inverse_transform(prediction)[0]

print(f"Prediction: {result}")
