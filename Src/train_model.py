from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "students.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PASS_MARK = 50
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["result"] = df["average_score"].ge(PASS_MARK).map({True: "Pass", False: "Fail"})

encoders = {}
categorical_columns = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "result",
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

X = df.drop(columns=["result", "average_score"])
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODELS_DIR / "student_model.pkl")
joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
print(f"Saved model artifacts to {MODELS_DIR}")
