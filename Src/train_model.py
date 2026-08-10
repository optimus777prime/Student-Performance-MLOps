import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib

# Load dataset
df = pd.read_csv("Data/students.csv")

# Create average score
df["average_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

# Create Pass/Fail target
df["result"] = df["average_score"].apply(
    lambda x: "Pass" if x >= 50 else "Fail"
)

# Convert text columns into numbers
encoder = LabelEncoder()

encoders = {}

for column in [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "result"
]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Features and Target
X = df.drop(["result", "average_score"], axis=1)
y = df["result"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100)
print('Confusion matrix',confusion_matrix(y_test,y_pred))
print('Classification report',classification_report(y_test,y_pred))

joblib.dump(model,'models/student_model.pkl')
joblib.dump(encoders,'models/encoders.pkl')
print('Encoders saved successfully')
print('model saved successfully')