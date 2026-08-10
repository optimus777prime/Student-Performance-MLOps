import joblib
import pandas as pd

model=joblib.load('models/student_model.pkl')
student = {
    "gender": 1,
    "race/ethnicity": 2,
    "parental level of education": 3,
    "lunch": 1,
    "test preparation course": 1,
    "math score": 70,
    "reading score": 75,
    "writing score": 80
}

# Convert to DataFrame
student_df = pd.DataFrame([student])

# Predict
prediction = model.predict(student_df)

# Display result
if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")