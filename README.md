# Student Performance Predictor 

A Machine Learning and MLOps project that predicts whether a student is likely to PASS or FAIL based on academic and demographic information.

## Features

- Student performance prediction
- Data preprocessing
- Machine Learning model
- Streamlit web application
- Prediction confidence score
- Score visualization

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Git & GitHub

## Project Structure

Student-Performance-MLOps/
│
├── Data/
│   └── students.csv
│
├── Src/
│   ├── eda.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── test.py
│   └── train_model.py
│
├── app/
│   └── app.py
│
└── models/
    ├── encoders.pkl
    └── student_model.pkl

How to Run
1. Clone the Repository
git clone https://github.com/optimus777prime/Student-Performance-MLOps.git
cd Student-Performance-MLOps
2. Install Required Libraries
pip install pandas scikit-learn streamlit joblib
3. Run the Application
streamlit run app/app.py
4. Open the Application
After running the command, open:
http://localhost:8501
5. Use the Predictor
Enter the student's details and scores, then click Predict.
The application displays:
PASS or FAIL prediction
Prediction confidence
Math, Reading and Writing scores
Score visualization

MLOps Workflow

Data Collection
      ↓
Data Preprocessing
      ↓
EDA & Feature Analysis
      ↓
Model Training
      ↓
Model Testing
      ↓
Model Serialization
      ↓
Streamlit Deployment
      ↓
Student Prediction

Output
The application predicts whether a student is likely to PASS or FAIL and displays the prediction confidence along with subject score visualization.

Author
Darshan Hadagali

### After pasting
Click **Commit changes → Commit changes** ✅

Then your GitHub repository will have a proper **project description + features + technologies + structure + installation + MLOps flow + usage**. 🔥
