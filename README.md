# Student Performance Predictor

A Python and Streamlit project that estimates whether a student is likely to pass or fail using academic and demographic information.

## Features

- Interactive Streamlit web application
- Student-details form and subject-score visualization
- Random Forest classification model
- Saved model and encoder artifacts
- Exploratory data analysis and dataset validation scripts
- Reproducible training settings

## Tech Stack

- Python
- Pandas
- scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
Student-Performance-MLOps/
├── Data/
│   └── students.csv
├── Src/
│   ├── eda.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── test.py
│   └── train_model.py
├── app/
│   └── app.py
├── models/
│   ├── encoders.pkl
│   └── student_model.pkl
├── .gitignore
├── requirements.txt
└── README.md
```

## Run Locally

```bash
git clone https://github.com/optimus777prime/Student-Performance-MLOps.git
cd Student-Performance-MLOps
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Train or refresh the saved model artifacts:

```bash
python Src/train_model.py
```

Run the web application:

```bash
streamlit run app/app.py
```

The app will be available at `http://localhost:8501`.

## Useful Scripts

```bash
python Src/eda.py           # Dataset overview
python Src/test.py          # Missing-value and duplicate-row checks
python Src/preprocessing.py # Preview calculated result labels
python Src/predict.py       # Example command-line prediction
```

## Model Notes and Limitation

The current label is defined as **Pass** when the average of math, reading, and writing scores is at least 50. Because these same scores are model inputs, the model is learning a rule derived from its inputs. This is suitable as a learning demonstration, but it is not an early-warning model.

A stronger next version would predict final performance using only information known before the final exam, such as attendance, prior assessments, assignments, and study habits.

## MLOps Workflow

```text
Data collection → validation and EDA → preprocessing → training
→ evaluation → model serialization → Streamlit prediction app
```

## Author

Darshan Hadagali
