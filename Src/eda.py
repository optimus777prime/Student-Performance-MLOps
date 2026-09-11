from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "students.csv"

df = pd.read_csv(DATA_PATH)

print("========== DATASET INFO ==========")
df.info()

print("\n========== STATISTICS ==========")
print(df.describe())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== UNIQUE VALUES ==========")
print(df.nunique())
