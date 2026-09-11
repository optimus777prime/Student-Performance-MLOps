from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "students.csv"
PASS_MARK = 50

df = pd.read_csv(DATA_PATH)

df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["result"] = df["average_score"].ge(PASS_MARK).map({True: "Pass", False: "Fail"})

print(df[["math score", "reading score", "writing score", "average_score", "result"]].head())

print("\nResult Count:")
print(df["result"].value_counts())
