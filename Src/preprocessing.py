import pandas as pd

# Load dataset
df = pd.read_csv("Data/students.csv.csv")

# Calculate average score
df["average_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

# Create Pass/Fail column
df["result"] = df["average_score"].apply(
    lambda x: "Pass" if x >= 50 else "Fail"
)

print(df[["math score",
          "reading score",
          "writing score",
          "average_score",
          "result"]].head())

print("\nResult Count:")
print(df["result"].value_counts())