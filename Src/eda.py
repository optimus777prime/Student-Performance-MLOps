import pandas as pd

# Load the dataset
df = pd.read_csv("Data/students.csv.csv")

print("========== DATASET INFO ==========")
print(df.info())

print("\n========== STATISTICS ==========")
print(df.describe())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== UNIQUE VALUES ==========")
print(df.nunique())