
from pathlib import Path
import pandas as pd

csv_path = Path(
    "data/railway_50km_200nodes_clip_conditions_dataset.csv"
)

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)

print("\nColumn names:")
for column in df.columns:
    print(column)

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isnull().sum())

print("\nPossible target columns:")
for column in df.columns:
    if any(word in column.lower() for word in [
        "label", "condition", "class", "defect"
    ]):
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).to_string())
