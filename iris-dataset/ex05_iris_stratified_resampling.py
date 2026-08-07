# 5. Resample the Iris dataset by selecting 10 elements from each class.
import pandas as pd
import numpy as np

data = pd.read_csv("data/iris-with-errors.csv")
data = data.replace("?", pd.NA).dropna(subset=['species'])
data = data.dropna(subset=['species'])

resampled_data = data.groupby('species').sample(n=10, random_state=42)

print("Sample count per class: ")
print(resampled_data['species'].value_counts())

print(f"\nTotal resampled rows: {len(resampled_data)}")