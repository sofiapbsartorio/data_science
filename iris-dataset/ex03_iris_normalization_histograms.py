# 3. Consider the `iris.csv` dataset. Display the histogram for each variable before and after normalization.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Selecting only the numeric columns, as non-numeric columns (e.g., 'species') can't be normalized or plotted in histograms
data = pd.read_csv("data/iris-with-errors.csv")
data = data.replace("?", np.nan) 

feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = data[feature_cols].astype(float)
X = X.fillna(X.mean())

# Normalizing the data
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

fig, axes = plt.subplots(2, 4, figsize=(16,8))


for i, col in enumerate(feature_cols):
    # Before
    sns.histplot(X[col], kde=True, ax=axes[0, i], color='skyblue')
    axes[0, i].set_title(f'Before: {col}')

    # After
    sns.histplot(X_scaled[col], kde=True, ax=axes[1, i], color='salmon')
    axes[1, i].set_title(f'After: {col}')

plt.tight_layout()
plt.show()