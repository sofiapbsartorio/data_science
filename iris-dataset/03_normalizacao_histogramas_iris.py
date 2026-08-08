# 3. Considere a base `iris.csv`. Mostre o histograma de cada variável antes e depois da normalização.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Selecionando somente as colunas numéricas, já que valores não numéricos (como a coluna 'species') não podem ser normalizados e nem plotados em histogramas
data = pd.read_csv("data/iris-with-errors.csv")
data = data.replace("?", np.nan) 

feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = data[feature_cols].astype(float)
X = X.fillna(X.mean())

# Normalizando os dados
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

fig, axes = plt.subplots(2, 4, figsize=(16,8))


for i, col in enumerate(feature_cols):
    # Antes
    sns.histplot(X[col], kde=True, ax=axes[0, i], color='skyblue')
    axes[0, i].set_title(f'Before: {col}')

    # Depois
    sns.histplot(X_scaled[col], kde=True, ax=axes[1, i], color='salmon')
    axes[1, i].set_title(f'After: {col}')

plt.tight_layout()
plt.show()