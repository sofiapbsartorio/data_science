# 4. Considere a base `iris.csv`. Mostre o histograma de cada variável antes e depois da padronização.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Carregando os dados e lidando com os valores faltantes/inválidos
data = pd.read_csv("data/iris-with-errors.csv")
data = data.replace("?", np.nan)

# Selecionando as colunas numéricas e limpando os dados
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = data[feature_cols].astype(float)
X = X.fillna(X.mean())

# Padronizando os dados 
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

fig, axes = plt.subplots(2, 4, figsize=(16,8))

for i, col in enumerate(feature_cols):
    # Antes
    sns.histplot(X[col], kde=True, ax=axes[0, i], color='skyblue')
    axes[0, i].set_title(f'Before: {col}')

    # Depois
    sns.histplot(X_scaled[col], kde=True, ax=axes[1, i], color='purple')
    axes[1, i].set_title(f'After: {col}')

plt.tight_layout()
plt.show()