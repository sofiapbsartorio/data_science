# 1. Leia a base `iris-with-errors.csv`, faça a limpeza dos dados e remova as duas últimas colunas.
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)

data = pd.read_csv("data/iris-with-errors.csv", header=0)
print("------------- Step 1 ------------- ")
print("Number of rows and columns:", data.shape)
print(data.head(25))

# Inspecionando os dados
print("------------- Step 2 ------------- ")
print("Dataset columns:")
print(data.columns)
print("\nData types:")
print(data.dtypes)
print("\nNumber of missing values per column:")
print(data.isna().sum())
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Substituindo '?' com valores NaN
print("------------- Step 3 ------------- ")
data = data.replace("?", np.nan) 
print("Missing values per column after replacing '?':")
print(data.isna().sum())

# Removendo linhas com valores NaN e linhas duplicadas
print("------------- Step 4 ------------- ")
clean_data = data.dropna().drop_duplicates()
print("Original shape:", data.shape)
print("Shape after cleaning:", clean_data.shape)
print(clean_data.head(25))

# Removendo as últimas duas colunas do dataset limpo
print("------------- Step 5 ------------- ")
example_drop = clean_data.copy()
print("Current features:", list(example_drop.columns))
print("Columns to be removed", list(example_drop.columns[[-1,-2]]))

example_drop = example_drop.drop(example_drop.columns[[-1,-2]], axis=1)
print(example_drop.head(25))