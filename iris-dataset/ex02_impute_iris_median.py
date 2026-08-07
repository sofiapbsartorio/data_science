# 2. Read the `iris-with-errors.csv` dataset and replace missing values with the **median** of each attribute.
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)

data = pd.read_csv("data/iris-with-errors.csv", header=0, na_values="?") #na_values converts "?" to NaN

data_with_median = data.fillna(data.median(numeric_only=True))

print(data_with_median.head(25))