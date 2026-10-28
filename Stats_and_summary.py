import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('debutanizer_data.csv')

print("Dataset shape:" ,df.shape)
print("Dataset columns:", df.columns)
print("Missing values:")
print(df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())
print("Summary statistics:")
print(df.describe())


correlation = df.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Matrix")
plt.show()