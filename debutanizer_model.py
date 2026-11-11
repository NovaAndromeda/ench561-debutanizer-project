# Load Libraries
import pandas as pd 
import kennard_stone as ks

# Load Data
df = pd.read_csv('debutanizer_data.csv')

# Train Test Split
X_train, X_test, y_train, y_test = ks.train_test_split(df.drop('y', axis=1), df['y'], test_size=0.2)

# Statistical Models
# Linear Regression (MLR)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("MLR Coefficients:", lr.coef_)
print("MLR Predictions:", lr.predict(X_test.head()))

# Principal Component Regression (PCR)
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
lr_pcr = LinearRegression()
lr_pcr.fit(X_train_pca, y_train)
print("PCR Coefficients:", lr_pcr.coef_)
print("PCR Predictions:", lr_pcr.predict(X_test_pca[:5]))