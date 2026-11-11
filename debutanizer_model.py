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

# Principal Component Regression (PCR)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# Determine number of principal components to use
pca = PCA()
pca.fit(X_train_scale)
print("Eigenvalues:", pca.explained_variance_)
print("Percent of eigenvalue:", pca.explained_variance_ratio_)
print("Cumulative percent of eigenvalue:", pca.explained_variance_ratio_.cumsum())

pca = PCA(n_components=5)  # Choose number of components based >95% cumulative percent of eigenvalue
pca.fit(X_train_scale)
X_train_pca = pca.transform(X_train_scale)
X_test_pca = pca.transform(X_test_scale)

lr_pcr = LinearRegression()
lr_pcr.fit(X_train_pca, y_train)
print("PCR Coefficients:", lr_pcr.coef_)