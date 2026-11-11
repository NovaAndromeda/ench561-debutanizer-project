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
print(lr.coef_)