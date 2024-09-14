import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
housing_data = fetch_california_housing()
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['target'] = housing_data.target
df.head()
df.shape
df.describe()
plt.figure(figsize=(5,5))
sns.heatmap(df.corr(), cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size':8}, cmap='coolwarm')
print(housing_data)
plt.figure(figsize=(8, 6))
sns.histplot(df['target'],kde=True)
plt.title('Distribution of House Prices (Median House Value)')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Frequency')
plt.show()
x=df.drop(['target'], axis=1)
y=df['target']
x.head()
y.head()
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=2)
x.shape, x_train.shape, x_test.shape
classify = XGBRegressor()
classify.fit(x_train , y_train)
train_data_prediction=classify.predict(x_train)
print(train_data_prediction)
score_1=metrics.r2_score(y_train, train_data_prediction)
score_2=metrics.mean_absolute_error(y_train, train_data_prediction)
print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)
print(score_1-score_2)
plt.figure(figsize=(5, 5))
plt.scatter(y_train, train_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()
test_data_prediction=classify.predict(x_test)
print(test_data_prediction)
score_1=metrics.r2_score(y_test, test_data_prediction)
score_2=metrics.mean_absolute_error(y_test, test_data_prediction)
print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)
print(score_1-score_2)
