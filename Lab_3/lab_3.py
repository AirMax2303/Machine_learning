import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Task 1: Download the "Boston Housing" dataset
boston_data = load_boston()
boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['PRICE'] = boston_data.target

# Task 2: Study the structure and content of the data
print("Dataset structure:")
print(boston_df.info())

# Task 3: Preliminary analysis of the data
print("\nSummary statistics:")
print(boston_df.describe())

# Check for missing values
print("\nMissing values:")
print(boston_df.isnull().sum())

# Check for outliers using boxplots
plt.figure(figsize=(15, 8))
sns.boxplot(data=boston_df, orient="h")
plt.title("Boxplot of Boston Housing Dataset")
plt.show()

# Check correlations between variables
correlation_matrix = boston_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Task 4: Compare average real estate prices in different areas of Boston
average_prices_by_area = boston_df.groupby('RAD')['PRICE'].mean()
print("\nAverage real estate prices by area:")
print(average_prices_by_area)

# Identify the most expensive and cheapest areas
most_expensive_area = average_prices_by_area.idxmax()
cheapest_area = average_prices_by_area.idxmin()
print("\nMost expensive area:", most_expensive_area)
print("Cheapest area:", cheapest_area)

# Task 5: Investigate the influence of various factors on the price of real estate
influence_factors = ['CRIM', 'RM', 'AGE', 'DIS', 'TAX']
for factor in influence_factors:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=boston_df[factor], y=boston_df['PRICE'])
    plt.title(f'Real Estate Price vs {factor}')
    plt.show()

# Task 6: Split the data into training and test samples
X = boston_df.drop('PRICE', axis=1)
y = boston_df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Task 7: Train a linear regression model on a training sample
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

# Task 8: Evaluate the quality of the model on the test sample
y_pred = linear_reg_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")

# Task 9: Improve the quality of the model by using Lasso regularization
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)

# Task 10: Visualize the results of the prediction
y_pred_lasso = lasso_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lasso)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Lasso)")
plt.show()

# Task 11: Perform cross-validation of the model
cv_mse = cross_val_score(lasso_model, X, y, scoring='neg_mean_squared_error', cv=5)
cv_rmse = np.sqrt(-cv_mse)
cv_r2 = cross_val_score(lasso_model, X, y, scoring='r2', cv=5)

print("\nCross-Validation Metrics:")
print(f"Cross-Validation Mean Squared Error (CV MSE): {cv_mse.mean()}")
print(f"Cross-Validation Root Mean Squared Error (CV RMSE): {cv_rmse.mean()}")
print(f"Cross-Validation R^2 Score (CV R^2): {cv_r2.mean()}")
