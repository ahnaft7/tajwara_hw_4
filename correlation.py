"""
Ahnaf Tajwar
Class: CS 677
Date: 4/07/24
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): This problem is to
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

print(df.head())

# Filtering dataframe for death event = 0 and the four specific features
df_0 = df[df['DEATH_EVENT'] == 0][['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets']]

# Filtering dataframe for death event = 1 and the four specific features
df_1 = df[df['DEATH_EVENT'] == 1][['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets']]

print("DataFrame for death event = 0:")
print(df_0.head())

print("\nDataFrame for death event = 1:")
print(df_1.head())

# Calculate correlation matrices
corr_matrix_0 = df_0.corr()
corr_matrix_1 = df_1.corr()

# Plot correlation matrix for df_0
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_0, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Death Event = 0")
plt.savefig("correlation_matrix_death_0.png")
plt.close()

# Plot correlation matrix for df_1
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Death Event = 1")
plt.savefig("correlation_matrix_death_1.png")
plt.close()

# Comparing Models for surviving patients
print("\n-----Comparing Models for surviving patients-----\n")

X = np.array(df_0['creatinine_phosphokinase'].values)
Y = np.array(df_0['platelets'].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)

# Simple Linear Regression
print("\n-----Linear-----\n")
degree = 1
weights = np.polyfit(X_train,Y_train, degree)
print("weights: ", weights)
model = np.poly1d(weights)
predicted = model(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print("rmse: ", rmse)
# Calculate residuals
residuals = Y_test - predicted
# Calculate squared residuals
squared_residuals = residuals ** 2
# Compute SSE (Sum of Squared Residuals)
SSE = sum(squared_residuals)
print("Sum of Squared Residuals (SSE):", SSE)
r2 = r2_score(Y_test, predicted)
print("r2: ", r2)

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Quadratic
print("\n-----Quadratic-----\n")
degree = 2
weights = np.polyfit(X_train,Y_train, degree)
print("weights: ", weights)
model = np.poly1d(weights)
predicted = model(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print("rmse: ", rmse)
# Calculate residuals
residuals = Y_test - predicted
# Calculate squared residuals
squared_residuals = residuals ** 2
# Compute SSE (Sum of Squared Residuals)
SSE = sum(squared_residuals)
print("Sum of Squared Residuals (SSE):", SSE)
r2 = r2_score(Y_test, predicted)
print("r2: ", r2)

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Quadratic Model')
plt.legend()
plt.show()

# Cubic
print("\n-----Cubic-----\n")
degree = 3
weights = np.polyfit(X_train,Y_train, degree)
print("weights: ", weights)
model = np.poly1d(weights)
predicted = model(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print("rmse: ", rmse)
# Calculate residuals
residuals = Y_test - predicted
# Calculate squared residuals
squared_residuals = residuals ** 2
# Compute SSE (Sum of Squared Residuals)
SSE = sum(squared_residuals)
print("Sum of Squared Residuals (SSE):", SSE)
r2 = r2_score(Y_test, predicted)
print("r2: ", r2)

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Cubic Model')
plt.legend()
plt.show()

# Logarithmic
print("\n-----y = a log x + b (GLM - generalized linear model)-----\n")

# Transform the independent variable using natural logarithm
X_log = np.log(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_log, Y, test_size=0.5, random_state=1)

# Fit the model
degree = 1
weights = np.polyfit(X_train, Y_train, degree)
print("weights: ", weights)
model = np.poly1d(weights)

# Make predictions
predicted = model(X_test)

# Compute metrics
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print("rmse: ", rmse)

# Calculate residuals
residuals = Y_test - predicted
squared_residuals = residuals ** 2
SSE = sum(squared_residuals)
print("Sum of Squared Residuals (SSE):", SSE)
r2 = r2_score(Y_test, predicted)
print("r2: ", r2)

# Plot
x_points = np.linspace(0, max(X_log), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('log(Creatinine Phosphokinase)')
plt.ylabel('Platelets')
plt.title('Simple Linear Regression with Log Transformation X')
plt.legend()
plt.show()

# Logarithmic
print("\n-----log y = a log x + b (GLM - generalized linear model)-----\n")

# Transform the independent variable using natural logarithm
X = np.array(df_0['creatinine_phosphokinase'].values.reshape(-1, 1))
Y = np.array(df_0['platelets'].values)
X_log = np.log(X)
Y_log = np.log(Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_log, Y_log, test_size=0.5, random_state=1)

# Fit the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Obtain the coefficients
a = model.coef_[0]
b = model.intercept_

# Make predictions
predicted = model.predict(X_test)

# Compute metrics
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
r2 = r2_score(Y_test, predicted)

# Calculate residuals
residuals = Y_test - predicted
squared_residuals = residuals ** 2
SSE = sum(squared_residuals)
print("Slope (a):", a)
print("Intercept (b):", b)
print("rmse:", rmse)
print("SSE:", SSE)
print("r2:", r2)

# Plot
x_points = np.linspace(0, max(X_log), 100)
y_points = model.predict(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('log(Creatinine Phosphokinase)')
plt.ylabel('log(Platelets)')
plt.title('Simple Linear Regression with Log Transformation Y and X')
plt.legend()
plt.show()