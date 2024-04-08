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

x_points = np.linspace(0,5000, 10000)
y_points = model(x_points)

ax, fig = plt.subplots()
plt.xlim(0, 3000)
plt.ylim(0, 500000)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.plot(x_points, y_points, lw =3, color='blue')
plt.scatter(X_test, Y_test, color ='black', s=50)
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

x_points = np.linspace(0,5000, 10000)
y_points = model(x_points)

ax, fig = plt.subplots()
plt.xlim(0, 3000)
plt.ylim(0, 500000)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.plot(x_points, y_points, lw =3, color='blue')
plt.scatter(X_test, Y_test, color ='black', s=50)
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

x_points = np.linspace(0,5000, 10000)
y_points = model(x_points)

ax, fig = plt.subplots()
plt.xlim(0, 3000)
plt.ylim(0, 500000)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.plot(x_points, y_points, lw =3, color='blue')
plt.scatter(X_test, Y_test, color ='black', s=50)
plt.show()