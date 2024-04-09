"""
Ahnaf Tajwar
Class: CS 677
Date: 4/07/24
Homework Problem # 1-3
Description of Problem (just a 1-2 line summary!): These problems are to calculate the correlation matrix for four features related to heart failure in surviving and deceased patients.
    It also is to fit 5 different models to the data, calculate some metrics, and observe which is the best fit.
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

# Calculating correlation matrices
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

# List of models
models = ["Linear", "Quadratic", "Cubic", "GLM (log)", "GLM (log-log)"]

# Defining a dictionary to store SSE values
sse_dict = {model: {"Surviving": None, "Deceased": None} for model in models}

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

sse_dict["Linear"]["Surviving"] = SSE

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Simple Linear Regression (Surviving)')
plt.legend()
# plt.show()
plt.savefig("Linear_death_0.png")

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

sse_dict["Quadratic"]["Surviving"] = SSE

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Quadratic Model (Surviving)')
plt.legend()
# plt.show()
plt.savefig("Quadratic_death_0.png")

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

sse_dict["Cubic"]["Surviving"] = SSE

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Cubic Model (Surviving)')
plt.legend()
# plt.show()
plt.savefig("Cubic_death_0.png")

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

sse_dict["GLM (log)"]["Surviving"] = SSE

# Plot
x_points = np.linspace(0, max(X_log), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('log(Creatinine Phosphokinase)')
plt.ylabel('Platelets')
plt.title('Simple Linear Regression with Log Transformation X (Surviving)')
plt.legend()
# plt.show()
plt.savefig("Log_death_0.png")

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

# Residuals in the original space
residuals_original = np.exp(Y_test) - np.exp(predicted)

# Squared residuals
squared_residuals_original = residuals_original ** 2

# SSE in the original space
SSE_original = sum(squared_residuals_original)
print("SSE in original space:", SSE_original)

sse_dict["GLM (log-log)"]["Surviving"] = SSE_original

# Plot
x_points = np.linspace(0, max(X_log), 100)
y_points = model.predict(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('log(Creatinine Phosphokinase)')
plt.ylabel('log(Platelets)')
plt.title('Simple Linear Regression with Log Transformation Y and X (Surviving)')
plt.legend()
# plt.show()
plt.savefig("Log_log_death_0.png")

# Comparing Models for deceased patients
print("\n-----Comparing Models for deceased patients-----\n")

X = np.array(df_1['creatinine_phosphokinase'].values)
Y = np.array(df_1['platelets'].values)
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

sse_dict["Linear"]["Deceased"] = SSE

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Simple Linear Regression (Deceased)')
plt.legend()
# plt.show()
plt.savefig("Linear_death_1.png")

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

sse_dict["Quadratic"]["Deceased"] = SSE

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Quadratic Model (Deceased)')
plt.legend()
# plt.show()
plt.savefig("Quadratic_death_1.png")

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

sse_dict["Cubic"]["Deceased"] = SSE

# Plot
x_points = np.linspace(0, max(X), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('Creatinine Phosphokinase')
plt.ylabel('Platelets')
plt.title('Cubic Model (Deceased)')
plt.legend()
# plt.show()
plt.savefig("Cubic_death_1.png")

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

sse_dict["GLM (log)"]["Deceased"] = SSE

# Plot
x_points = np.linspace(0, max(X_log), 100)
y_points = model(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('log(Creatinine Phosphokinase)')
plt.ylabel('Platelets')
plt.title('Simple Linear Regression with Log Transformation X (Deceased)')
plt.legend()
# plt.show()
plt.savefig("Log_death_1.png")

# Logarithmic
print("\n-----log y = a log x + b (GLM - generalized linear model)-----\n")

# Transform the independent variable using natural logarithm
X = np.array(df_1['creatinine_phosphokinase'].values.reshape(-1, 1))
Y = np.array(df_1['platelets'].values)
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

# Residuals in the original space
residuals_original = np.exp(Y_test) - np.exp(predicted)

# Squared residuals
squared_residuals_original = residuals_original ** 2

# SSE in the original space
SSE_original = sum(squared_residuals_original)
print("SSE in original space:", SSE_original)

sse_dict["GLM (log-log)"]["Deceased"] = SSE_original

# Plot
x_points = np.linspace(0, max(X_log), 100)
y_points = model.predict(x_points)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Actual')
plt.plot(x_points, y_points, color='blue', label='Predicted')
plt.xlabel('log(Creatinine Phosphokinase)')
plt.ylabel('log(Platelets)')
plt.title('Simple Linear Regression with Log Transformation Y and X (Deceased)')
plt.legend()
# plt.show()
plt.savefig("Log_log_death_1.png")

print('\nSSE Values: ', sse_dict)

# Initialize variables to store the smallest and largest SSE values and corresponding model names
smallest_surviving_sse = float('inf')
largest_surviving_sse = float('-inf')
smallest_surviving_model = None
largest_surviving_model = None

smallest_deceased_sse = float('inf')
largest_deceased_sse = float('-inf')
smallest_deceased_model = None
largest_deceased_model = None

# Iterate through each model and SSE value
for model, sse_values in sse_dict.items():
    # For surviving dataset
    if sse_values['Surviving'] < smallest_surviving_sse:
        smallest_surviving_sse = sse_values['Surviving']
        smallest_surviving_model = model
    if sse_values['Surviving'] > largest_surviving_sse:
        largest_surviving_sse = sse_values['Surviving']
        largest_surviving_model = model
    
    # For deceased dataset
    if sse_values['Deceased'] < smallest_deceased_sse:
        smallest_deceased_sse = sse_values['Deceased']
        smallest_deceased_model = model
    if sse_values['Deceased'] > largest_deceased_sse:
        largest_deceased_sse = sse_values['Deceased']
        largest_deceased_model = model

# Print the smallest and largest SSE values and corresponding model names for surviving dataset
print("\nSmallest SSE for Surviving Dataset:")
print("Model:", smallest_surviving_model)
print("SSE:", smallest_surviving_sse)

print("\nLargest SSE for Surviving Dataset:")
print("Model:", largest_surviving_model)
print("SSE:", largest_surviving_sse)

# Print the smallest and largest SSE values and corresponding model names for deceased dataset
print("\nSmallest SSE for Deceased Dataset:")
print("Model:", smallest_deceased_model)
print("SSE:", smallest_deceased_sse)

print("\nLargest SSE for Deceased Dataset:")
print("Model:", largest_deceased_model)
print("SSE:", largest_deceased_sse)