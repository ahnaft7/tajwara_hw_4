"""
Ahnaf Tajwar
Class: CS 677
Date: 4/07/24
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): This problem is to
"""

import pandas as pd

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