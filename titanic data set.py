import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Titanic-Dataset.csv')
print(df.describe())
df = df.drop('Cabin', axis=1)
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop_duplicates(inplace=True)
print("\nSummary of the cleaned dataset:")
print(df.info())
print(df.describe())

plt.figure(figsize=(10, 6))
df['Age'].value_counts().sort_index().plot(kind='line')
plt.title('Number of Passengers Across Different Ages')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()


plt.figure(figsize=(8, 6))
df['Pclass'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of Passengers for Each Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count of Passengers')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Fare'], c=df['Survived'], cmap='viridis')
plt.title('Relationship Between Age and Fare (Color-coded by Survival)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.colorbar(label='Survived (0 = No, 1 = Yes)')
plt.show()


plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=20)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()


