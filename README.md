# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import and preprocess the data Load dataset and encode categorical values

2.Split features and target Define independent variable X and dependent variable y

3.Train the model Fit DecisionTreeRegressor to the data

4.Predict and visualize Predict new values and plot the results


## Program:
```
/*

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kervin.S
RegisterNumber:  212225220051 */

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("Employee.csv")
df.head()
df.info()
df.isnull().sum()
df.rename(columns={'Departments ': 'Department'}, inplace=True)
df = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)
df.head()
X = df.drop('left', axis=1)  
y = df['left']  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)
plt.show()

```

## Output:
<img width="1369" height="883" alt="image" src="https://github.com/user-attachments/assets/c3571cd3-c570-4bf1-b5eb-4ef15537cc1b" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
