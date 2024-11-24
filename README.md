# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation:

Load the dataset and preprocess it by handling missing values, encoding categorical data, and splitting features (X) and target (y).
Data Splitting:

Split the data into training and test sets using an 80-20 split.
Model Training:

Train a logistic regression model on the training set.
Evaluation and Prediction:

Predict on the test set, calculate accuracy, and evaluate the model using metrics (e.g., confusion matrix, classification report).






 
 
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sanjai M
RegisterNumber:  24901269
*/import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv("Placement_Data.csv")
print(data.head())

# Create a copy of the dataset
data1 = data.copy()

# Drop unnecessary columns
data1 = data1.drop(["sl_no", "salary"], axis=1)
print(data1.head())

# Check for missing and duplicate values
print("Missing values:\n", data1.isnull().sum())
print("Duplicate rows:\n", data1.duplicated().sum())

# Encode categorical columns
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Prepare features (X) and target (y)
X = data1.iloc[:, :-1]
y = data1["status"]

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

# Make predictions
y_pred = lr.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_report1)

# Predict for a new sample
sample = [[1, 80, 1, 0, 1, 1, 0, 1, 0, 85, 1, 85]]
prediction = lr.predict(sample)
print("Prediction for sample:", prediction)

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png) ![image](https://github.com/user-attachments/assets/b1c01883-dfb0-4d86-b082-e6a6db400712)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
