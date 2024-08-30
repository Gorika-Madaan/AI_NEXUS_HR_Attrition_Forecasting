import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load the dataset
df = pd.read_csv("hr_attrition.csv")

# Define predictors (X) and target (Y)
X = df.drop(columns=["Attrition", "EmployeeNumber"])
Y = df["Attrition"]  # Target


label_encoders = {}
for column in ['Department', 'BusinessTravel','EducationField','Gender','JobRole', 'MaritalStatus','Over18', 'OverTime']:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Scale the predictors
scale = MinMaxScaler()
X = scale.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model_poly = SVC(kernel = "poly")
model_poly.fit(X_train, Y_train)
pred_test_poly = model_poly.predict(X_test)

res_poly = np.mean(pred_test_poly == Y_test)
print(res_poly)


pickle.dump(model_poly, open("model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

import os
os.getcwd()

