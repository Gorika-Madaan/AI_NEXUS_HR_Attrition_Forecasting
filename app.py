
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder 


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Retrieve form values
    form_values = request.form.to_dict()

    
    feature_names = [
        'Age', 'BusinessTravel', 'DailyRate',
       'Department', 'DistanceFromHome', 'Education', 'EducationField',
       'EmployeeCount', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager'
        
    ]

    # To Convert categorical variables to numeric using saved encoders
    for column in ['Department', 'BusinessTravel','EducationField','Gender','JobRole', 'MaritalStatus','Over18', 'OverTime']:
        if form_values[column] in label_encoders[column].classes_:
            form_values[column] = label_encoders[column].transform([form_values[column]])[0]
        else:
            return render_template("index.html", prediction_text="Error: Invalid categorical value for {}".format(column))


    # To Extract features in the same order as used in training
    try:
        float_features = [float(form_values.get(col, 0)) for col in feature_names]
    except ValueError:
        return render_template("index.html", prediction_text="Error: Invalid input format")

    # Convert to numpy array and make prediction
    features = [np.array(float_features)]
    prediction = model.predict(features)

    # Render the result
    return render_template("index.html", prediction_text="The likeliness of Attrition is : {}".format(prediction[0]))


if __name__ == "__main__":
    flask_app.run(debug=True)