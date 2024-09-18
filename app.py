from flask import Flask, render_template, request
import pickle
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load the model once at startup
with open('MLMODEL.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home() -> str:
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict() -> str:
    if request.method == 'POST':
        # Collect form data
        form_data = {
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'Credithistory': float(request.form['credithistory']),
            'Property_Area': request.form['Property_Area'],
            'ApplicantIncome': float(request.form['Applicantincome']),
            'CoapplicantIncome': float(request.form['Coapplicantincome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
        }

        # Convert categorical variables to binary variables
        Gender_Male = 1 if form_data['Gender'] == "Male" else 0
        Married_Yes = 1 if form_data['Married'] == "Yes" else 0
        Education_Not_Graduate = 1 if form_data['Education'] == "Non-graduated" else 0
        Self_Employed_Yes = 1 if form_data['Self_Employed'] == "Yes" else 0
        Property_Area_Urban = 1 if form_data['Property_Area'] == "Urban" else 0
        
        # Dependents encoding
        Dependents_1 = int(form_data['Dependents'] == '1')
        Dependents_2 = int(form_data['Dependents'] == '2')
        Dependents_3 = int(form_data['Dependents'] == '3+')

        # Log transformations
        ApplicantIncomelog = np.log(form_data['ApplicantIncome'])
        CoapplicantincomeLog = np.log(1 + form_data['CoapplicantIncome'])
        LoanAmountlog = np.log(form_data['LoanAmount'])
        Loan_Amount_Termlog = np.log(form_data['Loan_Amount_Term'])

        # Prepare input features for prediction
        features = [
            form_data['Credithistory'], 
            ApplicantIncomelog,
            LoanAmountlog, 
            Loan_Amount_Termlog, 
            CoapplicantincomeLog, 
            Gender_Male, 
            Married_Yes, 
            Dependents_1, 
            Dependents_2, 
            Dependents_3,
            Education_Not_Graduate, 
            Self_Employed_Yes, 
            Property_Area_Urban
        ]

        # Make prediction
        prediction = model.predict([features])[0]

        # Map prediction to the output string
        prediction_text = "No" if prediction == "N" else "Yes"

        return render_template("prediction.html", prediction_text=f"--- LOAN STATUS IS --- {prediction_text}")

    return render_template("prediction.html")

if __name__ == "__main__": 
    app.run(debug=True)
