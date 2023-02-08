# save this as app.py
from flask import Flask, escape, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('MLMODEL.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method ==  'POST':
        Gender = request.form['Gender']
        Married = request.form['Married']
        Dependents = request.form['Dependents']
        Education = request.form['Education']
        Employed = request.form['Self_Employed']
        Credithistory = float(request.form['credithistory'])
        Area = request.form['Property_Area']
        ApplicantIncome = float(request.form['Applicantincome'])
        CoapplicantIncome = float(request.form['Coapplicantincome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])

        # gender
        if (Gender == "Male"):
            Gender_Male=1
        else:
            Gender_Male=0
        
        # married
        if(Married=="Yes"):
            Married_Yes = 1
        else:
            Married_Yes=0

        # dependents
        if(Dependents=='1'):
            Dependents_1 = 1
            Dependents_2 = 0
            Dependents_3 = 0
        elif(Dependents == '2'):
            Dependents_1 = 0
            Dependents_2 = 1
            Dependents_3 = 0
        elif(Dependents=="3+"):
            Dependents_1 = 0
            Dependents_2 = 0
            Dependents_3 = 1
        else:
            Dependents_1 = 0
            Dependents_2 = 0
            Dependents_3 = 0  

        # education
        if (Education=="Non-graduated"):
            Education_Not_Graduate	=1
        else:
            Education_Not_Graduate	=0

        # employed
        if (Employed == "Yes"):
            Self_Employed_Yes=1
        else:
            Self_Employed_Yes=0

        # property area

        if(Area=="Yes"):
            Property_Area_Urban=1
        else:
            Property_Area_Urban=0
        


        ApplicantIncomelog = np.log(ApplicantIncome)
        CoapplicantincomeLog = np.log(1 + CoapplicantIncome)
        LoanAmountlog = np.log(LoanAmount)
        Loan_Amount_Termlog = np.log(Loan_Amount_Term)

        prediction = model.predict([[ Credithistory, ApplicantIncomelog,LoanAmountlog, Loan_Amount_Termlog, CoapplicantincomeLog, Gender_Male, Married_Yes, Dependents_1, Dependents_2, Dependents_3,Education_Not_Graduate, Self_Employed_Yes, Property_Area_Urban]])

        # print(prediction)

        if(prediction=="N"):
            prediction="No"
        else:
            prediction="Yes"


        return render_template("prediction.html", prediction_text="--- LOAN STATUS IS ---{}".format(prediction))




    else:
        return render_template("prediction.html")



if __name__ == "__main__":
    app.run(debug=True)