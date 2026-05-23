from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load('hr_attrition_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():

    prediction_text = None
    probability = None
    risk = None
    recommendation = None

    if request.method == 'POST':

        satisfaction_level = float(request.form['satisfaction_level'])
        last_evaluation = float(request.form['last_evaluation'])
        number_project = int(request.form['number_project'])
        average_montly_hours = int(request.form['average_montly_hours'])
        time_spend_company = int(request.form['time_spend_company'])
        Work_accident = int(request.form['Work_accident'])
        promotion_last_5years = int(request.form['promotion_last_5years'])
        Department = request.form['Department']
        salary = request.form['salary']
        overworked = request.form['overworked']
        satisfaction = request.form['satisfaction']

        # Create DataFrame
        input_data = pd.DataFrame([{
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': Work_accident,
            'promotion_last_5years': promotion_last_5years,
            'Department': Department,
            'salary': salary,
            'overworked': overworked,
            'satisfaction': satisfaction
        }])

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        percent = round(probability * 100, 2)

        prediction_text = 'Employee Will Leave' if prediction == 1 else 'Employee Will Stay'

        # Risk Level
        if percent < 40:
            risk = 'Low Attrition Risk'
            recommendation = 'Employee needs continuous engagement and recognition.'

        elif percent < 70:
            risk = 'Medium Attrition Risk'
            recommendation = 'Review employee promotion, salary and satisfaction.'

        else:
            risk = 'High Attrition Risk'
            recommendation = 'Urgent HR intervention and direct interaction required.'

        probability = percent

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        probability=probability,
        risk=risk,
        recommendation=recommendation
    )


if __name__ == '__main__':
    app.run(debug=True)