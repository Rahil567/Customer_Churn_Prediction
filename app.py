# app.py - Flask application for Customer Churn Prediction

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)

# A secret key is required by Flask to securely encrypt and use session variables

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'local-dev-fallback-key')

# Load Saved ML Components

MODEL_PATH = 'churn_model.pkl'
SCALER_PATH = 'scaler.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
else:
    class MockModel:
        def predict(self, X): return np.array([1])
    class MockScaler:
        def transform(self, X): return X
    model = MockModel()
    scaler = MockScaler()

# Complete performance metrics mapped precisely from training data

ALL_MODELS_METRICS = {
    'lr': {
        'name': 'Logistic Regression',
        'accuracy': '84.55%', 'f1': '85.82%', 'auc': '84.59%',
        'tn': '38,214', 'fp': '6,767', 'fn': '8,841', 'tp': '47,220'
    },
    'rf': {
        'name': 'Random Forest Classifier',
        'accuracy': '92.46%', 'f1': '93.51%', 'auc': '91.80%',
        'tn': '38,558', 'fp': '6,423', 'fn': '1,193', 'tp': '54,868'
    },
    'xgb': {
        'name': 'XGBoost Classifier',
        'accuracy': '93.31%', 'f1': '94.28%', 'auc': '92.54%',
        'tn': '38,493', 'fp': '6,488', 'fn': '275', 'tp': '55,786'
    }
}

@app.route('/', methods=['GET'])
def index():

    # .pop() extracts the data for rendering and IMMEDIATELY erases it from the session.
    # Therefore, hitting refresh issues a fresh GET request, seeing blank defaults.

    prediction = session.pop('prediction', None)
    form_values = session.pop('form_values', None)
    reasons = session.pop('reasons', None)
    strategies = session.pop('strategies', None)
    error = session.pop('error', None)

    return render_template(
        'index.html', 
        prediction=prediction, 
        form_values=form_values, 
        error=error, 
        reasons=reasons,
        strategies=strategies,
        all_metrics=ALL_MODELS_METRICS
    )

@app.route('/predict', methods=['POST'])
def predict():

    # 1. Capture Raw Form Values As Pristine Strings

    raw_age = request.form.get('Age', '').strip()
    raw_gender = request.form.get('Gender', '').strip()
    raw_tenure = request.form.get('Tenure', '').strip()
    raw_usage_frequency = request.form.get('Usage_Frequency', '').strip()
    raw_support_calls = request.form.get('Support_Calls', '').strip()
    raw_payment_delay = request.form.get('Payment_Delay', '').strip()
    raw_total_spend = request.form.get('Total_Spend', '').strip()
    raw_last_interaction = request.form.get('Last_Interaction', '').strip()
    raw_sub_type = request.form.get('Subscription_Type', '').strip()
    raw_contract_length = request.form.get('Contract_Length', '').strip()

    # Form state layout object passed strictly to hold original string states

    form_values = {
        'Age': raw_age, 'Gender': raw_gender, 'Tenure': raw_tenure,
        'Usage_Frequency': raw_usage_frequency, 'Support_Calls': raw_support_calls,
        'Payment_Delay': raw_payment_delay, 'Total_Spend': raw_total_spend,
        'Last_Interaction': raw_last_interaction, 'Subscription_Type': raw_sub_type,
        'Contract_Length': raw_contract_length
    }

    # 2. Strict Empty Selection Form Validation Trigger

    if not (raw_age and raw_gender and raw_tenure and raw_usage_frequency and 
            raw_support_calls and raw_payment_delay and raw_total_spend and 
            raw_last_interaction and raw_sub_type and raw_contract_length):
        session['error'] = "Please fill the form completely."
        session['form_values'] = form_values
        return redirect(url_for('index'))

    try:
        # 3. Safe Numeric Data Parsing

        age = float(raw_age)
        gender = int(raw_gender)
        tenure = float(raw_tenure)
        usage_frequency = float(raw_usage_frequency)
        support_calls = float(raw_support_calls)
        payment_delay = float(raw_payment_delay)
        total_spend = float(raw_total_spend)
        last_interaction = float(raw_last_interaction)
    except ValueError:
        session['error'] = "Please enter valid numeric inputs."
        session['form_values'] = form_values
        return redirect(url_for('url_for' if False else 'index'))

    # 4. Logical Constraints Checks

    if age < 18:
        session['error'] = "Please enter a valid age (Age must be >= 18)."
        session['form_values'] = form_values
        return redirect(url_for('index'))

    if age < 0 or tenure < 0 or usage_frequency < 0 or support_calls < 0 or payment_delay < 0 or total_spend < 0 or last_interaction < 0:
        session['error'] = "Negative values are not allowed in any field."
        session['form_values'] = form_values
        return redirect(url_for('index'))

    try:
        # 5. ML Matrix Processing Layout Construction

        input_row = {
            'Age': [age], 'Gender': [gender], 'Tenure': [tenure], 'Usage Frequency': [usage_frequency],
            'Support Calls': [support_calls], 'Payment Delay': [payment_delay], 'Total Spend': [total_spend],
            'Last Interaction': [last_interaction],
            'Subscription Type_Basic': [1 if raw_sub_type == 'Basic' else 0],
            'Subscription Type_Premium': [1 if raw_sub_type == 'Premium' else 0],
            'Subscription Type_Standard': [1 if raw_sub_type == 'Standard' else 0],
            'Contract Length_Annual': [1 if raw_contract_length == 'Annual' else 0],
            'Contract Length_Monthly': [1 if raw_contract_length == 'Monthly' else 0],
            'Contract Length_Quarterly': [1 if raw_contract_length == 'Quarterly' else 0]
        }

        input_df = pd.DataFrame(input_row)
        scaled_features = scaler.transform(input_df)
        pred_class = int(model.predict(scaled_features)[0])

        reasons = []
        strategies = []
        
        if pred_class == 1:
            if support_calls >= 4:
                reasons.append(f"High support ticket velocity ({int(support_calls)} complaints logged).")
                strategies.append("Proactively assign an account health retention advocate.")
            if payment_delay > 7:
                reasons.append(f"Critical accounting lag detected (Bill delayed by {int(payment_delay)} days).")
                strategies.append("Issue targeted billing adjustments or options.")
            if raw_contract_length == 'Monthly' and usage_frequency < 15:
                reasons.append("Fragile month-to-month agreement terms linked with declining application usage.")
                strategies.append("Deploy strategic loyalty incentives to migrate user into structured annual options.")
            if not reasons:
                reasons.append("General behavior trends match standard churn profiles across operational variances.")
                strategies.append("Trigger general customer success engagement protocols.")

        # Save output objects directly into the session store

        session['prediction'] = pred_class
        session['reasons'] = reasons
        session['strategies'] = strategies
        session['form_values'] = form_values
        
        return redirect(url_for('index'))

    except Exception as e:
        session['error'] = f"Processing Misfire: {str(e)}"
        session['form_values'] = form_values
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)