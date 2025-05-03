from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])

        # Create dataframe to mimic training structure
        input_df = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'SibSp': sibsp,
            'Parch': parch
        }])

        # Apply one-hot encoding like in training
        input_encoded = pd.get_dummies(input_df)

        # Align columns to match model input
        expected_cols = ['Pclass', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']
        for col in expected_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0  # Add missing dummy cols

        input_encoded = input_encoded[expected_cols]

        prediction = model.predict(input_encoded)[0]
        result = "Survived" if prediction == 1 else "Did not survive"
        return render_template('index.html', prediction_text=f'The passenger {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
