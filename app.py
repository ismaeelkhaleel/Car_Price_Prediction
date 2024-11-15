from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
data = pd.read_csv('car-data.csv')

# Preprocess dataset
data = pd.get_dummies(data, columns=['fuel', 'condition'], drop_first=True)

# Define features (X) and target variable (y)
target_col = 'sellingPrice'
X = data.drop(columns=[target_col, 'carName', 'model'])
y = data[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Flask Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction based on form inputs."""
    try:
        # Get input data from form
        year = int(request.form['year'])
        mileage = float(request.form['mileage'])
        driven = int(request.form['driven'])
        condition = request.form['condition']
        fuel = request.form['fuel']

        # Prepare data for prediction
        input_data = pd.DataFrame({
            'year': [year],
            'mileage': [mileage],
            'driven': [driven],
            'fuel_Diesel': [1 if fuel == 'Diesel' else 0],
            'fuel_Electric': [1 if fuel == 'Electric' else 0],
            'fuel_Petrol': [1 if fuel == 'Petrol' else 0],
            'condition_Good': [1 if condition == 'Good' else 0],
            'condition_Needs Work': [1 if condition == 'Needs Work' else 0]
        })

        # Add missing columns
        for column in X.columns:
            if column not in input_data.columns:
                input_data[column] = 0

        # Reorder columns
        input_data = input_data[X.columns]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Render the result
        return render_template('index.html', prediction=f"Predicted Selling Price: ₹{prediction:.2f}")

    except Exception as e:
        return render_template('index.html', error=f"Error during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)
