from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and preprocess it
data = pd.read_csv('car-data.csv')

# Dropping unnecessary columns and encoding categorical data
data.drop(columns=['Car_Name'], inplace=True)  # Drop Car_Name as it is not used for prediction

# One-hot encoding for categorical columns
data = pd.get_dummies(data, drop_first=True)

# Define features (X) and target variable (y)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split the data into training and testing sets
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
        # Get data from the form
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        present_price = float(request.form['present_price'])
        fuel_type = request.form['fuel_type']  # Fuel Type (e.g., Petrol, Diesel)
        transmission = request.form['transmission']  # Transmission (Automatic or Manual)
        seller_type = request.form['seller_type']  # Seller Type (Individual or Dealer)

        # Encoding categorical features for prediction
        fuel_type_dummies = {'Petrol': 1, 'Diesel': 0}  # Simple encoding for fuel type
        transmission_dummies = {'Manual': 1, 'Automatic': 0}  # Encoding for transmission
        seller_type_dummies = {'Individual': 1, 'Dealer': 0}  # Seller Type encoding

        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [km_driven],
            'Fuel_Type_Diesel': [fuel_type_dummies.get(fuel_type, 0)],  # Encode Fuel_Type
            'Transmission_Manual': [transmission_dummies.get(transmission, 0)],  # Encode Transmission
            'Seller_Type_Dealer': [seller_type_dummies.get(seller_type, 0)],  # Encode Seller_Type
        })

        # Ensure the same columns as the training data by adding missing columns
        for column in X.columns:
            if column not in input_data.columns:
                input_data[column] = 0  # If the column is missing, add it with value 0

        # Reorder the columns to match the training data order
        input_data = input_data[X.columns]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Render the result on the same page
        return render_template('index.html', prediction=f"Predicted Selling Price: ₹{prediction:.2f}")

    except KeyError as e:
        return render_template('index.html', error=f"Missing input field: {e}")

    except ValueError as e:
        return render_template('index.html', error=f"Invalid input: {e}")

if __name__ == '__main__':
    app.run(debug=True)
