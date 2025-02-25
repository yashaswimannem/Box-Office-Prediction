from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('movie_random_forest_model.pkl')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form (from user input)
        budget = float(request.form['budget'])
        vote_count = float(request.form['vote_count'])
        popularity = float(request.form['popularity'])
        runtime = float(request.form['runtime'])
        vote_average = float(request.form['vote_average'])
        
        # Format the input into a numpy array for prediction
        features = np.array([[budget, vote_count, popularity, runtime, vote_average]])
        
        # Predict the revenue using the model
        prediction = model.predict(features)
        
        # Return the predicted revenue as part of the response
        return render_template('index.html', prediction_text=f'Predicted Revenue: ${prediction[0]:,.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
