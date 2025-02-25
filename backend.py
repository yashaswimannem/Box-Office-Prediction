from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('random_forest_model.pkl')

# Function to preprocess input data
def preprocess_movie_data(movie_data):
    movie_data['log_budget'] = movie_data['budget'].apply(lambda x: np.log1p(x))
    
    # Assuming you're using these features for prediction
    features = ['budget', 'vote_count', 'popularity', 'runtime', 'vote_average', 'log_budget']
    
    return movie_data[features]

@app.route('/predict', methods=['POST'])
def predict():
    # Get movie data from the request
    data = request.get_json()

    # Convert the movie data to a DataFrame
    movie_df = pd.DataFrame([data])

    # Preprocess the data
    processed_movie_data = preprocess_movie_data(movie_df)

    # Make prediction
    predicted_log_revenue = model.predict(processed_movie_data)

    # Convert log revenue back to actual revenue
    predicted_revenue = np.expm1(predicted_log_revenue)

    # Return the predicted revenue as a JSON response
    return jsonify({'prediction': predicted_revenue[0]})

if __name__ == '__main__':
    app.run(debug=True)
