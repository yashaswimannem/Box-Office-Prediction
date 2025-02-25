import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
model = joblib.load('random_forest_model.pkl')

# Function to preprocess input data (for simplicity, this is just a placeholder)
def preprocess_movie_data(movie_data):
    """
    This function takes raw movie data, preprocesses it (like encoding categorical variables),
    and returns a DataFrame ready for prediction.
    
    For example, if the cast and crew are text, you would need to process them into features.
    Here, it's just returning numerical features for simplicity.
    """
    # Extracting log_budget as an example; you should add other preprocessing steps as needed
    movie_data['log_budget'] = movie_data['budget'].apply(lambda x: np.log1p(x))
    
    # For simplicity, let's assume we're using the following features:
    features = ['budget', 'vote_count', 'popularity', 'runtime', 'vote_average', 'log_budget']
    
    return movie_data[features]

# Sample input movie data
movie_input = {
    "name": "Bahubali",
    "cast": ["prabhas", "Anushka", "rana"],
    "crew": ["rajamouli", "dvv dannaya"],
    "budget": 29730100,
    "vote_count": 5733,
    "popularity": 91.0,
    "runtime": 158,
    "vote_average": 8.0
}

# Convert the movie_input dictionary to a DataFrame
movie_df = pd.DataFrame([movie_input])

# Preprocess the input movie data
processed_movie_data = preprocess_movie_data(movie_df)

# Make prediction
predicted_log_revenue = model.predict(processed_movie_data)

# Convert log revenue back to actual revenue
predicted_revenue = np.expm1(predicted_log_revenue)

print(f"Predicted box office revenue for '{movie_input['name']}': ${predicted_revenue[0]:,.2f}")
