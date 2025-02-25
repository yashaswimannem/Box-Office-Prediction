import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the files
credits = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_credits.csv')
movies = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_movies.csv')

credits.rename(columns={'movie_id': 'id'}, inplace=True)

# Merge the datasets on the 'id' column
df = pd.merge(movies, credits, on='id')

# Handle missing values
df = df.dropna(subset=['budget', 'revenue'])  # Drop rows without budget or revenue
df['runtime'].fillna(df['runtime'].median(), inplace=True)  # Fill missing runtime with median

# Drop irrelevant columns
columns_to_drop = ['title_x', 'release_date', 'status', 'crew']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Convert 'genres' from string to list of dictionaries, and then one-hot encode them
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
df_exploded = df.explode('genres')  # Explode the list into multiple rows
df = pd.concat([df, pd.get_dummies(df_exploded['genres'])], axis=1)  # One-hot encode and concatenate back
df = df.drop(columns=['genres'])  # Drop original genres column

# Check for 'release_date' column
if 'release_date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['summer_release'] = df['release_month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)

# Log-transform budget and revenue to handle skewness
df['log_budget'] = df['budget'].apply(lambda x: np.log1p(x))
df['log_revenue'] = df['revenue'].apply(lambda x: np.log1p(x))

# Extract top 3 actors from the cast and join them into a string
df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
df['cast'] = df['cast'].apply(lambda x: ' '.join(x))

# Extract the director from the crew
if 'crew' in df.columns:
    df['director'] = df['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
    df['director'] = df['director'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    df = df.dropna(subset=['director'])  # Drop rows with missing directors

# One-hot encode 'original_language'
df = pd.get_dummies(df, columns=['original_language'], drop_first=True)

# Extract top production companies and one-hot encode them
df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:1])
df['production_companies'] = df['production_companies'].apply(lambda x: ' '.join(x))
df = pd.get_dummies(df, columns=['production_companies'], drop_first=True)

# Separate features and target
X = df[['budget', 'vote_count', 'popularity', 'runtime', 'vote_average']]  # Adjust features accordingly
y = df['revenue']  # Target variable (adjust as needed)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Save the trained model to a file
joblib.dump(model, 'movie_random_forest_model.pkl')
print("Model saved successfully!")

# Now for making predictions with new data:
# Ensure that the new input data matches the trained model's features

# Assuming input data for prediction is provided as individual variables
budget = 15000000
vote_count = 500
popularity = 20.5
runtime = 120
vote_average = 7.5

# Input features for prediction
input_features = pd.DataFrame([{
    'budget': budget,
    'vote_count': vote_count,
    'popularity': popularity,
    'runtime': runtime,
    'vote_average': vote_average
}])

# Check if the input features match the trained model's feature columns
expected_columns = X_train.columns.tolist()

# Add missing columns (if any) to input data
for col in expected_columns:
    if col not in input_features.columns:
        input_features[col] = 0  # Add missing feature with a default value (e.g., 0)

# Reorder columns to match the training set
input_features = input_features[expected_columns]

# Make predictions
prediction = model.predict(input_features)
print(f"Predicted Revenue: {prediction[0]}")

import matplotlib.pyplot as plt

# Plot actual vs predicted revenue
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Line')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.legend()
plt.show()