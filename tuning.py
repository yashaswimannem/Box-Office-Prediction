# Required Libraries
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify

# Load the files
credits = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_credits.csv')
movies = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_movies.csv')

# Rename the column and merge datasets
credits.rename(columns={'movie_id': 'id'}, inplace=True)
df = pd.merge(movies, credits, on='id')

# Data Preprocessing
df = df.dropna(subset=['budget', 'revenue'])
df['runtime'].fillna(df['runtime'].median(), inplace=True)
columns_to_drop = ['title_x', 'release_date', 'status', 'crew']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
df_exploded = df.explode('genres')
df = pd.concat([df, pd.get_dummies(df_exploded['genres'])], axis=1)
df = df.drop(columns=['genres'])

# Log-transform budget and revenue
df['log_budget'] = df['budget'].apply(lambda x: np.log1p(x))
df['log_revenue'] = df['revenue'].apply(lambda x: np.log1p(x))

# Extract top 3 actors from the cast
df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
df['cast'] = df['cast'].apply(lambda x: ' '.join(x))

# Extract the director from the crew (if column exists)
if 'crew' in df.columns:
    df['director'] = df['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
    df['director'] = df['director'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    df = df.dropna(subset=['director'])

# One-hot encode 'original_language'
df = pd.get_dummies(df, columns=['original_language'], drop_first=True)

# One-hot encode 'production_companies'
df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:1])
df['production_companies'] = df['production_companies'].apply(lambda x: ' '.join(x))
df = pd.get_dummies(df, columns=['production_companies'], drop_first=True)

# Separate features and target
X = df[['budget', 'vote_count', 'popularity', 'runtime', 'vote_average', 'log_budget']]  # Adjust features
y = df['log_revenue']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best Parameters after tuning
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Final Model Training with best parameters
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'random_forest_model.pkl')
print("Model saved successfully!")

# Model Evaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Flask App for Deployment
app = Flask(__name__)

# Load the saved model
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Assuming input comes as JSON
    data = np.array([data])  # Convert the input to a NumPy array for model input
    prediction = model.predict(data)
    return jsonify({'predicted_log_revenue': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
