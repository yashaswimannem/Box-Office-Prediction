import pandas as pd
import numpy as np
import ast

# Load the files
credits = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_credits.csv')
movies = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_movies.csv')

credits.rename(columns={'movie_id': 'id'}, inplace=True)

# Merge the datasets on the 'id' column
df = pd.merge(movies, credits, on='id')

# Debugging: Check columns after merging
print("Columns after merging:", df.columns.tolist())

# Handle missing values
df = df.dropna(subset=['budget', 'revenue'])  # Drop rows without budget or revenue
df['runtime'].fillna(df['runtime'].median(), inplace=True)  # Fill missing runtime with median

# Drop irrelevant columns (check if columns exist before dropping)
columns_to_drop = ['title_x', 'release_date', 'status', 'crew']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Debugging: Check columns after dropping irrelevant columns
print("Columns after dropping irrelevant columns:", df.columns.tolist())

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
else:
    print("'release_date' column not found. Available columns:", df.columns.tolist())

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
else:
    print("'crew' column not found. Available columns:", df.columns.tolist())

# One-hot encode 'original_language'
df = pd.get_dummies(df, columns=['original_language'], drop_first=True)

# Extract top production companies and one-hot encode them
df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:1])
df['production_companies'] = df['production_companies'].apply(lambda x: ' '.join(x))
df = pd.get_dummies(df, columns=['production_companies'], drop_first=True)

# Separate features and target
X = df.drop(columns=['log_revenue', 'revenue'])
y = df['log_revenue']

# Your data is now ready for model training!
print("Data preprocessing complete. Features and target are ready.")

# Import necessary libraries
# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Assuming df is your cleaned DataFrame

# Features and Target Variable
# X = Features, y = Target (replace 'target_column' with the name of your target variable, e.g., revenue)
X = df[['budget', 'vote_count', 'popularity', 'runtime', 'vote_average']]  # Adjust features accordingly
y = df['revenue']  # Target variable (adjust as needed)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Save the trained model to a file
joblib.dump(model, 'random_forest_model.pkl')
print("Model saved successfully!")

# To load the model later and make predictions:
# loaded_model = joblib.load('random_forest_model.pkl')
# predictions = loaded_model.predict(X_test)
# print(predictions)
