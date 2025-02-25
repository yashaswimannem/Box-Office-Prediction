import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
credits = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_credits.csv')
movies = pd.read_csv(r'C:\Users\Yashaswi\Documents\Projects\Box office recommendation\tmdb_5000_movies.csv')
df = pd.merge(movies, credits, on='id')
credits.rename(columns={'movie_id': 'id'}, inplace=True)

print("Initial Data:")
print(df.head())


# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Impute missing values
# For numeric columns, use mean imputation
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# For categorical columns, use mode imputation
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Check again for missing values after imputation
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Check the DataFrame columns
print("\nDataFrame Columns:")
print(df.columns)

# Define features and target variable
X = df.drop('correct_target_variable_name', axis=1)  # Replace 'correct_target_variable_name' with your actual target column name
y = df['correct_target_variable_name']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Save the trained model (optional)
import joblib
joblib.dump(model, 'clothing_suggestion_model.pkl')

print("Model saved as 'clothing_suggestion_model.pkl'.")