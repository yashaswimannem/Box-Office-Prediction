# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Assuming your preprocessed dataframe is 'df' and your target variable is 'log_revenue'

# Features (X) and Target (y)
# Select relevant features you want to use for prediction
X = df[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']]  # Example feature columns
y = np.log1p(df['revenue'])  # Log transform the revenue for normalization

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training (Using RandomForest Regressor)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# 4. Make Predictions on the Test Set
y_pred = rf_model.predict(X_test)

# 5. Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# 6. Feature Importance (Optional - to understand which features are important)
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.show()
