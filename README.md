# Box Office Prediction using Random Forest

## 📌 Overview
This project predicts **box office revenue** using a **Random Forest Regressor** trained on the **TMDB 5000 Movies** dataset. It leverages key features like **budget, popularity, vote count, runtime, and vote average** to estimate revenue accurately.

## 📂 Dataset
The dataset consists of two CSV files:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

These files contain movie-related data, including **budget, revenue, genres, cast, crew, popularity, and ratings**.

## 🔧 Features Used
- **Budget** 💰
- **Vote Count** 🗳️
- **Popularity** ⭐
- **Runtime** ⏳
- **Vote Average** 📊
- **Genres, Cast, and Director** (one-hot encoded)
- **Production Companies** (one-hot encoded)
- **Seasonal Release Factors** (e.g., summer release)

## 🛠️ Project Workflow
1. **Data Preprocessing** 🛠️
   - Handling missing values
   - One-hot encoding categorical data
   - Extracting key features from genres, cast, and production companies
   - Log transformation of budget and revenue

2. **Model Training** 🚀
   - Using **Random Forest Regressor** with 100 estimators
   - Splitting data into training (80%) and testing (20%)

3. **Model Evaluation** 📉
   - **Root Mean Squared Error (RMSE)**: `1058.45`
   - **R² Score**: `0.4662`
   
4. **Making Predictions** 🎬
   - The model predicts revenue based on input features like budget, popularity, and runtime
   - Visualizing actual vs. predicted revenue using Matplotlib

## 📦 Installation & Usage
### 1️⃣ Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### 2️⃣ Run the Model
```bash
python box_office_prediction.py
```

### 3️⃣ Predict Revenue
You can use the trained model to make predictions on new data:
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('movie_random_forest_model.pkl')

# Example input data
input_features = pd.DataFrame([{
    'budget': 15000000,
    'vote_count': 500,
    'popularity': 20.5,
    'runtime': 120,
    'vote_average': 7.5
}])

# Make prediction
prediction = model.predict(input_features)
print(f"Predicted Revenue: {prediction[0]}")
```

## 📊 Results Visualization
The project includes a **scatter plot** comparing actual vs. predicted revenue for better analysis.

![Actual vs Predicted Revenue](path/to/your/plot.png)

## 🚀 Future Enhancements
- Implement a **web app** for interactive predictions
- Improve accuracy using **Deep Learning models**
- Include additional **features** like marketing spend, social media buzz, and competition at release

## 🤝 Contributing
Feel free to fork this repository and contribute! If you find any issues, create a pull request or open an issue. 😊

## 📜 License
This project is **open-source** and available under the **MIT License**.
