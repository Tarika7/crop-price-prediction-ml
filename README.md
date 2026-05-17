# 🌾 Crop Price Prediction Dashboard

Welcome to the **Crop Price Prediction** project! This is a smart tool designed to help farmers, traders, and agricultural enthusiasts predict future crop prices based on historical market trends. It uses Machine Learning to analyze past data and gives you realistic price forecasts, along with smart recommendations on whether to "Sell Now" or "Wait".

*(And yes, I built this with a little vibe coding ✨)*

## 🚀 What Does It Do?
- **Predicts Future Prices:** Select your state and crop, and the model will tell you the predicted price per quintal.
- **Smart Recommendations:** Based on the predicted price vs current price, it advises you to either hold your stock or sell it immediately.
- **Visualizes Trends:** Shows beautiful charts comparing historical prices and future forecasts.
- **Multi-Region Support:** Covers major agricultural hubs like Tamil Nadu, Punjab, and Haryana.
- **Variety of Crops:** Predicts prices for Rice, Wheat, Potatoes, Onions, Tomatoes, Lentils, and more!

## 🧠 How It Works
Under the hood, this project is powered by **Machine Learning** (Random Forest and Gradient Boosting models). 
We took years of historical agricultural pricing data (from 1994 to 2025) and taught the AI to recognize seasonal patterns, price lags, and market volatility. 

The models are extremely accurate—for example, our Wheat model for Punjab has an average error of just ~Rs. 56 per quintal!

## 🛠️ Tech Stack
- **Frontend:** Streamlit (for the beautiful, interactive dashboard)
- **Backend API:** FastAPI (handles the heavy lifting and model inference)
- **Machine Learning:** Scikit-Learn (Random Forest & Gradient Boosting Regressors)
- **Data Processing:** Pandas & NumPy
- **Containerization:** Docker & Docker Compose
  
### The Project is live at 'https://crop-price-prediction-ml-hdknp7bjtguz7iwbzhhsvy.streamlit.app/'

## 💻 How to Run It Locally

### Option 1: Using Python
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI backend server:
   ```bash
   uvicorn app.api:app --reload
   ```
3. In a new terminal, start the Streamlit dashboard:
   ```bash
   streamlit run app/app.py
   ```

### Option 2: Using Docker (Super Easy!)
If you have Docker installed, you can spin up the entire project with one command:
```bash
docker-compose up --build
```
Once it's running, simply open your browser and go to `http://localhost:8501`.

## 📁 Project Structure
- `app/` - Contains the Streamlit frontend, FastAPI backend, and ML loading logic.
- `src/` - The raw Python scripts used for cleaning data and training the ML models.
- `models/` - The saved `.pkl` model files for each specific crop and state.
- `data/` - Raw and processed historical CSV datasets.

---
*Built to make agriculture a little more predictable.* 🌱

