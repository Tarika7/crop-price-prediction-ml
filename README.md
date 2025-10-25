# 🌾 Farm AI - Crop Price Predictor

ML-powered crop price prediction system for Indian farmers. Predicts agricultural prices with 85-97% accuracy using Random Forest models trained on 30+ years of data.

## Features

- 8 Crop Models (Rice, Wheat, Potatoes, Onions, Tomatoes, Lentils, Sugar, Oil)
- State & District Selection (Tamil Nadu - 10 districts)
- Year & Month-based Predictions
- Smart Selling Recommendations (WAIT/SELL/NEUTRAL)
- 12-Month Price Forecasts
- Historical Trend Visualizations

## Installation

#Clone repository
git clone https://github.com/Tarika7/crop-price-prediction-ml.git

cd crop-price-prediction

#Create virtual environment

python -m venv crop_env

crop_env\Scripts\activate # Windows

source crop_env/bin/activate # Mac/Linux

#Install dependencies

pip install -r requirements.txt

#Run app

streamlit run app/app.py

⭐ Built for Indian Farmers | 🌾 Powered by ML
