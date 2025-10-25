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
Crop_Price_Prediction folder should look like:
Crop_Price_Prediction/
├── app/
│   ├── app.py                    
│   ├── ml_backend.py             
│   └── ui_components.py          
│
├── models/
│   ├── tn_rice_model.pkl         
│   ├── tn_wheat_model.pkl        
│   ├── tn_lentils_moong_model.pkl
│   ├── tn_potatoes_model.pkl     
│   ├── tn_onions_model.pkl       
│   ├── tn_tomatoes_model.pkl     
│   ├── tn_sugar_model.pkl        
│   └── tn_oil_groundnut_model.pkl
│
├── src/
│   ├── 04_baseline_model.py      
│   ├── 07_tamilnadu_analysis.py  
│   └── 10_tn_multicrop.py        
│                 
└── requirements.txt              

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
