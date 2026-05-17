import pandas as pd
import json

print("Loading WFP data...")
wfp_data = pd.read_csv('data/raw/wfp_food_prices_ind.csv', low_memory=False)
wfp_clean = wfp_data[wfp_data['date'] != '#date'].copy()
wfp_clean['date'] = pd.to_datetime(wfp_clean['date'])
wfp_clean['price'] = pd.to_numeric(wfp_clean['price'], errors='coerce')
wfp_clean['price'] = wfp_clean['price'].apply(lambda x: x * 100 if x < 100 else x)

tn_data = wfp_clean[wfp_clean['admin1'] == 'Tamil Nadu'].copy()

target_crops = [
    'Rice', 'Wheat', 'Potatoes', 'Onions', 'Tomatoes', 
    'Lentils (moong)', 'Chickpeas', 'Oil (groundnut)', 'Sugar'
]

stats = {}

for crop in target_crops:
    crop_data = tn_data[tn_data['commodity'] == crop].copy()
    if len(crop_data) == 0:
        continue
    
    crop_data['year_month'] = crop_data['date'].dt.to_period('M')
    monthly = crop_data.groupby('year_month').agg({
        'price': ['mean', 'min', 'max', 'std']
    }).reset_index()
    monthly.columns = ['year_month', 'avg_price', 'min_price', 'max_price', 'std_price']
    
    # Get the last two months of data
    last_two = monthly.tail(2)
    if len(last_two) >= 2:
        lag_1 = last_two.iloc[-1]['avg_price']
        lag_2 = last_two.iloc[-2]['avg_price']
    else:
        lag_1 = last_two.iloc[-1]['avg_price']
        lag_2 = lag_1
        
    # Get overall stats from the last year to represent recent history
    last_year = monthly.tail(12)
    min_price = last_year['min_price'].min()
    max_price = last_year['max_price'].max()
    std_price = last_year['avg_price'].std()
    
    if pd.isna(std_price):
        std_price = 0.0
        
    stats[crop] = {
        'lag_1': float(lag_1),
        'lag_2': float(lag_2),
        'min_price': float(min_price) if not pd.isna(min_price) else float(lag_1 * 0.8),
        'max_price': float(max_price) if not pd.isna(max_price) else float(lag_1 * 1.2),
        'std_price': float(std_price) if not pd.isna(std_price) else float(lag_1 * 0.1)
    }

with open('app/crop_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Saved app/crop_stats.json")
