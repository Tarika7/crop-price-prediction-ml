import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("      TAMIL NADU CROP PRICE PREDICTION")
print("=" * 70)

# Load WFP data
print("\n[1/4] Loading data for Tamil Nadu...")
wfp_data = pd.read_csv('data/raw/wfp_food_prices_ind.csv', low_memory=False)
wfp_clean = wfp_data[wfp_data['date'] != '#date'].copy()
wfp_clean['date'] = pd.to_datetime(wfp_clean['date'])
wfp_clean['price'] = pd.to_numeric(wfp_clean['price'], errors='coerce')

# Filter for Tamil Nadu
tn_data = wfp_clean[wfp_clean['admin1'] == 'Tamil Nadu'].copy()

print(f"   ✓ Found {len(tn_data)} records for Tamil Nadu")
print(f"   Date range: {tn_data['date'].min()} to {tn_data['date'].max()}")
print(f"   Commodities available: {tn_data['commodity'].unique()}")

# Focus on Rice (major crop in TN)
rice_tn = tn_data[tn_data['commodity'] == 'Rice'].copy()
print(f"\n   Rice records: {len(rice_tn)}")
print(f"   Districts covered: {rice_tn['admin2'].nunique()}")

# Aggregate monthly
rice_tn['year_month'] = rice_tn['date'].dt.to_period('M')
monthly_rice = rice_tn.groupby('year_month').agg({
    'price': ['mean', 'min', 'max', 'std', 'count']
}).reset_index()

monthly_rice.columns = ['year_month', 'avg_price', 'min_price', 'max_price', 'std_price', 'count']
monthly_rice['date'] = monthly_rice['year_month'].dt.to_timestamp()
monthly_rice = monthly_rice.sort_values('date')

print(f"   Monthly aggregated records: {len(monthly_rice)}")

# Feature engineering
print("\n[2/4] Creating features for Tamil Nadu rice prices...")
monthly_rice['year'] = monthly_rice['date'].dt.year
monthly_rice['month'] = monthly_rice['date'].dt.month
monthly_rice['quarter'] = monthly_rice['date'].dt.quarter
monthly_rice['price_lag_1'] = monthly_rice['avg_price'].shift(1)
monthly_rice['price_lag_2'] = monthly_rice['avg_price'].shift(2)
monthly_rice['price_rolling_3'] = monthly_rice['avg_price'].rolling(3).mean()

data_clean = monthly_rice.dropna()

print(f"   ✓ {len(data_clean)} samples ready for modeling")

# Prepare features
feature_cols = ['year', 'month', 'quarter', 'price_lag_1', 'price_lag_2', 
                'price_rolling_3', 'min_price', 'max_price', 'std_price']
X = data_clean[feature_cols]
y = data_clean['avg_price']

# Train model
print("\n[3/4] Training Random Forest for Tamil Nadu rice...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"   ✓ Model trained!")
print(f"   Test Accuracy: {r2*100:.2f}%")
print(f"   RMSE: ₹{rmse:.2f}")

# Visualization
print("\n[4/4] Creating Tamil Nadu-specific visualization...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price trend
axes[0, 0].plot(monthly_rice['date'], monthly_rice['avg_price'], 
                marker='o', linewidth=2, color='darkgreen')
axes[0, 0].set_title('Tamil Nadu Rice Price Trends (1994-2025)', 
                     fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Price (₹/kg)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Seasonal pattern
seasonal = data_clean.groupby('month')['avg_price'].mean()
axes[0, 1].bar(range(1, 13), seasonal.values, color='coral', edgecolor='black')
axes[0, 1].set_title('Seasonal Price Pattern - Rice (TN)', 
                     fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Average Price (₹/kg)')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Actual vs Predicted
axes[1, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
axes[1, 0].set_title('Model Predictions - TN Rice', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Actual Price (₹/kg)')
axes[1, 0].set_ylabel('Predicted Price (₹/kg)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'],
                color='skyblue', edgecolor='black')
axes[1, 1].set_title('Feature Importance - TN Rice Model', 
                     fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('data/processed/tamilnadu_rice_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: data/processed/tamilnadu_rice_analysis.png")

plt.show()

# Summary
print("\n" + "=" * 70)
print("TAMIL NADU RICE PRICE PREDICTION - RESULTS")
print("=" * 70)
print(f"  Dataset: {len(data_clean)} monthly records (1994-2025)")
print(f"  Commodity: Rice (major Tamil Nadu crop)")
print(f"  Model Accuracy: {r2*100:.2f}%")
print(f"  Average Price: ₹{monthly_rice['avg_price'].mean():.2f}/kg")
print(f"  Price Range: ₹{monthly_rice['avg_price'].min():.2f} - ₹{monthly_rice['avg_price'].max():.2f}")
print("=" * 70)
print("\n✓ Tamil Nadu-specific model ready!")
print("This provides locally-relevant insights for Tamil Nadu farmers.")
print("=" * 70 + "\n")
# Save the trained model
import pickle
import os

os.makedirs('models', exist_ok=True)

with open('models/tn_rice_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✓ Model saved: models/tn_rice_model.pkl")
