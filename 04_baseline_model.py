import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("      BASELINE MACHINE LEARNING MODEL - RANDOM FOREST")
print("=" * 70)

# ===== STEP 1: LOAD AND PREPARE DATA =====
print("\n[1/6] Loading monthly aggregated data...")
data = pd.read_csv('data/processed/wheat_monthly_by_state.csv')
data['date'] = pd.to_datetime(data['date'])

print(f"   ✓ Loaded {len(data)} records")
print(f"   Date range: {data['date'].min()} to {data['date'].max()}")

# ===== STEP 2: FEATURE ENGINEERING =====
print("\n[2/6] Creating features for machine learning...")

# Extract time-based features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['quarter'] = data['date'].dt.quarter

# Create state encoding (Punjab=0, Haryana=1)
data['state_encoded'] = data['state'].map({'Punjab': 0, 'Haryana': 1})

# Create lag features (previous month's price)
data = data.sort_values(['state', 'date'])
data['price_lag_1'] = data.groupby('state')['avg_price'].shift(1)
data['price_lag_2'] = data.groupby('state')['avg_price'].shift(2)

# Create rolling average (3-month moving average)
data['price_rolling_3'] = data.groupby('state')['avg_price'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Remove rows with NaN values (from lag features)
data_clean = data.dropna().copy()

print(f"   ✓ Created features: year, month, quarter, state_encoded")
print(f"   ✓ Created lag features: price_lag_1, price_lag_2")
print(f"   ✓ Created rolling average: price_rolling_3")
print(f"   Records after feature engineering: {len(data_clean)}")

# ===== STEP 3: PREPARE FEATURES AND TARGET =====
print("\n[3/6] Preparing training data...")

# Select features for training
feature_columns = [
    'year', 'month', 'quarter', 'state_encoded',
    'price_lag_1', 'price_lag_2', 'price_rolling_3',
    'avg_min', 'avg_max', 'std_price'
]

X = data_clean[feature_columns]
y = data_clean['avg_price']

print(f"   Features used: {feature_columns}")
print(f"   Total samples: {len(X)}")
print(f"   Target variable: avg_price")

# ===== STEP 4: SPLIT DATA =====
print("\n[4/6] Splitting data into train and test sets...")

# Use 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"   Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ===== STEP 5: TRAIN RANDOM FOREST MODEL =====
print("\n[5/6] Training Random Forest model...")

# Create and train the model
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    min_samples_split=2,   # Minimum samples to split a node
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)

rf_model.fit(X_train, y_train)
print("   ✓ Model trained successfully!")

# Make predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# ===== STEP 6: EVALUATE MODEL =====
print("\n[6/6] Evaluating model performance...")

# Calculate metrics for training set
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for test set
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n   TRAINING SET PERFORMANCE:")
print(f"   - RMSE (Root Mean Squared Error): ₹{train_rmse:.2f}")
print(f"   - MAE (Mean Absolute Error): ₹{train_mae:.2f}")
print(f"   - R² Score (Accuracy): {train_r2:.4f} ({train_r2*100:.2f}%)")

print("\n   TEST SET PERFORMANCE:")
print(f"   - RMSE (Root Mean Squared Error): ₹{test_rmse:.2f}")
print(f"   - MAE (Mean Absolute Error): ₹{test_mae:.2f}")
print(f"   - R² Score (Accuracy): {test_r2:.4f} ({test_r2*100:.2f}%)")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   TOP 5 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")

# ===== SAVE RESULTS =====
print("\n[7/7] Saving results...")

# Create predictions dataframe properly
test_indices = X_test.index
test_dates = data_clean.loc[test_indices, 'date'].values
test_states = data_clean.loc[test_indices, 'state'].values

predictions_df = pd.DataFrame({
    'date': test_dates,
    'state': test_states,
    'actual_price': y_test.values,
    'predicted_price': y_test_pred,
    'error': y_test.values - y_test_pred,
    'error_percentage': np.abs((y_test.values - y_test_pred) / y_test.values * 100)
})

predictions_df.to_csv('data/processed/model_predictions.csv', index=False)
print("   ✓ Predictions saved: data/processed/model_predictions.csv")

# ===== VISUALIZE PREDICTIONS =====
print("\nCreating prediction visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Price (₹)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Price (₹)', fontsize=11)
axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Prediction errors
axes[0, 1].hist(predictions_df['error'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Prediction Error (₹)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Time series of predictions
axes[1, 0].plot(predictions_df['date'], predictions_df['actual_price'], 
                marker='o', label='Actual', linewidth=2, markersize=6)
axes[1, 0].plot(predictions_df['date'], predictions_df['predicted_price'], 
                marker='s', label='Predicted', linewidth=2, markersize=6, alpha=0.8)
axes[1, 0].set_xlabel('Date', fontsize=11)
axes[1, 0].set_ylabel('Price (₹)', fontsize=11)
axes[1, 0].set_title('Predictions Over Time', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Feature importance
top_features = feature_importance.head(8)
axes[1, 1].barh(top_features['feature'], top_features['importance'], color='coral')
axes[1, 1].set_xlabel('Importance', fontsize=11)
axes[1, 1].set_title('Top Feature Importances', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/model_performance.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualization saved: data/processed/model_performance.png")

plt.show()

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("✓ BASELINE MODEL COMPLETE!")
print("=" * 70)
print(f"\nModel Performance Summary:")
print(f"  - Test Accuracy (R²): {test_r2*100:.2f}%")
print(f"  - Average Error (MAE): ₹{test_mae:.2f}")
print(f"  - RMSE: ₹{test_rmse:.2f}")
print(f"\nFiles saved:")
print(f"  1. data/processed/model_predictions.csv")
print(f"  2. data/processed/model_performance.png")
print("\n" + "=" * 70)
print("🎉 SUCCESS! Your first ML model is working!")
print("=" * 70)
print("\nNext steps:")
print("  1. Review the predictions in model_predictions.csv")
print("  2. Check model_performance.png for visualizations")
print("  3. Tomorrow: Build LSTM model for better accuracy")
print("=" * 70 + "\n")
# Save the trained model
import pickle
import os

os.makedirs('models', exist_ok=True)

with open('models/punjab_wheat_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✓ Model saved: models/punjab_wheat_model.pkl")

