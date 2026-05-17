import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("      LSTM MODEL - TAMIL NADU RICE PRICE PREDICTION")
print("=" * 70)

# ===== STEP 1: LOAD TAMIL NADU RICE DATA =====
print("\n[1/7] Loading Tamil Nadu rice data from WFP...")
wfp_data = pd.read_csv('data/raw/wfp_food_prices_ind.csv', low_memory=False)
wfp_clean = wfp_data[wfp_data['date'] != '#date'].copy()
wfp_clean['date'] = pd.to_datetime(wfp_clean['date'])
wfp_clean['price'] = pd.to_numeric(wfp_clean['price'], errors='coerce')

# Filter for Tamil Nadu Rice
tn_rice = wfp_clean[
    (wfp_clean['commodity'] == 'Rice') & 
    (wfp_clean['admin1'] == 'Tamil Nadu')
].copy()

# Aggregate monthly
tn_rice['year_month'] = tn_rice['date'].dt.to_period('M')
monthly_prices = tn_rice.groupby('year_month')['price'].mean().reset_index()
monthly_prices['date'] = monthly_prices['year_month'].dt.to_timestamp()
monthly_prices = monthly_prices.sort_values('date')

print(f"   ✓ Loaded {len(monthly_prices)} monthly records")
print(f"   Date range: {monthly_prices['date'].min()} to {monthly_prices['date'].max()}")
print(f"   Price range: ₹{monthly_prices['price'].min():.2f} - ₹{monthly_prices['price'].max():.2f}")

# ===== STEP 2: PREPARE DATA FOR LSTM =====
print("\n[2/7] Preparing data for LSTM...")

prices = monthly_prices['price'].values.reshape(-1, 1)

# Scale data to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

print(f"   ✓ Data normalized to [0, 1] range")

# ===== STEP 3: CREATE SEQUENCES =====
print("\n[3/7] Creating time sequences...")

def create_sequences(data, lookback=12):
    """Create sequences with 12-month lookback"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Use 12 months of history to predict next month
lookback = 12
X, y = create_sequences(scaled_prices, lookback)

# Reshape for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"   ✓ Created {len(X)} sequences with {lookback}-month lookback")
print(f"   X shape: {X.shape}")

# ===== STEP 4: TRAIN/TEST SPLIT =====
print("\n[4/7] Splitting data into train and test sets...")

# Use 80% for training
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"   Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ===== STEP 5: BUILD LSTM MODEL =====
print("\n[5/7] Building LSTM neural network...")

model = Sequential([
    # First LSTM layer
    LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    
    # Third LSTM layer
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Dense layers
    Dense(16),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print("   ✓ Model architecture:")
print(f"      - Layer 1: LSTM (64 units)")
print(f"      - Layer 2: LSTM (64 units)")
print(f"      - Layer 3: LSTM (32 units)")
print(f"      - Dense layers: 16 → 1")

# ===== STEP 6: TRAIN MODEL =====
print("\n[6/7] Training LSTM model...")
print("   (This may take 2-3 minutes with more data...)")

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1,  # Show progress
    callbacks=[early_stop]
)

print("\n   ✓ Training complete!")
print(f"   Final training loss: {history.history['loss'][-1]:.6f}")
print(f"   Final validation loss: {history.history['val_loss'][-1]:.6f}")

# ===== STEP 7: MAKE PREDICTIONS AND EVALUATE =====
print("\n[7/7] Making predictions and evaluating...")

# Predictions
train_predictions = model.predict(X_train, verbose=0)
test_predictions = model.predict(X_test, verbose=0)

# Inverse transform to get actual prices
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
train_mae = mean_absolute_error(y_train_actual, train_predictions)
train_r2 = r2_score(y_train_actual, train_predictions)

test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
test_mae = mean_absolute_error(y_test_actual, test_predictions)
test_r2 = r2_score(y_test_actual, test_predictions)

print("\n   TRAINING SET PERFORMANCE:")
print(f"   - RMSE: ₹{train_rmse:.2f}")
print(f"   - MAE: ₹{train_mae:.2f}")
print(f"   - R² Score: {train_r2:.4f} ({train_r2*100:.2f}%)")

print("\n   TEST SET PERFORMANCE:")
print(f"   - RMSE: ₹{test_rmse:.2f}")
print(f"   - MAE: ₹{test_mae:.2f}")
print(f"   - R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")

# ===== SAVE PREDICTIONS =====
print("\nSaving predictions...")

# Get corresponding dates
test_dates = monthly_prices['date'].values[lookback + train_size:]

predictions_df = pd.DataFrame({
    'date': test_dates,
    'actual_price': y_test_actual.flatten(),
    'predicted_price': test_predictions.flatten(),
    'error': y_test_actual.flatten() - test_predictions.flatten(),
    'error_percentage': np.abs((y_test_actual.flatten() - test_predictions.flatten()) / y_test_actual.flatten() * 100)
})

predictions_df.to_csv('data/processed/lstm_tamilnadu_predictions.csv', index=False)
print("   ✓ Saved: data/processed/lstm_tamilnadu_predictions.csv")

# ===== VISUALIZATION =====
print("\nCreating visualization...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Training history
ax1 = plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('LSTM Training History - TN Rice', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
ax2 = plt.subplot(2, 3, 2)
plt.scatter(y_test_actual, test_predictions, alpha=0.6, edgecolors='k', color='coral')
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (₹/kg)', fontsize=11)
plt.ylabel('Predicted Price (₹/kg)', fontsize=11)
plt.title('Actual vs Predicted - LSTM', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Time series predictions
ax3 = plt.subplot(2, 3, 3)
plt.plot(predictions_df['date'], predictions_df['actual_price'], 
         marker='o', label='Actual', linewidth=2.5, markersize=6, color='darkgreen')
plt.plot(predictions_df['date'], predictions_df['predicted_price'], 
         marker='s', label='LSTM Predicted', linewidth=2.5, markersize=6, alpha=0.8, color='orange')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Price (₹/kg)', fontsize=11)
plt.title('LSTM Predictions Over Time - TN Rice', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 4: Prediction errors
ax4 = plt.subplot(2, 3, 4)
plt.hist(predictions_df['error'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Prediction Error (₹/kg)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Error Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 5: Model Comparison
ax5 = plt.subplot(2, 3, 5)
models = ['RF\nPunjab Wheat', 'LSTM\nPunjab Wheat', 'RF\nTN Rice', 'LSTM\nTN Rice']
accuracies = [70.55, -239.61, 92.70, test_r2*100]
colors = ['lightgreen' if acc > 0 else 'lightcoral' for acc in accuracies]
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', width=0.6)

plt.ylabel('Accuracy (R² %)', fontsize=11)
plt.title('Model Comparison Across Regions', fontsize=12, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, 
             height + (5 if height > 0 else -10),
             f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=10)

# Plot 6: Summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

comparison_data = [
    ['Model', 'Region/Crop', 'Accuracy', 'MAE'],
    ['Random Forest', 'Punjab Wheat', '70.55%', '₹57.57'],
    ['LSTM', 'Punjab Wheat', '-239%', 'N/A'],
    ['Random Forest', 'TN Rice', '92.70%', '₹74.36'],
    ['LSTM', 'TN Rice', f'{test_r2*100:.2f}%', f'₹{test_mae:.2f}']
]

table = ax6.table(cellText=comparison_data, cellLoc='center', loc='center',
                  colWidths=[0.30, 0.30, 0.20, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best results
if test_r2 > 0.90:
    for i in range(4):
        table[(4, i)].set_facecolor('#FFE082')  # Highlight LSTM TN Rice row

plt.title('Comprehensive Model Comparison', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('data/processed/lstm_tamilnadu_performance.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualization saved: data/processed/lstm_tamilnadu_performance.png")

plt.show()

# ===== FINAL SUMMARY =====
print("\n" + "=" * 70)
print("✓ LSTM MODEL FOR TAMIL NADU RICE - COMPLETE!")
print("=" * 70)
print(f"\n📊 COMPREHENSIVE RESULTS:")
print(f"\n   Punjab/Haryana Wheat:")
print(f"   • Random Forest: 70.55%")
print(f"   • LSTM: -239% (insufficient data)")
print(f"\n   Tamil Nadu Rice:")
print(f"   • Random Forest: 92.70%")
print(f"   • LSTM: {test_r2*100:.2f}%")
print(f"   • Improvement: {(test_r2*100 - 92.70):+.2f}%")

print(f"\n🎯 KEY INSIGHT:")
if test_r2*100 > 92.70:
    print(f"   LSTM OUTPERFORMS Random Forest on Tamil Nadu rice!")
    print(f"   This demonstrates that LSTM excels with more historical data.")
elif test_r2*100 > 85:
    print(f"   LSTM performs competitively with Random Forest!")
    print(f"   With sufficient data, LSTM captures temporal patterns well.")
else:
    print(f"   Random Forest remains superior even with more data.")
    print(f"   Traditional ML may be more robust for agricultural data.")

print("\n" + "=" * 70)
print("🎉 PROJECT COMPLETE! You now have:")
print("   ✓ Multi-regional analysis (Punjab, Haryana, Tamil Nadu)")
print("   ✓ Multi-crop models (Wheat, Rice)")
print("   ✓ RF + LSTM comparison with valid conclusions")
print("   ✓ SHAP explainability")
print("   ✓ Publication-ready results!")
print("=" * 70 + "\n")
