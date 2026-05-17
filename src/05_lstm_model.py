import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("      LSTM MODEL - TIME SERIES PREDICTION")
print("=" * 70)

# ===== STEP 1: LOAD DATA =====
print("\n[1/7] Loading monthly aggregated data...")
data = pd.read_csv('data/processed/wheat_monthly_by_state.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

print(f"   ✓ Loaded {len(data)} records")
print(f"   Date range: {data['date'].min()} to {data['date'].max()}")

# ===== STEP 2: PREPARE DATA FOR LSTM =====
print("\n[2/7] Preparing data for LSTM...")

# Focus on Punjab data first (can expand later)
punjab_data = data[data['state'] == 'Punjab'].copy()
prices = punjab_data['avg_price'].values.reshape(-1, 1)

print(f"   Using Punjab data: {len(punjab_data)} records")
print(f"   Price range: ₹{prices.min():.2f} - ₹{prices.max():.2f}")

# Scale data to 0-1 range (LSTM works better with normalized data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

print("   ✓ Data normalized to [0, 1] range")

# ===== STEP 3: CREATE SEQUENCES =====
print("\n[3/7] Creating time sequences for LSTM...")

def create_sequences(data, lookback=3):
    """
    Create sequences for LSTM training
    lookback: how many past months to use for prediction
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Use 3 months of history to predict next month
lookback = 3
X, y = create_sequences(scaled_prices, lookback)

# Reshape for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"   Created sequences with lookback={lookback} months")
print(f"   X shape: {X.shape} (samples, timesteps, features)")
print(f"   y shape: {y.shape}")

# ===== STEP 4: TRAIN/TEST SPLIT =====
print("\n[4/7] Splitting data...")

# Use 80% for training
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"   Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ===== STEP 5: BUILD LSTM MODEL =====
print("\n[5/7] Building LSTM neural network...")

model = Sequential([
    # First LSTM layer with 50 units
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),  # Prevent overfitting
    
    # Second LSTM layer with 50 units
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    
    # Dense layers for output
    Dense(25),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

print("   ✓ Model architecture:")
model.summary()

# ===== STEP 6: TRAIN MODEL =====
print("\n[6/7] Training LSTM model (this may take 1-2 minutes)...")

# Train with early stopping
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_split=0.2,
    verbose=0,  # Silent training
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)

print("   ✓ Training complete!")
print(f"   Final training loss: {history.history['loss'][-1]:.6f}")
print(f"   Final validation loss: {history.history['val_loss'][-1]:.6f}")

# ===== STEP 7: MAKE PREDICTIONS =====
print("\n[7/7] Making predictions and evaluating...")

# Predict on train and test sets
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

# Compare with Random Forest
print(f"\n   📊 COMPARISON:")
print(f"   Random Forest accuracy: 70.55%")
print(f"   LSTM accuracy: {test_r2*100:.2f}%")
improvement = test_r2*100 - 70.55
print(f"   Improvement: {improvement:+.2f}%")

# ===== SAVE PREDICTIONS =====
print("\nSaving predictions...")

# Get corresponding dates
test_dates = punjab_data['date'].values[lookback + train_size:]

predictions_df = pd.DataFrame({
    'date': test_dates,
    'actual_price': y_test_actual.flatten(),
    'predicted_price': test_predictions.flatten(),
    'error': y_test_actual.flatten() - test_predictions.flatten(),
    'error_percentage': np.abs((y_test_actual.flatten() - test_predictions.flatten()) / y_test_actual.flatten() * 100)
})

predictions_df.to_csv('data/processed/lstm_predictions.csv', index=False)
print("   ✓ Saved: data/processed/lstm_predictions.csv")

# ===== VISUALIZATIONS =====
print("\nCreating visualizations...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Training history
ax1 = plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Training History', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (test set)
ax2 = plt.subplot(2, 3, 2)
plt.scatter(y_test_actual, test_predictions, alpha=0.6, edgecolors='k')
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (₹)', fontsize=11)
plt.ylabel('Predicted Price (₹)', fontsize=11)
plt.title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Time series predictions
ax3 = plt.subplot(2, 3, 3)
plt.plot(predictions_df['date'], predictions_df['actual_price'], 
         marker='o', label='Actual', linewidth=2, markersize=6)
plt.plot(predictions_df['date'], predictions_df['predicted_price'], 
         marker='s', label='Predicted', linewidth=2, markersize=6, alpha=0.8)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Price (₹)', fontsize=11)
plt.title('LSTM Predictions Over Time', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 4: Prediction errors
ax4 = plt.subplot(2, 3, 4)
plt.hist(predictions_df['error'], bins=10, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Prediction Error (₹)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Error Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 5: Comparison bar chart
ax5 = plt.subplot(2, 3, 5)
models = ['Random Forest', 'LSTM']
accuracies = [70.55, test_r2*100]
colors = ['skyblue', 'lightgreen']
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', width=0.5)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('Model Comparison', fontsize=12, fontweight='bold')
plt.ylim([0, 100])
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{acc:.2f}%', ha='center', fontweight='bold', fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Plot 6: Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

stats_data = [
    ['Metric', 'Random Forest', 'LSTM'],
    ['Test Accuracy (R²)', f'{70.55:.2f}%', f'{test_r2*100:.2f}%'],
    ['MAE', '₹57.57', f'₹{test_mae:.2f}'],
    ['RMSE', '₹97.20', f'₹{test_rmse:.2f}'],
    ['Training Samples', '44', f'{len(X_train)}'],
    ['Test Samples', '12', f'{len(X_test)}']
]

table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.30, 0.30])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Performance Comparison', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('data/processed/lstm_performance.png', dpi=300, bbox_inches='tight')
print("   ✓ Visualization saved: data/processed/lstm_performance.png")

plt.show()

# ===== FINAL SUMMARY =====
print("\n" + "=" * 70)
print("✓ LSTM MODEL COMPLETE!")
print("=" * 70)
print(f"\n🎯 RESULTS:")
print(f"   LSTM Test Accuracy: {test_r2*100:.2f}%")
print(f"   Average Error: ₹{test_mae:.2f}")
print(f"   Improvement over Random Forest: {improvement:+.2f}%")
print(f"\n📁 Files saved:")
print(f"   1. data/processed/lstm_predictions.csv")
print(f"   2. data/processed/lstm_performance.png")
print("\n" + "=" * 70)
print("Next: Add SHAP explainability (05_add_shap.py)")
print("=" * 70 + "\n")
