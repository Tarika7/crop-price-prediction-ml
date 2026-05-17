import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load your processed dataset
df = pd.read_csv("data/processed/wheat_monthly_by_state.csv")
df['date'] = pd.to_datetime(df['date'])

# Extract time-based features
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

# Create state encoding
df['state_encoded'] = df['state'].map({'Punjab': 0, 'Haryana': 1})

# Create lag features
df = df.sort_values(['state', 'date'])
df['price_lag_1'] = df.groupby('state')['avg_price'].shift(1)
df['price_lag_2'] = df.groupby('state')['avg_price'].shift(2)

# Create rolling average
df['price_rolling_3'] = df.groupby('state')['avg_price'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

data_clean = df.dropna().copy()

# 2. Split into Train and Test based on Time
# We use the last 20% of the dataset as the hidden test set
split_idx = int(len(data_clean) * 0.8)
test_df = data_clean.iloc[split_idx:].copy()

# 3. Load your saved model
with open('models/punjab_wheat_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 4. Preprocess the Test Data exactly like training
feature_columns = [
    'month', 'quarter', 'state_encoded',
    'price_lag_1', 'price_lag_2', 'price_rolling_3',
    'avg_min', 'avg_max', 'std_price'
]

X_test = test_df[feature_columns]
y_actual = test_df['avg_price']

# 5. Generate Predictions
y_pred = model.predict(X_test)

# 6. Calculate Statistical Metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("="*40)
print("       MODEL METRICS REPORT       ")
print("="*40)
print(f"Mean Absolute Error (MAE): Rs.{mae:.2f}")
print(f"Root Mean Squared Error (RMSE): Rs.{rmse:.2f}")
print(f"True R-squared Score (R2): {r2*100:.1f}%")
print("="*40)

# 7. Plot Actual vs Predicted Prices for your presentation
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6, color='purple')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Crop Prices (Validation Set)')
plt.xlabel('Actual Price (Rs./quintal)')
plt.ylabel('Predicted Price (Rs./quintal)')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
print("Saved plot to actual_vs_predicted.png")
