import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("   COMPREHENSIVE NORTHERN WHEAT BELT ANALYSIS")
print("=" * 70)

# Load data
print("\n[1/3] Loading Northern India wheat belt data...")
wfp_data = pd.read_csv('data/raw/wfp_food_prices_ind.csv', low_memory=False)
wfp_clean = wfp_data[wfp_data['date'] != '#date'].copy()
wfp_clean['date'] = pd.to_datetime(wfp_clean['date'])
wfp_clean['price'] = pd.to_numeric(wfp_clean['price'], errors='coerce')
wfp_clean['price'] = wfp_clean['price'].apply(lambda x: x * 100 if x < 100 else x)

# Northern wheat belt states (top 5 wheat producers)
wheat_states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan']

print("\n   Checking wheat data availability:")
for state in wheat_states:
    state_data = wfp_clean[wfp_clean['admin1'] == state]
    wheat_data = state_data[state_data['commodity'] == 'Wheat']
    print(f"   {state:<20} {len(wheat_data):>6} wheat records")

results = []

print("\n[2/3] Training Random Forest models for wheat across states...")

for state in wheat_states:
    print(f"\n   Processing: {state}")
    
    # Filter wheat data
    state_data = wfp_clean[wfp_clean['admin1'] == state].copy()
    wheat_data = state_data[state_data['commodity'] == 'Wheat'].copy()
    
    if len(wheat_data) < 50:
        print(f"       ⚠ Insufficient data ({len(wheat_data)} records). Skipping...")
        continue
    
    # Monthly aggregation
    wheat_data['year_month'] = wheat_data['date'].dt.to_period('M')
    monthly = wheat_data.groupby('year_month').agg({
        'price': ['mean', 'min', 'max', 'std', 'count']
    }).reset_index()
    
    monthly.columns = ['year_month', 'avg_price', 'min_price', 'max_price', 'std_price', 'count']
    monthly['date'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly.sort_values('date')
    
    print(f"       Monthly records: {len(monthly)}")
    print(f"       Date range: {monthly['date'].min().year}-{monthly['date'].max().year}")
    
    # Feature engineering
    monthly['year'] = monthly['date'].dt.year
    monthly['month'] = monthly['date'].dt.month
    monthly['quarter'] = monthly['date'].dt.quarter
    monthly['price_lag_1'] = monthly['avg_price'].shift(1)
    monthly['price_lag_2'] = monthly['avg_price'].shift(2)
    monthly['price_rolling_3'] = monthly['avg_price'].rolling(3).mean()
    
    data_clean = monthly.dropna()
    
    if len(data_clean) < 30:
        print(f"       ⚠ Insufficient clean data. Skipping...")
        continue
    
    # Train model
    feature_cols = ['month', 'quarter', 'price_lag_1', 'price_lag_2',
                   'price_rolling_3', 'min_price', 'max_price', 'std_price']
    X = data_clean[feature_cols]
    y = data_clean['avg_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    print(f"       ✓ Accuracy: {r2*100:.2f}% | MAE: ₹{mae:.2f}")
    
    results.append({
        'state': state,
        'crop': 'Wheat',
        'accuracy': r2 * 100,
        'mae': mae,
        'rmse': rmse,
        'records': len(monthly),
        'date_range': f"{monthly['date'].min().year}-{monthly['date'].max().year}",
        'avg_price': monthly['avg_price'].mean()
    })

# Create DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('accuracy', ascending=False)

print("\n[3/3] Generating comprehensive wheat belt analysis...")

print("\n" + "=" * 70)
print("NORTHERN WHEAT BELT PREDICTION RESULTS")
print("=" * 70)
print("\n")
print(results_df[['state', 'accuracy', 'mae', 'records']].to_string(index=False))

# Calculate wheat belt average
wheat_avg = results_df['accuracy'].mean()
print(f"\n   Average accuracy across wheat belt: {wheat_avg:.2f}%")
print(f"   States with >70% accuracy: {len(results_df[results_df['accuracy'] > 70])}")

# Save results
results_df.to_csv('data/processed/north_wheat_belt_results.csv', index=False)
print("\n   ✓ Saved: data/processed/north_wheat_belt_results.csv")

# Visualization
print("\nCreating comprehensive visualization...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Accuracy by state
ax1 = plt.subplot(2, 3, 1)
states = results_df['state'].values
accuracies = results_df['accuracy'].values
colors = ['#27ae60' if acc > 70 else '#f39c12' if acc > 50 else '#e74c3c' for acc in accuracies]

bars = ax1.barh(states, accuracies, color=colors, edgecolor='black')
ax1.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Wheat Price Prediction: Northern India Wheat Belt', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.axvline(x=70, color='red', linestyle='--', alpha=0.5, label='70% threshold')

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(acc + 2, i, f'{acc:.1f}%', va='center', fontweight='bold')

# Plot 2: MAE comparison
ax2 = plt.subplot(2, 3, 2)
maes = results_df['mae'].values
ax2.barh(states, maes, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Mean Absolute Error (₹)', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Error by State', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Data coverage
ax3 = plt.subplot(2, 3, 3)
records = results_df['records'].values
ax3.bar(range(len(states)), records, color='skyblue', edgecolor='black')
ax3.set_xticks(range(len(states)))
ax3.set_xticklabels(states, rotation=45, ha='right')
ax3.set_ylabel('Monthly Records', fontsize=11, fontweight='bold')
ax3.set_title('Data Availability by State', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for i, (x, y) in enumerate(zip(range(len(states)), records)):
    ax3.text(x, y + 5, str(y), ha='center', fontweight='bold')

# Plot 4: Map-style comparison (visual representation)
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')

# Create visual wheat belt representation
import matplotlib.patches as mpatches

positions = {
    'Punjab': (0.2, 0.8),
    'Haryana': (0.4, 0.7),
    'Uttar Pradesh': (0.6, 0.6),
    'Madhya Pradesh': (0.4, 0.3),
    'Rajasthan': (0.2, 0.4)
}

for state in results_df['state'].values:
    if state in positions:
        acc = results_df[results_df['state'] == state]['accuracy'].values[0]
        color = '#27ae60' if acc > 70 else '#f39c12' if acc > 50 else '#e74c3c'
        
        pos = positions[state]
        circle = mpatches.Circle(pos, 0.08, color=color, ec='black', linewidth=2)
        ax4.add_patch(circle)
        ax4.text(pos[0], pos[1], f'{acc:.0f}%', ha='center', va='center', 
                fontweight='bold', fontsize=10)
        ax4.text(pos[0], pos[1]-0.12, state, ha='center', fontsize=8)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_title('Wheat Belt Accuracy Map', fontsize=12, fontweight='bold')

# Plot 5: Accuracy distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(accuracies, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
ax5.axvline(wheat_avg, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {wheat_avg:.1f}%')
ax5.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Distribution of Accuracies', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

best_state = results_df.iloc[0]
summary_data = [
    ['Metric', 'Value'],
    ['States Covered', str(len(results_df))],
    ['Average Accuracy', f'{wheat_avg:.2f}%'],
    ['Best State', best_state['state']],
    ['Best Accuracy', f'{best_state["accuracy"]:.2f}%'],
    ['Total Records', f'{results_df["records"].sum():,}'],
    ['States >70%', str(len(results_df[results_df['accuracy'] > 70]))]
]

table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.55, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(2):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax6.set_title('Wheat Belt Summary', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('data/processed/north_wheat_belt_comprehensive.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: data/processed/north_wheat_belt_comprehensive.png")

# plt.show()

print("\n" + "=" * 70)
print("✓ NORTHERN WHEAT BELT ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n📊 COVERAGE:")
print(f"   • {len(results_df)} major wheat-producing states")
print(f"   • Average accuracy: {wheat_avg:.2f}%")
print(f"   • Best performing: {best_state['state']} ({best_state['accuracy']:.2f}%)")
print(f"\n💡 RESEARCH NARRATIVE:")
print(f"   ✓ Tamil Nadu: Crop diversity (9 crops, 82-97%)")
print(f"   ✓ Northern Wheat Belt: Regional specialization ({len(results_df)} states, {wheat_avg:.0f}% avg)")
print(f"   → Perfect symmetric design: Diversity vs Specialization!")
print("=" * 70 + "\n")
