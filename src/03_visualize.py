import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("      DATA VISUALIZATION")
print("=" * 70)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load monthly aggregated data
print("\nLoading monthly data...")
monthly_data = pd.read_csv('data/processed/wheat_monthly_by_state.csv')
monthly_data['date'] = pd.to_datetime(monthly_data['date'])

print(f"✓ Loaded {len(monthly_data)} monthly records")
print(f"  Date range: {monthly_data['date'].min().strftime('%Y-%m')} to {monthly_data['date'].max().strftime('%Y-%m')}")
print(f"  States: {monthly_data['state'].unique()}")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# ===== PLOT 1: Price Trends Over Time =====
print("\nCreating Plot 1: Price trends over time...")
ax1 = plt.subplot(3, 2, 1)
for state in monthly_data['state'].unique():
    state_data = monthly_data[monthly_data['state'] == state]
    plt.plot(state_data['date'], state_data['avg_price'], 
             marker='o', label=state, linewidth=2.5, markersize=6)

plt.title('Wheat Price Trends: Punjab vs Haryana', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Average Modal Price (₹/quintal)', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# ===== PLOT 2: Price Distribution by State =====
print("Creating Plot 2: Price distribution...")
ax2 = plt.subplot(3, 2, 2)
monthly_data.boxplot(column='avg_price', by='state', ax=ax2, patch_artist=True)
plt.suptitle('')  # Remove default title
plt.title('Price Distribution by State', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('State', fontsize=11)
plt.ylabel('Average Price (₹/quintal)', fontsize=11)
plt.grid(True, alpha=0.3)

# ===== PLOT 3: Monthly Price Change =====
print("Creating Plot 3: Monthly price changes...")
ax3 = plt.subplot(3, 2, 3)
for state in monthly_data['state'].unique():
    state_data = monthly_data[monthly_data['state'] == state].copy()
    state_data['price_change'] = state_data['avg_price'].diff()
    plt.plot(state_data['date'], state_data['price_change'], 
             marker='s', label=state, linewidth=2, markersize=5, alpha=0.7)

plt.title('Monthly Price Changes', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Price Change (₹)', fontsize=11)
plt.legend(fontsize=10)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# ===== PLOT 4: Price Volatility (Standard Deviation) =====
print("Creating Plot 4: Price volatility...")
ax4 = plt.subplot(3, 2, 4)
for state in monthly_data['state'].unique():
    state_data = monthly_data[monthly_data['state'] == state]
    plt.plot(state_data['date'], state_data['std_price'], 
             marker='^', label=state, linewidth=2, markersize=5)

plt.title('Price Volatility (Standard Deviation)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Standard Deviation (₹)', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# ===== PLOT 5: Average Min vs Max Prices =====
print("Creating Plot 5: Min vs Max prices...")
ax5 = plt.subplot(3, 2, 5)
punjab_data = monthly_data[monthly_data['state'] == 'Punjab']
if len(punjab_data) > 0:
    plt.plot(punjab_data['date'], punjab_data['avg_min'], 
             marker='v', label='Punjab Min', linewidth=2, color='green', alpha=0.7)
    plt.plot(punjab_data['date'], punjab_data['avg_max'], 
             marker='^', label='Punjab Max', linewidth=2, color='darkgreen', alpha=0.7)

haryana_data = monthly_data[monthly_data['state'] == 'Haryana']
if len(haryana_data) > 0:
    plt.plot(haryana_data['date'], haryana_data['avg_min'], 
             marker='v', label='Haryana Min', linewidth=2, color='orange', alpha=0.7)
    plt.plot(haryana_data['date'], haryana_data['avg_max'], 
             marker='^', label='Haryana Max', linewidth=2, color='red', alpha=0.7)

plt.title('Min vs Max Price Ranges', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Price (₹/quintal)', fontsize=11)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# ===== PLOT 6: Summary Statistics Table =====
print("Creating Plot 6: Summary statistics...")
ax6 = plt.subplot(3, 2, 6)
ax6.axis('tight')
ax6.axis('off')

# Calculate statistics
stats_data = []
for state in monthly_data['state'].unique():
    state_df = monthly_data[monthly_data['state'] == state]
    stats_data.append([
        state,
        f"₹{state_df['avg_price'].mean():.2f}",
        f"₹{state_df['avg_price'].min():.2f}",
        f"₹{state_df['avg_price'].max():.2f}",
        f"₹{state_df['std_price'].mean():.2f}",
        f"{len(state_df)}"
    ])

table = ax6.table(cellText=stats_data,
                  colLabels=['State', 'Avg Price', 'Min Price', 'Max Price', 'Avg Volatility', 'Months'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.15, 0.15, 0.15, 0.15, 0.20, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style the header
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Summary Statistics', fontsize=14, fontweight='bold', pad=15)

# Adjust layout
plt.tight_layout()

# Save figure
output_file = 'data/processed/wheat_price_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output_file}")

# Show the plot
plt.show()

print("\n" + "=" * 70)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 70)
print("\nGenerated visualization includes:")
print("  1. Price trends over time (Punjab vs Haryana)")
print("  2. Price distribution boxplots")
print("  3. Monthly price changes")
print("  4. Price volatility (standard deviation)")
print("  5. Min vs Max price ranges")
print("  6. Summary statistics table")
print("\nImage saved in: data/processed/wheat_price_analysis.png")
print("Use this for your presentation slides!")
print("\n" + "=" * 70)
print("Next step: Run 04_baseline_model.py to build your first ML model!")
print("=" * 70 + "\n")

