import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("      DATA CLEANING AND PREPARATION")
print("=" * 70)

# ===== STEP 1: CLEAN CEDA WHEAT PUNJAB =====
print("\n[1/4] Cleaning CEDA Wheat Punjab data...")

wheat_punjab = pd.read_csv('data/raw/wheat_punjab_monthly.csv')

# Convert date column
wheat_punjab['date'] = pd.to_datetime(wheat_punjab['t'], errors='coerce')

# Convert price columns to numeric (handle empty strings and non-numeric values)
wheat_punjab['p_modal'] = pd.to_numeric(wheat_punjab['p_modal'], errors='coerce')
wheat_punjab['p_min'] = pd.to_numeric(wheat_punjab['p_min'], errors='coerce')
wheat_punjab['p_max'] = pd.to_numeric(wheat_punjab['p_max'], errors='coerce')

# Remove rows with missing modal price (most important column)
wheat_punjab_clean = wheat_punjab[wheat_punjab['p_modal'].notna()].copy()

# Remove rows with invalid dates
wheat_punjab_clean = wheat_punjab_clean[wheat_punjab_clean['date'].notna()]

# Keep only relevant columns
wheat_punjab_clean = wheat_punjab_clean[[
    'date', 'state_name', 'district_name', 'market_name', 
    'variety', 'p_min', 'p_max', 'p_modal'
]]

# Sort by date
wheat_punjab_clean = wheat_punjab_clean.sort_values('date')

print(f"   Original records: {len(wheat_punjab):,}")
print(f"   After cleaning: {len(wheat_punjab_clean):,}")
print(f"   Removed: {len(wheat_punjab) - len(wheat_punjab_clean):,} records ({((len(wheat_punjab) - len(wheat_punjab_clean))/len(wheat_punjab)*100):.1f}%)")
print(f"   Date range: {wheat_punjab_clean['date'].min()} to {wheat_punjab_clean['date'].max()}")
print(f"   Price range: ₹{wheat_punjab_clean['p_modal'].min():.0f} - ₹{wheat_punjab_clean['p_modal'].max():.0f}")

# Save cleaned data
wheat_punjab_clean.to_csv('data/processed/wheat_punjab_clean.csv', index=False)
print("   ✓ Saved: data/processed/wheat_punjab_clean.csv")

# ===== STEP 2: CLEAN CEDA WHEAT HARYANA =====
print("\n[2/4] Cleaning CEDA Wheat Haryana data...")

wheat_haryana = pd.read_csv('data/raw/wheat_haryana_monthly.csv')

# Same cleaning process
wheat_haryana['date'] = pd.to_datetime(wheat_haryana['t'], errors='coerce')
wheat_haryana['p_modal'] = pd.to_numeric(wheat_haryana['p_modal'], errors='coerce')
wheat_haryana['p_min'] = pd.to_numeric(wheat_haryana['p_min'], errors='coerce')
wheat_haryana['p_max'] = pd.to_numeric(wheat_haryana['p_max'], errors='coerce')

wheat_haryana_clean = wheat_haryana[wheat_haryana['p_modal'].notna()].copy()
wheat_haryana_clean = wheat_haryana_clean[wheat_haryana_clean['date'].notna()]

wheat_haryana_clean = wheat_haryana_clean[[
    'date', 'state_name', 'district_name', 'market_name', 
    'variety', 'p_min', 'p_max', 'p_modal'
]]
wheat_haryana_clean = wheat_haryana_clean.sort_values('date')

print(f"   Original records: {len(wheat_haryana):,}")
print(f"   After cleaning: {len(wheat_haryana_clean):,}")
print(f"   Removed: {len(wheat_haryana) - len(wheat_haryana_clean):,} records")
print(f"   Date range: {wheat_haryana_clean['date'].min()} to {wheat_haryana_clean['date'].max()}")

wheat_haryana_clean.to_csv('data/processed/wheat_haryana_clean.csv', index=False)
print("   ✓ Saved: data/processed/wheat_haryana_clean.csv")

# ===== STEP 3: COMBINE PUNJAB + HARYANA =====
print("\n[3/4] Combining Punjab and Haryana wheat data...")

combined_wheat = pd.concat([wheat_punjab_clean, wheat_haryana_clean], ignore_index=True)
combined_wheat = combined_wheat.sort_values('date')

print(f"   Total combined records: {len(combined_wheat):,}")
print(f"   Date range: {combined_wheat['date'].min()} to {combined_wheat['date'].max()}")
print(f"   Price range: ₹{combined_wheat['p_modal'].min():.0f} - ₹{combined_wheat['p_modal'].max():.0f}")
print(f"   Average modal price: ₹{combined_wheat['p_modal'].mean():.2f}")
print(f"   States: {combined_wheat['state_name'].unique()}")
print(f"   Districts: {combined_wheat['district_name'].nunique()}")
print(f"   Markets: {combined_wheat['market_name'].nunique()}")

combined_wheat.to_csv('data/processed/wheat_combined_clean.csv', index=False)
print("   ✓ Saved: data/processed/wheat_combined_clean.csv")

# ===== STEP 4: CREATE MONTHLY AGGREGATED DATA =====
print("\n[4/4] Creating monthly aggregated dataset...")

# Extract year and month
combined_wheat['year'] = combined_wheat['date'].dt.year
combined_wheat['month'] = combined_wheat['date'].dt.month
combined_wheat['year_month'] = combined_wheat['date'].dt.to_period('M')

# Aggregate by state and month (average prices across all districts/markets)
monthly_state = combined_wheat.groupby(['year_month', 'state_name']).agg({
    'p_modal': ['mean', 'std', 'count'],
    'p_min': 'mean',
    'p_max': 'mean'
}).reset_index()

# Flatten multi-level columns
monthly_state.columns = ['year_month', 'state', 'avg_price', 'std_price', 'record_count', 'avg_min', 'avg_max']

# Convert period back to datetime
monthly_state['date'] = monthly_state['year_month'].dt.to_timestamp()
monthly_state = monthly_state.drop('year_month', axis=1)

# Reorder columns
monthly_state = monthly_state[['date', 'state', 'avg_price', 'avg_min', 'avg_max', 'std_price', 'record_count']]

# Sort by date
monthly_state = monthly_state.sort_values('date')

print(f"   Monthly records created: {len(monthly_state)}")
print(f"   Date range: {monthly_state['date'].min()} to {monthly_state['date'].max()}")
print(f"   States: {monthly_state['state'].unique()}")
print(f"\n   Sample monthly data:")
print(monthly_state.head(10))

monthly_state.to_csv('data/processed/wheat_monthly_by_state.csv', index=False)
print("\n   ✓ Saved: data/processed/wheat_monthly_by_state.csv")

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("✓ DATA CLEANING COMPLETE!")
print("=" * 70)
print("\nCleaned files created in data/processed/:")
print("  1. wheat_punjab_clean.csv      - Punjab detailed data")
print("  2. wheat_haryana_clean.csv     - Haryana detailed data")
print("  3. wheat_combined_clean.csv    - Combined detailed data")
print("  4. wheat_monthly_by_state.csv  - Monthly aggregated (READY FOR ML!)")
print("\n" + "=" * 70)
print("Next step: Run 03_visualize.py to see price trends")
print("=" * 70 + "\n")
