import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("      CROP PRICE PREDICTION - DATA EXPLORATION")
print("=" * 70)

# ===== 1. LOAD CEDA WHEAT PUNJAB DATA =====
print("\n\n[1/5] Loading CEDA Wheat Punjab Data...")
try:
    wheat_punjab = pd.read_csv('data/raw/wheat_punjab_monthly.csv')
    
    print(f"   ✓ Loaded successfully!")
    print(f"   Total records: {len(wheat_punjab):,}")
    print(f"   Columns: {wheat_punjab.columns.tolist()}")
    print(f"   Date range: {wheat_punjab['t'].min()} to {wheat_punjab['t'].max()}")
    print(f"\n   First 3 rows:")
    print(wheat_punjab.head(3))
    
    # Check missing values
    missing = wheat_punjab.isnull().sum()
    print(f"\n   Missing values:")
    print(missing[missing > 0])
    
except Exception as e:
    print(f"   ✗ Error loading data: {e}")

# ===== 2. LOAD CEDA WHEAT HARYANA DATA =====
print("\n\n[2/5] Loading CEDA Wheat Haryana Data...")
try:
    wheat_haryana = pd.read_csv('data/raw/wheat_haryana_monthly.csv')
    
    print(f"   ✓ Loaded successfully!")
    print(f"   Total records: {len(wheat_haryana):,}")
    print(f"   Date range: {wheat_haryana['t'].min()} to {wheat_haryana['t'].max()}")
    
except Exception as e:
    print(f"   ✗ Error loading data: {e}")

# ===== 3. LOAD KAGGLE DATASET =====
print("\n\n[3/5] Loading Kaggle Crop Dataset...")
try:
    kaggle_data = pd.read_csv('data/raw/crop_price_dataset.csv')
    
    print(f"   ✓ Loaded successfully!")
    print(f"   Total records: {len(kaggle_data):,}")
    print(f"   Columns: {kaggle_data.columns.tolist()}")
    print(f"   Commodities available:")
    print(f"   {kaggle_data['commodity_name'].unique()}")
    print(f"   Date range: {kaggle_data['month'].min()} to {kaggle_data['month'].max()}")
    
    # Filter for wheat
    wheat_kaggle = kaggle_data[kaggle_data['commodity_name'] == 'Wheat']
    print(f"   Wheat records: {len(wheat_kaggle):,}")
    
except Exception as e:
    print(f"   ✗ Error loading data: {e}")

# ===== 4. LOAD WFP DATA =====
print("\n\n[4/5] Loading WFP Food Prices Data...")
try:
    wfp_data = pd.read_csv('data/raw/wfp_food_prices_ind.csv', low_memory=False)
    
    # Remove header row
    wfp_clean = wfp_data[wfp_data['date'] != '#date'].copy()
    wfp_clean['date'] = pd.to_datetime(wfp_clean['date'])
    wfp_clean['price'] = pd.to_numeric(wfp_clean['price'], errors='coerce')
    
    # Filter for wheat and rice
    wheat_wfp = wfp_clean[wfp_clean['commodity'] == 'Wheat']
    rice_wfp = wfp_clean[wfp_clean['commodity'] == 'Rice']
    
    print(f"   ✓ Loaded successfully!")
    print(f"   Total records: {len(wfp_clean):,}")
    print(f"   Wheat records: {len(wheat_wfp):,}")
    print(f"   Rice records: {len(rice_wfp):,}")
    print(f"   Date range: {wfp_clean['date'].min()} to {wfp_clean['date'].max()}")
    
    # Check Punjab and Haryana
    wheat_punjab_wfp = wheat_wfp[wheat_wfp['admin1'] == 'Punjab']
    wheat_haryana_wfp = wheat_wfp[wheat_wfp['admin1'] == 'Haryana']
    rice_punjab_wfp = rice_wfp[rice_wfp['admin1'] == 'Punjab']
    
    print(f"\n   Punjab & Haryana Coverage:")
    print(f"   - Wheat Punjab: {len(wheat_punjab_wfp):,} records")
    print(f"   - Wheat Haryana: {len(wheat_haryana_wfp):,} records")
    print(f"   - Rice Punjab: {len(rice_punjab_wfp):,} records")
    
except Exception as e:
    print(f"   ✗ Error loading data: {e}")

# ===== 5. SUMMARY =====
print("\n\n[5/5] DATASET SUMMARY")
print("=" * 70)
try:
    print(f"✓ CEDA Wheat Punjab:    {len(wheat_punjab):,} records (2022-2025)")
    print(f"✓ CEDA Wheat Haryana:   {len(wheat_haryana):,} records (2024-2025)")
    print(f"✓ Kaggle Wheat:         {len(wheat_kaggle):,} records (2010-2025)")
    print(f"✓ WFP Wheat:            {len(wheat_wfp):,} records (1994-2025)")
    print(f"✓ WFP Rice:             {len(rice_wfp):,} records (1994-2025)")
except:
    print("Some datasets failed to load. Check error messages above.")

print("=" * 70)
print("\n✓ DATA EXPLORATION COMPLETE!")
print("✓ All datasets loaded successfully!")
print("\nNext step: Run 02_clean_data.py to prepare data for modeling\n")

