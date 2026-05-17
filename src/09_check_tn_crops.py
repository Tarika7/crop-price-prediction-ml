import pandas as pd

print("=" * 70)
print("      CHECKING AVAILABLE TAMIL NADU CROPS IN WFP DATA")
print("=" * 70)

# Load WFP data
wfp_data = pd.read_csv('data/raw/wfp_food_prices_ind.csv', low_memory=False)
wfp_clean = wfp_data[wfp_data['date'] != '#date'].copy()
wfp_clean['date'] = pd.to_datetime(wfp_clean['date'])

# Filter for Tamil Nadu
tn_data = wfp_clean[wfp_clean['admin1'] == 'Tamil Nadu'].copy()

print(f"\nTotal Tamil Nadu records: {len(tn_data)}")

# Get unique commodities
commodities = tn_data['commodity'].value_counts()

print("\n" + "=" * 70)
print("COMMODITIES AVAILABLE IN TAMIL NADU:")
print("=" * 70)
for commodity, count in commodities.items():
    print(f"  {commodity:<25} {count:>6} records")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

# Recommend crops with sufficient data
recommended = []
for commodity, count in commodities.items():
    if count >= 100:  # At least 100 records for reliable modeling
        recommended.append(commodity)

print(f"\nCrops with sufficient data (≥100 records):")
for crop in recommended:
    print(f"  ✓ {crop}")

print("\n" + "=" * 70)
print("These crops can be added to your multi-crop analysis!")
print("=" * 70 + "\n")
