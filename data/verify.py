import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
print("Shape:", df.shape)
print("Null values:", df.isnull().sum().sum())
print("Unique PCs:", df['pc'].nunique())
print("Vocab size (unique delta_ids):", df['delta_id'].nunique())
print("delta_id min/max:", df['delta_id'].min(), df['delta_id'].max())
print("Top 5 most common deltas:")
print(df['raw_delta'].value_counts().head())
