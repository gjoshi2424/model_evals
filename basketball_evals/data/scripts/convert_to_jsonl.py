import pandas as pd

# Read the CSV file
df = pd.read_csv('../player-season-stats.csv')

print(f"Converting {len(df)} rows to JSONL format...")

# Convert to JSONL (each row is a JSON object on a separate line)
df.to_json('../player-season-stats.jsonl', orient='records', lines=True)

print(f"Successfully saved to player-season-stats.jsonl")
