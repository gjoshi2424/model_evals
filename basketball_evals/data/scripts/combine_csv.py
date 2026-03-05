import pandas as pd

# Read both CSV files
player_totals = pd.read_csv('./player-totals.csv')
players_advanced = pd.read_csv('./players-advanced.csv')

print(f"Player totals shape: {player_totals.shape}")
print(f"Players advanced shape: {players_advanced.shape}")

# Merge the dataframes on Player, Age, and Team
# Use 'inner' to keep only records that exist in BOTH dataframes
combined = pd.merge(
    player_totals, 
    players_advanced, 
    on=['Player', 'Age', 'Team'],
    how='inner',
    suffixes=('', '_dup')
)

# Remove duplicate columns (those with _dup suffix)
duplicate_cols = [col for col in combined.columns if col.endswith('_dup')]
print(f"\nDuplicate columns found: {duplicate_cols}")

# Drop the duplicate columns
combined = combined.drop(columns=duplicate_cols)

print(f"\nCombined shape: {combined.shape}")
print(f"Combined columns: {list(combined.columns)}")

# Save the combined dataframe
combined.to_csv('./player-season-stats.csv', index=False)
print("\nSuccessfully saved to data/player-season-stats.csv")
