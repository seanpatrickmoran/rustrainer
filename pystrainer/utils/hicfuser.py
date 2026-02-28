import sqlite3
import pandas as pd

# 1. Load both databases into DataFrames
db1 = sqlite3.connect('/Users/sean/Documents/Master/2025/process_hic_massive_data/fused_database.db')
db2 = sqlite3.connect('/Users/sean/Documents/Master/2025/process_hic_massive_data/SHMC_3.db')

# Replace 'users' with your actual table name
df1 = pd.read_sql_query("SELECT * FROM imag_with_seqs", db1)
df2 = pd.read_sql_query("SELECT * FROM imag_with_seqs", db2)

# 2. Clean: Drop the old ID columns so we can generate new ones
df1 = df1.drop(columns=['key_id'], errors='ignore')
df2 = df2.drop(columns=['key_id'], errors='ignore')

# 3. Fuse: Concatenate them. 
# Pandas automatically adds NULLs where columns don't match.
fused_df = pd.concat([df1, df2], ignore_index=True, sort=False)

# 4. Save to the brand-new Database
new_db = sqlite3.connect('fused_database_2.db')
fused_df.to_sql('imag_with_seqs', new_db, if_exists='replace', index=True, index_label='key_id')

print("Fusion complete! New keys generated in 'new_key_id' column.")