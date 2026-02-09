import pandas as pd
import glob
import os

print("Searching for CSV files...")

files = glob.glob("data/raw/*.csv")

print("Files Found:", files)

df_list = []

for file in files:
    print("Reading:", file)
    df = pd.read_csv(file, low_memory=False)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

print("Final Dataset Shape:", merged_df.shape)

os.makedirs("data/processed", exist_ok=True)
merged_df.to_csv("data/processed/merged_data.csv", index=False)

print("Merged dataset saved successfully.")
