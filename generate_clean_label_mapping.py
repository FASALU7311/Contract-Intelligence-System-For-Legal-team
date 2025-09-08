import pandas as pd

# Load the existing label mapping
df = pd.read_csv("label_mapping.csv")

# Extract clean clause types from verbose descriptions
def extract_clause_type(label):
    if '"' in label:
        return label.split('"')[1]  # get text inside first quote pair
    elif ":" in label:
        return label.split(":")[0].strip()
    else:
        return label.strip()

# Apply transformation
df["clause_type"] = df["clause_type"].apply(extract_clause_type)

# Save clean label mapping
df.to_csv("label_mapping.csv", index=False)

print("✅ Cleaned label_mapping.csv saved successfully.")
