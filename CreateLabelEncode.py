from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
from tqdm import tqdm

# Load Data
df = pd.read_csv(r"train.csv")

le_dict = {}

# Encoded categorical column
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in tqdm(categorical_cols, desc='Label Encoding....'):
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

print("Label Encoding Finished.")

# Save the encoded dataset
df_encoded.to_csv('Encoded_train_data.csv', index=False)

# Gather encoding mapping information
encoded_mappings = {}
for col in tqdm(categorical_cols, desc='Creating Mapping Json...'):
    le = le_dict[col]
    encoded_mappings[col] = list(le.classes_)

# Save encoding mapping information as a JSON file
with open('encoded_mappings.json', 'w') as f:
    json.dump(encoded_mappings, f)

print("\nMapping JSON file created finished.")