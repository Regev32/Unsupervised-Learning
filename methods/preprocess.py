import os
import pandas as pd

# ---------- load ----------
data = pd.read_csv('../data/stroke.csv')

# ---------- clean ----------
# drop outlier row by id
data.drop(data[data['id'] == 56156].index, inplace=True)

# ---------- encode ----------
# binary flags
data['is_ever_married'] = (data['ever_married'] == 'Yes').astype(int)
data['is_male'] = (data['gender'] == 'Male').astype(int)
data['is_Residence_type_Urban'] = (data['Residence_type'] == 'Urban').astype(int)

data.drop(['ever_married', 'gender', 'Residence_type'], axis=1, inplace=True)
# one‑hot for multi‑category columns
data = pd.get_dummies(
    data,
    columns=['work_type', 'smoking_status'],
    prefix=['work_type', 'smoking'],
    drop_first=True,
    dtype=int
)

# ---------- move target to the end ----------
data['stroke'] = data.pop('stroke')

# ---------- save ----------
os.makedirs('../data', exist_ok=True)   # make sure path exists
data.to_csv('../data/stroke.csv', index=False)

print("Finished preprocessing data")