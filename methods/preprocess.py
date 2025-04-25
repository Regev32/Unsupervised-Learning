import os
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---------- suppress warnings ----------
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# ---------- load ----------
data = pd.read_csv('../data/stroke.csv')

# ---------- drop outlier ----------
data.drop(data[data['id'] == 56156].index, inplace=True)

# ---------- impute BMI via 5-NN ----------
num_cols = data.select_dtypes(include='number').columns.drop('id')
bmi_imputer = KNNImputer(n_neighbors=5)
data[num_cols] = bmi_imputer.fit_transform(data[num_cols])

# ---------- handle 'unknown' smoking_status for minors ----------
is_unknown = data['smoking_status'].str.lower() == 'unknown'
is_minor = data['age'] <= 18
data.loc[is_unknown & is_minor, 'smoking_status'] = 'never smoked'

# ---------- impute remaining smoking_status via 5-NN majority vote ----------
features = ['age', 'avg_glucose_level', 'bmi']
mask_unknown = data['smoking_status'].str.lower() == 'unknown'
known = data.loc[~mask_unknown, :]
unknown = data.loc[mask_unknown, :]
nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(known[features])
distances, all_indices = nbrs.kneighbors(unknown[features],
                                         n_neighbors=len(known))

for unk_idx, neighbor_list in zip(unknown.index, all_indices):
    k = 10
    fill = None
    # keep growing the neighborhood until there’s a clear mode
    while True:
        # take the top-k neighbors
        neigh = neighbor_list[:k]
        # count smoking_status among them
        counts = data.loc[neigh, 'smoking_status'].value_counts()
        if counts.empty:
            fill = 'never smoked'
            break
        top_count = counts.iloc[0]
        # if exactly one status has that top count, we have a winner
        if (counts == top_count).sum() == 1:
            fill = counts.index[0]
            break
        # otherwise bump k up
        k += 1
        # if we’ve exhausted all neighbors, just pick the first mode
        if k >= len(neighbor_list):
            fill = counts.index[0]
            break
    data.at[unk_idx, 'smoking_status'] = fill

# ---------- normalize numeric columns ----------
# create new columns with zero mean and unit variance
scale_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = StandardScaler()
scaled_values = scaler.fit_transform(data[scale_cols])
for col, vals in zip(scale_cols, scaled_values.T):
    data[f'normalized_{col}'] = vals
# drop originals
data.drop(scale_cols, axis=1, inplace=True)

# ---------- encoding ----------
# binary flags
data['is_ever_married'] = (data['ever_married'] == 'Yes').astype(int)
data['is_male'] = (data['gender'] == 'Male').astype(int)
data['is_Residence_type_Urban'] = (data['Residence_type'] == 'Urban').astype(int)
# drop originals
data.drop(['ever_married', 'gender', 'Residence_type'], axis=1, inplace=True)
# one-hot for work_type & smoking_status
data = pd.get_dummies(
    data,
    columns=['work_type', 'smoking_status'],
    prefix=['work_type', 'smoking'],
    drop_first=True,
    dtype=int
)
# move target to end
data['stroke'] = data.pop('stroke')

# ---------- save ----------
os.makedirs('../data', exist_ok=True)
data.to_csv('../data/stroke.csv', index=False)

print("Finished preprocessing (including normalized columns)")
