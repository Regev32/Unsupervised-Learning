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
is_minor   = data['age'] <= 18
data.loc[is_unknown & is_minor, 'smoking_status'] = 'never smoked'

# ---------- dynamic-impute smoking_status via expanding K-NN majority vote ----------
feat_cols = ['age', 'avg_glucose_level', 'bmi']
# split known vs unknown
mask_unk = data['smoking_status'].str.lower() == 'unknown'
known    = data.loc[~mask_unk].reset_index(drop=False)
unknown  = data.loc[ mask_unk].reset_index(drop=False)

nbrs = NearestNeighbors()
nbrs.fit(known[feat_cols])
# get neighbor positions for unknown samples
_, all_nbrs = nbrs.kneighbors(unknown[feat_cols], n_neighbors=known.shape[0])

# iterate unknown rows and fill smoking_status
for unk_row, nbr_positions in zip(unknown.itertuples(), all_nbrs):
    k = 10
    fill = 'never smoked'
    while True:
        # map positions to original data index
        topk_pos = nbr_positions[:k]
        topk_idx = known.loc[topk_pos, 'index']  # original indices
        # count values among those neighbors
        counts = data.loc[topk_idx, 'smoking_status'].value_counts()
        if counts.empty:
            break
        max_count = counts.iloc[0]
        # check if unique majority
        if (counts == max_count).sum() == 1:
            fill = counts.index[0]
            break
        k += 1
        if k > len(nbr_positions):
            fill = counts.index[0]
            break
    # assign to original data
    data.at[unk_row.index, 'smoking_status'] = fill

# ---------- normalize numeric columns ----------
scale_cols = ['age', 'avg_glucose_level', 'bmi']
scaler     = StandardScaler()
scaled_vals= scaler.fit_transform(data[scale_cols])
for col, vals in zip(scale_cols, scaled_vals.T):
    data[f'normalized_{col}'] = vals

# ---------- encoding ----------
data['is_ever_married']           = (data['ever_married']   == 'Yes').astype(int)
data['is_male']                   = (data['gender']         == 'Male').astype(int)
data['is_Residence_type_Urban']   = (data['Residence_type'] == 'Urban').astype(int)
# drop original categorical columns
data.drop(['ever_married', 'gender', 'Residence_type'], axis=1, inplace=True)
# one-hot encode work_type & smoking_status
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

print("Finished preprocessing with dynamic smoking imputation")
