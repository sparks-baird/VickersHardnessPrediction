"""Load the MPDS Vickers hardness dataset."""
from os.path import join
import numpy as np
import pandas as pd

from composition_based_feature_vector.composition import generate_features
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(
    join("vickers_hardness", "data", "mpds-vickers-hardness.csv")
).rename(columns={"pretty_formula": "formula", "vickers-hardness": "target"})
df = data[["formula", "target"]]
df = df.dropna()
df = df.groupby(by="formula", as_index=False).mean()
# df = df.drop_duplicates("formula")
X, y, formulae, skipped = generate_features(df)

data2 = pd.read_csv(join("vickers_hardness", "data", "hv_comp_load.csv")).rename(
    columns={"composition": "formula", "hardness": "target"}
)
df2 = data2[["formula", "target"]]
df2 = df2.groupby(by="formula", as_index=False).mean()
# df2 = df2.drop_duplicates("formula")
X2, y2, formulae2, skipped2 = generate_features(df2)

Xcomb = pd.concat((X, X2), axis=0, ignore_index=True)
scaler = MinMaxScaler()
scaler.fit(Xcomb)
Xscl = scaler.transform(Xcomb)
df_scl = pd.DataFrame(np.round(Xscl, decimals=2), columns=X.columns)
n_uniq = df_scl.drop_duplicates().shape[0]
n_dup = df.shape[0] + df2.shape[0] - n_uniq
print(f"Number of shared formulas: {n_dup}")
# Number of shared formulas: 277
# Number of unique formulas (collectively): 783
