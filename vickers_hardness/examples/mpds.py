"""Load the MPDS Vickers hardness dataset."""
from os.path import join
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

from composition_based_feature_vector.composition import generate_features
from sklearn.model_selection import KFold, cross_validate, GroupKFold

from vickers_hardness.utils.plotting import parity_with_err
from vickers_hardness.vickers_hardness_ import VickersHardness

mapper = {"pretty_formula": "formula", "vickers-hardness": "target"}
fpath = join("vickers_hardness", "data", "mpds-vickers-hardness.csv")
data = pd.read_csv(fpath).rename(columns=mapper)
df = data[["formula", "target"]]
df = df.dropna()
df = df.groupby(by="formula", as_index=False).mean()
X, y, formulae, skipped = generate_features(df)

# add formula back in for VickersHardness()
X["formula"] = formulae

hyperopt = True
recalibrate = True
split_by_groups = False  # doesn't do anything since repeat formulae skipped

# %% K-fold cross-validation
if split_by_groups:
    cv = GroupKFold()
    cvtype = "gcv"
    groups=X["load"]
else:
    cv = KFold(shuffle=True, random_state=100)  # ignores groups
    cvtype = "cv"

figure_dir = join("figures", "mpds", f"{cvtype}_hyper{hyperopt}")
result_dir = join("results", "mpds", f"{cvtype}_hyper{hyperopt}")

results = cross_validate(
    VickersHardness(hyperopt=hyperopt, recalibrate=recalibrate, result_dir=result_dir),
    X,
    y,
    # groups=formulae,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_estimator=True,
)

estimators = results["estimator"]
result_dfs = [estimator.result_df for estimator in estimators]
merge_df = pd.concat(result_dfs)
merge_df["actual_hardness"] = y

parity_with_err(
    merge_df,
    figfolder=figure_dir,
    error_y="y_upper",
    error_y_minus="y_lower",
    fname=f"parity_ci",
    size=None,
)
parity_with_err(
    merge_df, figfolder=figure_dir, error_y="y_std", fname=f"parity_stderr", size=None,
)
parity_with_err(
    merge_df, figfolder=figure_dir, fname=f"parity_stderr_calib", size=None,
)

y_true, y_pred = [merge_df["actual_hardness"], merge_df["predicted_hardness"]]
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")

# CV and GCV equivalent because duplicates are removed first
# GCV-MAE:  (HV)
# GCV-RMSE:  (HV)

merge_df.sort_index().to_csv(join(result_dir, "results.csv"))

1 + 1

