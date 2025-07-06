import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime

# === STEP 0: Load Data ===
df = pd.read_csv("drp_combined.csv")
print("Columns:", df.columns.tolist())

# === STEP 1: Define Binary Outcome ===
df['lr_event'] = (df['LR'] >= 1).astype(int)

# === STEP 2: Filter Out Future pyears ===
current_year = datetime.now().year
df = df[df['pyear'] < current_year]

# === STEP 3: Extract Month from Purchase Date ===
df['sellmonth'] = pd.to_datetime(df['purchasedate'], errors='coerce').dt.month

# === STEP 4: Custom Dummy Functions ===
def make_dummies_with_highest_as_base(series, prefix):
    categories = sorted(series.dropna().unique())
    dummies = pd.get_dummies(series, prefix=prefix)
    if f"{prefix}_{categories[-1]}" in dummies.columns:
        dummies = dummies.drop(f"{prefix}_{categories[-1]}", axis=1)
    return dummies

def make_dummies_with_lowest_as_base(series, prefix):
    categories = sorted(series.dropna().unique())
    dummies = pd.get_dummies(series, prefix=prefix)
    if f"{prefix}_{categories[0]}" in dummies.columns:
        dummies = dummies.drop(f"{prefix}_{categories[0]}", axis=1)
    return dummies

# === STEP 5: Build Feature Matrix ===
X = pd.concat([
    df[['ClassPrice', 'mil']],
    pd.get_dummies(df['CL'], prefix='CL', drop_first=True),
    pd.get_dummies(df['PF'], prefix='PF', drop_first=True),
    make_dummies_with_highest_as_base(df['pyear'], prefix='pyear'),
    make_dummies_with_lowest_as_base(df['length_code'], prefix='length'),
    make_dummies_with_lowest_as_base(df['sellmonth'], prefix='month'),
    make_dummies_with_highest_as_base(df['statecode_code'], prefix='state'),
], axis=1)

# === STEP 6: Clean Data ===
y = df['lr_event']
X = sm.add_constant(X)
X = X.dropna(axis=1, how='all')  # Drop all-NaN columns
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

valid_rows = X.notnull().all(axis=1) & y.notnull()
X = X[valid_rows]
y = y[valid_rows]

# Drop constant columns (no variance)
constant_cols = X.loc[:, X.nunique() <= 1].columns.tolist()
if constant_cols:
    print("⚠️ Dropping constant columns:", constant_cols)
    X = X.drop(columns=constant_cols)

X = X.astype(float)
y = y.astype(float)

# === STEP 7: VIF Check ===
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n📈 VIF Table:")
print(vif_data.sort_values(by="VIF", ascending=False))

# === STEP 8: Fit Model ===
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# === STEP 9: Print and Save Summary ===
print(result.summary())

with open("logit_summary.txt", "w") as f:
    f.write(result.summary().as_text())

# === STEP 10: Marginal Effects ===
marginal_effects = result.get_margeff(method='dydx')
print(marginal_effects.summary())

with open("logit_marginal_effects.txt", "w") as f:
    f.write(marginal_effects.summary().as_text())
