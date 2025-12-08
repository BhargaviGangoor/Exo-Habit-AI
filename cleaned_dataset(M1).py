import pandas as pd
from sklearn.impute import KNNImputer

# ==========================================================
# 1️⃣ LOAD FIRST DATASET
# ==========================================================
file_path = "/kaggle/input/nasa-exoplanet/nasa_exoplanet.csv"
df = pd.read_csv(file_path, comment='#', engine="python", on_bad_lines="skip")

needed = [
    "pl_rade", "pl_bmasse", "pl_dens", "pl_eqt",
    "pl_orbper", "sy_dist", "st_teff", 
    "st_lum", "st_spectype", "st_metfe",
    "pl_orbeccen", "pl_orbsmax"
]

df = df[[col for col in needed if col in df.columns]]

df = df.rename(columns={
    "pl_rade": "radius",
    "pl_bmasse": "mass",
    "pl_eqt": "temp",
    "pl_orbper": "orbital_period",
    "sy_dist": "distance_star",
    "st_teff": "star_temp",
    "st_spectype": "star_type",
    "pl_orbeccen": "eccentricity",
    "pl_orbsmax": "semi_major_axis"
})

selected_cols = [
    "radius", "mass", "temp", "orbital_period",
    "distance_star", "star_temp", "star_type",
    "eccentricity", "semi_major_axis"
]

df = df[selected_cols]

# ==========================================================
# 2️⃣ LOAD SECOND DATASET
# ==========================================================
file2_path = "/kaggle/input/exoplanetsdata1/exoplanetsdata1.csv"
df2 = pd.read_csv(file2_path, comment='#', engine="python", on_bad_lines="skip")

df2 = df2.loc[:, ~df2.columns.str.contains("^Unnamed")]

df2 = df2.rename(columns={
    "pl_rade": "radius",
    "pl_bmasse": "mass",
    "pl_eqt": "temp",
    "pl_orbper": "orbital_period",
    "sy_dist": "distance_star",
    "st_teff": "star_temp",
    "st_spectype": "star_type",
    "pl_orbeccen": "eccentricity",
    "pl_orbsmax": "semi_major_axis"
})

df2 = df2[selected_cols]

# ==========================================================
# 3️⃣ COMBINE BOTH DATASETS
# ==========================================================
combined_df = pd.concat([df, df2], ignore_index=True)
combined_df = combined_df.drop_duplicates()

# ==========================================================
# 4️⃣ FILL EASY MISSING VALUES FIRST
# ==========================================================
combined_df["star_type"] = combined_df["star_type"].fillna("Unknown")

combined_df["eccentricity"] = combined_df["eccentricity"].fillna(combined_df["eccentricity"].mean())
combined_df["semi_major_axis"] = combined_df["semi_major_axis"].fillna(combined_df["semi_major_axis"].mean())
combined_df["distance_star"] = combined_df["distance_star"].fillna(combined_df["distance_star"].mean())

# ==========================================================
# 5️⃣ CRITICAL COLUMNS — SMART IMPUTATION
# ==========================================================
critical_cols = ["radius", "mass", "temp", "star_temp", "orbital_period"]

# ---------------------------
# A) FIRST TRY: GROUP-WISE MEDIAN
# ---------------------------
for col in critical_cols:
    combined_df[col] = combined_df.groupby("star_type")[col].transform(
        lambda x: x.fillna(x.median())
    )

# Check if any missing still left
missing_after_median = combined_df[critical_cols].isnull().sum()
print("Missing AFTER group-wise median:\n", missing_after_median)

# ---------------------------
# B) NEXT TRY: KNN IMPUTATION (only if needed)
# ---------------------------
if combined_df[critical_cols].isnull().sum().sum() > 0:
    print("\nApplying KNN Imputation since some values are still missing...\n")
    
    numeric_df = combined_df.select_dtypes(include=['float64', 'int64'])
    imputer = KNNImputer(n_neighbors=5)
    
    numeric_filled = imputer.fit_transform(numeric_df)
    numeric_filled = pd.DataFrame(numeric_filled, columns=numeric_df.columns)
    
    combined_df[numeric_df.columns] = numeric_filled

# ---------------------------
# 6️⃣ DROP ROWS ONLY IF STILL IMPOSSIBLE TO FILL
# ---------------------------
df_clean = combined_df.dropna(subset=critical_cols)

print("\nFinal Missing Values:\n", df_clean.isnull().sum())
print("\nFinal dataset shape:", df_clean.shape)
