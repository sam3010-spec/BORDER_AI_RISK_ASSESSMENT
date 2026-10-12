"""Audit target construction and potential circular predictors in master dataset."""
import pandas as pd
import numpy as np

PATH = "data/processed/master_risk_dataset.csv"
df = pd.read_csv(PATH)

print("Shape:", df.shape)
print("\nRisk_Level distribution:")
print(df["Risk_Level"].value_counts(dropna=False))
print("\nUnique Zone_ID:", df["Zone_ID"].nunique(), "of", len(df))

derived = [
    "Environmental_Risk_Score", "Operational_Risk_Score",
    "Historical_Risk_Score", "Risk_Score"
]
print("\nDerived-score correlations with numeric label encoding (descriptive only):")
mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
y = df["Risk_Level"].map(mapping)
for col in derived:
    if col in df:
        print(f"{col:30s} correlation={df[col].corr(y):.4f}")

# Reconstruct documented formula from the source builder and compare.
required = ["Terrain_Risk","Vegetation_Risk","Weather_Risk","Patrol_Risk",
            "Camera_Activity_Score","Sensor_Activity_Score","Historical_Risk"]
missing = [c for c in required if c not in df]
if missing:
    print("Cannot reconstruct formula; missing:", missing)
else:
    ers = .40*df["Terrain_Risk"] + .35*df["Vegetation_Risk"] + .25*df["Weather_Risk"]
    ors = .60*df["Patrol_Risk"] + .20*df["Camera_Activity_Score"] + .20*df["Sensor_Activity_Score"]
    hrs = df["Historical_Risk"]
    raw = .35*ers + .35*ors + .30*hrs
    scaled = 5 + 90*(raw-raw.min())/(raw.max()-raw.min()+1e-8)
    level = pd.Series(scaled).map(lambda s: "Low" if s <= 25 else "Medium" if s <= 50 else "High" if s <= 72 else "Critical")
    print("\nReconstructed Risk_Score max abs difference:",
          float(np.max(np.abs(scaled - df["Risk_Score"]))))
    print("Reconstructed Risk_Level exact match rate:",
          float((level.values == df["Risk_Level"].values).mean()))

print("\nInterpretation:")
print("- A high match/correlation is expected because the target is constructed from these inputs.")
print("- This is not evidence of real-world predictive validity.")
print("- Do not use Risk_Score or component scores as predictors when the target is Risk_Level.")
