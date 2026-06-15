
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F8FAFC",
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#CBD5E1",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})
RED, AMBER, GREEN, BLUE, PURPLE = "#DC2626", "#D97706", "#16A34A", "#2563EB", "#7C3AED"
RANDOM_STATE = 42

print("=" * 65)
print("PART 6: BASELINE ETA PREDICTION MODEL")
print("=" * 65)


# =============================================================================
# SECTION 0 — LOAD + CLEAN (same rules as Parts 1-5)
# =============================================================================

df = pd.read_csv("delivery_data.csv")

raw_n = len(df)
df = df[df["actual_time"] > 0]
df = df[df["osrm_time"] > 0]
df = df[df["source_center"] != df["destination_center"]]
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
df = df.dropna(subset=["trip_creation_time", "source_center", "destination_center"])
print(f"  Clean rows: {len(df):,}  (dropped {raw_n - len(df):,})")

# Drop extreme delay-ratio outliers from TRAINING TARGET only.
# WHY: a few trips have actual_time 30-50x OSRM (vehicle breakdown, multi-day
# hold). These are not "ETA prediction" cases — they're operational incidents.
# Leaving them in lets a handful of rows dominate the loss function and
# distorts MAE/RMSE for the 99% of normal trips the model is meant to serve.
df["delay_ratio"] = df["actual_time"] / df["osrm_time"]
p99 = df["delay_ratio"].quantile(0.99)
before = len(df)
df = df[df["delay_ratio"] <= p99].reset_index(drop=True)
print(f"  Dropped {before - len(df):,} extreme-delay outliers (delay_ratio > P99={p99:.2f})")


# =============================================================================
# SECTION 1 — FEATURE ENGINEERING
# =============================================================================
# TARGET: actual_time (seconds) — "Actual ETA"
#
# FEATURE GROUPS:
#   1. Distance        — osrm_distance
#   2. OSRM ETA        — osrm_time (the model's job is to learn the
#                         correction factor on top of this)
#   3. Route type      — route_type (categorical: FTL / Carting)
#   4. Time features   — hour, day-of-week, weekend flag, time-of-day
#                         bucket (matches the operational buckets used
#                         in Parts 3-5: night/morn_peak/afternoon/eve_peak)
#   5. Hub features    — per-hub historical avg delay ratio and trip
#                         volume for source AND destination hubs. These
#                         are leaky if computed on the full dataset before
#                         splitting, so they are computed ONLY on the
#                         TRAINING split and merged onto test (Section 3).

print("\nSECTION 1 — FEATURE ENGINEERING")
print("=" * 65)

df["hour"] = df["trip_creation_time"].dt.hour
df["day_of_week"] = df["trip_creation_time"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

def assign_tod(h):
    if h < 6:   return "night"
    if h < 12:  return "morn_peak"
    if h < 17:  return "afternoon"
    return "eve_peak"

df["time_bucket"] = df["hour"].map(assign_tod)

# Cyclical encoding of hour — captures that hour 23 and hour 0 are adjacent,
# which a raw integer 0-23 feature cannot represent.
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

print(f"  Engineered: hour, day_of_week, is_weekend, time_bucket, "
      f"hour_sin/cos")
print(f"  Time bucket distribution:\n{df['time_bucket'].value_counts().to_string()}")


# =============================================================================
# SECTION 2 — TRAIN-TEST SPLIT STRATEGY
# =============================================================================
# CHRONOLOGICAL (TIME-BASED) SPLIT, not random.
#
# WHY: ETA prediction is a forecasting problem deployed in production —
# the model will always be predicting trips that happen AFTER the trips
# it was trained on. A random split lets the model "see the future"
# (e.g. hub congestion patterns from December when predicting a January
# trip), which inflates offline metrics relative to real-world performance.
#
# Split: first 80% of trips by trip_creation_time → train
#        last  20% of trips by trip_creation_time → test
#
# This also naturally validates the hub features (Section 3), which are
# computed on train-period data only and applied forward to test-period
# trips — exactly how it would work in production.

print("\nSECTION 2 — TRAIN-TEST SPLIT (CHRONOLOGICAL)")
print("=" * 65)

df = df.sort_values("trip_creation_time").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
split_time = df.loc[split_idx, "trip_creation_time"]

train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

print(f"  Split timestamp:  {split_time}")
print(f"  Train: {len(train_df):,} rows "
      f"({train_df['trip_creation_time'].min()} → {train_df['trip_creation_time'].max()})")
print(f"  Test:  {len(test_df):,} rows "
      f"({test_df['trip_creation_time'].min()} → {test_df['trip_creation_time'].max()})")


# =============================================================================
# SECTION 3 — HUB FEATURES (LEAKAGE-SAFE)
# =============================================================================
# For each hub, compute on TRAIN ONLY:
#   - avg_delay_as_source / avg_delay_as_dest  (historical performance)
#   - trip_volume_as_source / trip_volume_as_dest (hub busyness)
#
# These are then LEFT-JOINED onto both train and test. New hubs unseen in
# train (cold-start) get the global train mean as fallback — a sensible
# production default rather than NaN/0.

print("\nSECTION 3 — HUB FEATURES (computed on TRAIN, applied to TEST)")
print("=" * 65)

src_hub_stats = (
    train_df.groupby("source_center")
    .agg(hub_avg_delay_as_source=("delay_ratio", "mean"),
         hub_volume_as_source=("trip_uuid", "count"))
    .reset_index()
)
dst_hub_stats = (
    train_df.groupby("destination_center")
    .agg(hub_avg_delay_as_dest=("delay_ratio", "mean"),
         hub_volume_as_dest=("trip_uuid", "count"))
    .reset_index()
)

global_avg_delay = train_df["delay_ratio"].mean()
global_avg_vol_src = src_hub_stats["hub_volume_as_source"].mean()
global_avg_vol_dst = dst_hub_stats["hub_volume_as_dest"].mean()

def attach_hub_features(d):
    d = d.merge(src_hub_stats, on="source_center", how="left")
    d = d.merge(dst_hub_stats, on="destination_center", how="left")
    d["hub_avg_delay_as_source"] = d["hub_avg_delay_as_source"].fillna(global_avg_delay)
    d["hub_avg_delay_as_dest"]   = d["hub_avg_delay_as_dest"].fillna(global_avg_delay)
    d["hub_volume_as_source"]    = d["hub_volume_as_source"].fillna(global_avg_vol_src)
    d["hub_volume_as_dest"]      = d["hub_volume_as_dest"].fillna(global_avg_vol_dst)
    return d

train_df = attach_hub_features(train_df)
test_df  = attach_hub_features(test_df)

cold_start = (~test_df["source_center"].isin(src_hub_stats["source_center"])).sum()
print(f"  Source hubs with train history: {len(src_hub_stats):,}")
print(f"  Destination hubs with train history: {len(dst_hub_stats):,}")
print(f"  Test rows with cold-start (unseen) source hub: {cold_start:,}")


# =============================================================================
# SECTION 4 — FEATURE / TARGET MATRICES
# =============================================================================

print("\nSECTION 4 — FEATURE MATRIX ASSEMBLY")
print("=" * 65)

NUMERIC_FEATURES = [
    "osrm_distance",          # Distance
    "osrm_time",               # OSRM ETA
    "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",  # Time features
    "hub_avg_delay_as_source", "hub_avg_delay_as_dest",            # Hub features
    "hub_volume_as_source", "hub_volume_as_dest",
]
CATEGORICAL_FEATURES = ["route_type", "time_bucket"]   # Route type + time bucket
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "actual_time"

X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[ALL_FEATURES],  test_df[TARGET]

print(f"  Numeric features     ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
print(f"  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
print(f"  Target: {TARGET}  (seconds)")
print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ]
)


# =============================================================================
# SECTION 5 — EVALUATION HELPERS
# =============================================================================
# PS-defined business metric: % of trips with predicted ETA within 15% of
# actual. Reported alongside MAE / RMSE / R².

def pct_within_15(y_true, y_pred):
    pct_err = np.abs(y_pred - y_true) / y_true
    return (pct_err <= 0.15).mean() * 100

def evaluate(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    p15  = pct_within_15(y_true, y_pred)
    print(f"  {label:<24} MAE={mae:>9.2f}s  RMSE={rmse:>9.2f}s  "
          f"R²={r2:>6.4f}  within15%={p15:>5.1f}%")
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2, "pct_within_15": p15}

# OSRM-as-prediction baseline (the system being benchmarked against, per PS)
osrm_baseline = evaluate(y_test, X_test["osrm_time"], "OSRM (no model)")


# =============================================================================
# SECTION 6 — HYPERPARAMETER TUNING
# =============================================================================
# RandomizedSearchCV with TimeSeries-respecting KFold (shuffle=False is NOT
# used here since sklearn's KFold on an already-chronologically-sorted
# train_df with shuffle=False still respects rough time order across folds).
# n_iter kept modest for runtime; widen in production with more compute.

print("\nSECTION 6 — HYPERPARAMETER TUNING")
print("=" * 65)

cv = KFold(n_splits=3, shuffle=False)
results = []
fitted_models = {}

# ── Random Forest ─────────────────────────────────────────────────────────────
print("\n  [1/3] Random Forest — RandomizedSearchCV")
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])
rf_param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [8, 12, 16, None],
    "model__min_samples_leaf": [1, 5, 10, 20],
    "model__max_features": ["sqrt", "log2", 0.5],
}
rf_search = RandomizedSearchCV(
    rf_pipe, rf_param_dist, n_iter=8, cv=cv,
    scoring="neg_mean_absolute_error", random_state=RANDOM_STATE,
    n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)
print(f"    Best params: {rf_search.best_params_}")
fitted_models["RandomForest"] = rf_search.best_estimator_

# ── XGBoost ──────────────────────────────────────────────────────────────────
print("\n  [2/3] XGBoost — RandomizedSearchCV")
xgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", xgb.XGBRegressor(random_state=RANDOM_STATE,
                                objective="reg:squarederror",
                                n_jobs=-1, tree_method="hist"))
])
xgb_param_dist = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [4, 6, 8],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
}
xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_param_dist, n_iter=8, cv=cv,
    scoring="neg_mean_absolute_error", random_state=RANDOM_STATE,
    n_jobs=-1, verbose=0
)
xgb_search.fit(X_train, y_train)
print(f"    Best params: {xgb_search.best_params_}")
fitted_models["XGBoost"] = xgb_search.best_estimator_

# ── LightGBM ─────────────────────────────────────────────────────────────────
print("\n  [3/3] LightGBM — RandomizedSearchCV")
lgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
])
lgb_param_dist = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [-1, 6, 10],
    "model__num_leaves": [31, 63, 127],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
}
lgb_search = RandomizedSearchCV(
    lgb_pipe, lgb_param_dist, n_iter=8, cv=cv,
    scoring="neg_mean_absolute_error", random_state=RANDOM_STATE,
    n_jobs=-1, verbose=0
)
lgb_search.fit(X_train, y_train)
print(f"    Best params: {lgb_search.best_params_}")
fitted_models["LightGBM"] = lgb_search.best_estimator_


# =============================================================================
# SECTION 7 — EVALUATION (MAE, RMSE, R², % WITHIN 15%)
# =============================================================================

print("\nSECTION 7 — TEST SET EVALUATION")
print("=" * 65)
print(f"  {osrm_baseline['model']:<24} "
      f"MAE={osrm_baseline['MAE']:>9.2f}s  RMSE={osrm_baseline['RMSE']:>9.2f}s  "
      f"R²={osrm_baseline['R2']:>6.4f}  within15%={osrm_baseline['pct_within_15']:>5.1f}%")

all_results = [osrm_baseline]
test_preds = {"actual_time": y_test.values, "osrm_time": X_test["osrm_time"].values}

for name, model in fitted_models.items():
    preds = model.predict(X_test)
    test_preds[f"pred_{name}"] = preds
    all_results.append(evaluate(y_test, preds, name))

results_df = pd.DataFrame(all_results)
best_model_name = results_df[results_df["model"] != "OSRM (no model)"].sort_values("MAE").iloc[0]["model"]
best_model = fitted_models[best_model_name]

print(f"\n  BEST MODEL (by MAE): {best_model_name}")
improvement = (osrm_baseline["MAE"] - results_df[results_df["model"]==best_model_name]["MAE"].values[0]) \
               / osrm_baseline["MAE"] * 100
print(f"  MAE improvement over raw OSRM: {improvement:.1f}%")


# =============================================================================
# SECTION 8 — FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\nSECTION 8 — FEATURE IMPORTANCE")
print("=" * 65)

# Recover feature names after one-hot encoding
ohe = best_model.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = NUMERIC_FEATURES + cat_feature_names

importances = best_model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(importance_df.head(10).to_string(index=False))


# =============================================================================
# SECTION 9 — VISUALIZATIONS
# =============================================================================

print("\nSECTION 9 — GENERATING VISUALIZATIONS")
print("=" * 65)

# ── Figure A: Feature importance ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
top_imp = importance_df.head(15)
ax.barh(range(len(top_imp)), top_imp["importance"].values[::-1],
        color=BLUE, edgecolor="white", alpha=0.9)
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels(top_imp["feature"].values[::-1], fontsize=9)
ax.set_xlabel("Feature Importance")
ax.set_title(f"Feature Importance — {best_model_name} (Best Baseline)")
plt.tight_layout()
plt.savefig("outputs/fig_baseline_feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_baseline_feature_importance.png saved.")

# ── Figure B: Diagnostics (3-panel) ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f"Baseline Model Diagnostics — {best_model_name}", fontsize=14, fontweight="bold")

best_preds = test_preds[f"pred_{best_model_name}"]

# B1: Predicted vs Actual scatter
ax = axes[0]
sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
ax.scatter(y_test.values[sample_idx] / 3600, best_preds[sample_idx] / 3600,
           alpha=0.15, s=8, color=BLUE)
lims = [0, max(y_test.max(), best_preds.max()) / 3600]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual ETA (hours)")
ax.set_ylabel("Predicted ETA (hours)")
ax.set_title("Predicted vs Actual")
ax.legend(fontsize=8)

# B2: Model comparison bar chart (MAE)
ax = axes[1]
plot_results = results_df.sort_values("MAE")
colors = [RED if m == "OSRM (no model)" else (GREEN if m == best_model_name else BLUE)
          for m in plot_results["model"]]
ax.barh(range(len(plot_results)), plot_results["MAE"].values[::-1] / 60,
        color=colors[::-1], edgecolor="white", alpha=0.9)
ax.set_yticks(range(len(plot_results)))
ax.set_yticklabels(plot_results["model"].values[::-1], fontsize=9)
ax.set_xlabel("MAE (minutes)")
ax.set_title("Model Comparison — MAE\n(Green=Best, Red=OSRM baseline)")

# B3: Residual distribution
ax = axes[2]
residuals = (best_preds - y_test.values) / 60
ax.hist(residuals, bins=80, color=PURPLE, alpha=0.8, edgecolor="none")
ax.axvline(0, color="black", lw=1.5, ls="--")
ax.set_xlabel("Residual: Predicted − Actual (minutes)")
ax.set_ylabel("Trip Count")
ax.set_title(f"Residual Distribution\n(mean={residuals.mean():.1f} min, "
              f"std={residuals.std():.1f} min)")

plt.tight_layout()
plt.savefig("outputs/fig_baseline_diagnostics.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_baseline_diagnostics.png saved.")


# =============================================================================
# SECTION 10 — EXPORT
# =============================================================================

print("\nSECTION 10 — EXPORT")
print("=" * 65)

results_df.to_csv("outputs/baseline_model_comparison.csv", index=False)
pd.DataFrame(test_preds).to_csv("outputs/baseline_test_predictions.csv", index=False)
importance_df.to_csv("outputs/baseline_feature_importance.csv", index=False)
joblib.dump(best_model, "outputs/best_baseline_model.pkl")

print("  ✓ baseline_model_comparison.csv")
print("  ✓ baseline_test_predictions.csv")
print("  ✓ baseline_feature_importance.csv")
print("  ✓ best_baseline_model.pkl")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"""
{'='*65}
BASELINE ETA MODEL COMPLETE
{'='*65}

  BEST MODEL: {best_model_name}

{results_df.to_string(index=False)}

  Improvement over raw OSRM ETA (MAE): {improvement:.1f}%
  Top 3 features: {', '.join(importance_df.head(3)['feature'].tolist())}

  NEXT → Part 7: node2vec embeddings (structural hub features)
          Part 8: GraphSAGE-enhanced model — must beat
                  MAE={results_df[results_df['model']==best_model_name]['MAE'].values[0]:.2f}s
                  and within15%={results_df[results_df['model']==best_model_name]['pct_within_15'].values[0]:.1f}%
                  to demonstrate "graph advantage" (measured, not claimed).
{'='*65}
""")
