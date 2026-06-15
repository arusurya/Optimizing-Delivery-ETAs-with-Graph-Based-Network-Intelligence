

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from node2vec import Node2Vec
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.manifold import TSNE
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
np.random.seed(RANDOM_STATE)

print("=" * 65)
print("PART 7: NODE2VEC GRAPH-ENHANCED ETA MODEL")
print("=" * 65)


# =============================================================================
# SECTION 0 — LOAD + CLEAN (identical to Part 6, for a fair comparison)
# =============================================================================

df = pd.read_csv("delivery_data.csv")

raw_n = len(df)
df = df[df["actual_time"] > 0]
df = df[df["osrm_time"] > 0]
df = df[df["source_center"] != df["destination_center"]]
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
df = df.dropna(subset=["trip_creation_time", "source_center", "destination_center"])

df["delay_ratio"] = df["actual_time"] / df["osrm_time"]
p99 = df["delay_ratio"].quantile(0.99)
df = df[df["delay_ratio"] <= p99].reset_index(drop=True)
print(f"  Clean rows: {len(df):,}  (dropped {raw_n - len(df):,})")


# =============================================================================
# SECTION 1 — TIME / TIME-BUCKET FEATURES (identical to Part 6)
# =============================================================================

df["hour"] = df["trip_creation_time"].dt.hour
df["day_of_week"] = df["trip_creation_time"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

def assign_tod(h):
    if h < 6:   return "night"
    if h < 12:  return "morn_peak"
    if h < 17:  return "afternoon"
    return "eve_peak"

df["time_bucket"] = df["hour"].map(assign_tod)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


# =============================================================================
# SECTION 2 — CHRONOLOGICAL TRAIN-TEST SPLIT (identical to Part 6)
# =============================================================================
# CRITICAL FOR FAIR COMPARISON: same split point, same train/test rows as
# Part 6. The graph itself is ALSO built on TRAIN ONLY (Section 3) — this
# prevents test-period corridor/delay information leaking into the
# embeddings, mirroring the leakage-safe hub features from Part 6.

print("\nSECTION 2 — TRAIN-TEST SPLIT (CHRONOLOGICAL, matches Part 6)")
print("=" * 65)

df = df.sort_values("trip_creation_time").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
split_time = df.loc[split_idx, "trip_creation_time"]

train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

print(f"  Split timestamp: {split_time}")
print(f"  Train: {len(train_df):,} rows   Test: {len(test_df):,} rows")


# =============================================================================
# SECTION 3 — GRAPH CONSTRUCTION (TRAIN-ONLY, for Node2Vec)
# =============================================================================
# Directed weighted graph: facilities = nodes, corridors = edges.
# Edge weight = median delay_ratio on that corridor (TRAIN data only),
# matching Part 3's convention — high delay_ratio = "expensive" edge.
#
# Node2Vec's random walks use edge weights as transition probabilities,
# so corridors with chronic delay are naturally treated as "harder to
# traverse" — embeddings implicitly encode each hub's position relative
# to delay-prone parts of the network, not just raw topology.

print("\nSECTION 3 — GRAPH CONSTRUCTION (train-only)")
print("=" * 65)

edge_df = (
    train_df.groupby(["source_center", "destination_center"])
    .agg(weight=("delay_ratio", "median"), volume=("trip_uuid", "count"))
    .reset_index()
)
edge_df = edge_df[edge_df["volume"] >= 5]   # drop sparse/noisy edges

G = nx.from_pandas_edgelist(
    edge_df, source="source_center", target="destination_center",
    edge_attr="weight", create_using=nx.DiGraph()
)

print(f"  Nodes (hubs): {G.number_of_nodes():,}")
print(f"  Edges (corridors): {G.number_of_edges():,}")

# Node2Vec requires a connected-enough graph for meaningful walks; isolated
# nodes (hubs only in test, never in train) are handled via fallback in
# Section 5.


# =============================================================================
# SECTION 4 — NODE2VEC EMBEDDING GENERATION
# =============================================================================
# Hyperparameters:
#   dimensions=32      — embedding size (small enough to avoid overfitting
#                         a ~150-hub graph, large enough to capture
#                         meaningful structure)
#   walk_length=30      — long enough to traverse multi-hop corridors
#   num_walks=200       — per node, for stable embeddings
#   p=1, q=0.5          — q<1 biases walks towards BFS-like exploration
#                         (favors structural/role similarity over pure
#                         community detection) — appropriate since we
#                         care about a hub's ROLE (bottleneck vs. spoke)
#                         more than which community it's in
#   window=10           — skip-gram context window

print("\nSECTION 4 — NODE2VEC TRAINING")
print("=" * 65)

EMBED_DIM = 32

node2vec = Node2Vec(
    G, dimensions=EMBED_DIM, walk_length=30, num_walks=200,
    p=1, q=0.5, weight_key="weight", workers=4, quiet=True, seed=RANDOM_STATE
)
n2v_model = node2vec.fit(window=10, min_count=1, batch_words=128, seed=RANDOM_STATE)

embedding_dict = {node: n2v_model.wv[node] for node in G.nodes()}
print(f"  Generated {EMBED_DIM}-dim embeddings for {len(embedding_dict):,} hubs")

embed_cols = [f"n2v_{i}" for i in range(EMBED_DIM)]
embedding_df = pd.DataFrame.from_dict(embedding_dict, orient="index", columns=embed_cols)
embedding_df.index.name = "hub"
embedding_df.reset_index().to_csv("outputs/node2vec_embeddings.csv", index=False)
print("  ✓ node2vec_embeddings.csv saved.")

# Fallback embedding for hubs unseen at train time (cold start) = mean embedding
fallback_embedding = embedding_df[embed_cols].mean().values


# =============================================================================
# SECTION 5 — SOURCE & DESTINATION EMBEDDINGS PER SHIPMENT
# =============================================================================

print("\nSECTION 5 — ATTACHING SOURCE/DEST EMBEDDINGS TO SHIPMENTS")
print("=" * 65)

def get_embedding(hub):
    if hub in embedding_dict:
        return embedding_dict[hub]
    return fallback_embedding

def attach_embeddings(d):
    src_emb = np.vstack(d["source_center"].map(get_embedding).values)
    dst_emb = np.vstack(d["destination_center"].map(get_embedding).values)
    src_df = pd.DataFrame(src_emb, columns=[f"src_{c}" for c in embed_cols], index=d.index)
    dst_df = pd.DataFrame(dst_emb, columns=[f"dst_{c}" for c in embed_cols], index=d.index)
    return pd.concat([d, src_df, dst_df], axis=1)

train_df = attach_embeddings(train_df)
test_df  = attach_embeddings(test_df)

cold_start = (~test_df["source_center"].isin(embedding_dict.keys())).sum() + \
             (~test_df["destination_center"].isin(embedding_dict.keys())).sum()
print(f"  Source embedding cols: {len([c for c in train_df.columns if c.startswith('src_n2v')])}")
print(f"  Dest   embedding cols: {len([c for c in train_df.columns if c.startswith('dst_n2v')])}")
print(f"  Cold-start hub instances in test (fallback used): {cold_start:,}")


# =============================================================================
# SECTION 6 — MERGE WITH PART 6 BASELINE FEATURES (HUB STATS)
# =============================================================================
# Re-derive the same leakage-safe hub stats from Part 6 (train-only),
# so the graph model has access to identical non-graph context — any
# performance delta is attributable to the embeddings, not extra info.

print("\nSECTION 6 — MERGING WITH BASELINE FEATURES")
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

def attach_hub_stats(d):
    d = d.merge(src_hub_stats, on="source_center", how="left")
    d = d.merge(dst_hub_stats, on="destination_center", how="left")
    d["hub_avg_delay_as_source"] = d["hub_avg_delay_as_source"].fillna(global_avg_delay)
    d["hub_avg_delay_as_dest"]   = d["hub_avg_delay_as_dest"].fillna(global_avg_delay)
    d["hub_volume_as_source"]    = d["hub_volume_as_source"].fillna(global_avg_vol_src)
    d["hub_volume_as_dest"]      = d["hub_volume_as_dest"].fillna(global_avg_vol_dst)
    return d

train_df = attach_hub_stats(train_df)
test_df  = attach_hub_stats(test_df)


# =============================================================================
# SECTION 7 — FEATURE / TARGET MATRICES
# =============================================================================

print("\nSECTION 7 — FEATURE MATRIX ASSEMBLY")
print("=" * 65)

BASELINE_NUMERIC = [
    "osrm_distance", "osrm_time",
    "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "hub_avg_delay_as_source", "hub_avg_delay_as_dest",
    "hub_volume_as_source", "hub_volume_as_dest",
]
EMBEDDING_FEATURES = [f"src_n2v_{i}" for i in range(EMBED_DIM)] + \
                     [f"dst_n2v_{i}" for i in range(EMBED_DIM)]
NUMERIC_FEATURES = BASELINE_NUMERIC + EMBEDDING_FEATURES
CATEGORICAL_FEATURES = ["route_type", "time_bucket"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "actual_time"

X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[ALL_FEATURES],  test_df[TARGET]

print(f"  Baseline numeric features: {len(BASELINE_NUMERIC)}")
print(f"  Node2Vec embedding features: {len(EMBEDDING_FEATURES)} "
      f"({EMBED_DIM} src + {EMBED_DIM} dst)")
print(f"  Categorical features: {CATEGORICAL_FEATURES}")
print(f"  Total features: {len(ALL_FEATURES)}")
print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ]
)


# =============================================================================
# SECTION 8 — EVALUATION HELPERS (identical to Part 6)
# =============================================================================

def pct_within_15(y_true, y_pred):
    pct_err = np.abs(y_pred - y_true) / y_true
    return (pct_err <= 0.15).mean() * 100

def evaluate(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    p15  = pct_within_15(y_true, y_pred)
    print(f"  {label:<28} MAE={mae:>9.2f}s  RMSE={rmse:>9.2f}s  "
          f"R²={r2:>6.4f}  within15%={p15:>5.1f}%")
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2, "pct_within_15": p15}


# =============================================================================
# SECTION 9 — HYPERPARAMETER TUNING (graph-enhanced models)
# =============================================================================

print("\nSECTION 9 — HYPERPARAMETER TUNING (graph-enhanced)")
print("=" * 65)

cv = KFold(n_splits=3, shuffle=False)
fitted_models = {}

print("\n  [1/3] Random Forest (graph-enhanced) — RandomizedSearchCV")
rf_pipe = Pipeline([("prep", preprocessor),
                     ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
rf_param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [8, 12, 16, None],
    "model__min_samples_leaf": [1, 5, 10, 20],
    "model__max_features": ["sqrt", "log2", 0.5],
}
rf_search = RandomizedSearchCV(rf_pipe, rf_param_dist, n_iter=8, cv=cv,
                                scoring="neg_mean_absolute_error",
                                random_state=RANDOM_STATE, n_jobs=-1)
rf_search.fit(X_train, y_train)
print(f"    Best params: {rf_search.best_params_}")
fitted_models["RandomForest_graph"] = rf_search.best_estimator_

print("\n  [2/3] XGBoost (graph-enhanced) — RandomizedSearchCV")
xgb_pipe = Pipeline([("prep", preprocessor),
                      ("model", xgb.XGBRegressor(random_state=RANDOM_STATE,
                                                  objective="reg:squarederror",
                                                  n_jobs=-1, tree_method="hist"))])
xgb_param_dist = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [4, 6, 8],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
}
xgb_search = RandomizedSearchCV(xgb_pipe, xgb_param_dist, n_iter=8, cv=cv,
                                 scoring="neg_mean_absolute_error",
                                 random_state=RANDOM_STATE, n_jobs=-1)
xgb_search.fit(X_train, y_train)
print(f"    Best params: {xgb_search.best_params_}")
fitted_models["XGBoost_graph"] = xgb_search.best_estimator_

print("\n  [3/3] LightGBM (graph-enhanced) — RandomizedSearchCV")
lgb_pipe = Pipeline([("prep", preprocessor),
                      ("model", lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))])
lgb_param_dist = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [-1, 6, 10],
    "model__num_leaves": [31, 63, 127],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
}
lgb_search = RandomizedSearchCV(lgb_pipe, lgb_param_dist, n_iter=8, cv=cv,
                                 scoring="neg_mean_absolute_error",
                                 random_state=RANDOM_STATE, n_jobs=-1)
lgb_search.fit(X_train, y_train)
print(f"    Best params: {lgb_search.best_params_}")
fitted_models["LightGBM_graph"] = lgb_search.best_estimator_


# =============================================================================
# SECTION 10 — EVALUATION + COMPARISON AGAINST PART 6 BASELINE
# =============================================================================

print("\nSECTION 10 — TEST SET EVALUATION (graph-enhanced)")
print("=" * 65)

graph_results = []
test_preds = {"actual_time": y_test.values, "osrm_time": X_test["osrm_time"].values}
for name, model in fitted_models.items():
    preds = model.predict(X_test)
    test_preds[f"pred_{name}"] = preds
    graph_results.append(evaluate(y_test, preds, name))

graph_results_df = pd.DataFrame(graph_results)
best_graph_name = graph_results_df.sort_values("MAE").iloc[0]["model"]
best_graph_model = fitted_models[best_graph_name]
print(f"\n  BEST GRAPH MODEL (by MAE): {best_graph_name}")

# ── Load Part 6 baseline results for comparison ──────────────────────────────
print("\n  Loading Part 6 baseline results for comparison...")
try:
    baseline_results_df = pd.read_csv("outputs/baseline_model_comparison.csv")
    baseline_no_osrm = baseline_results_df[baseline_results_df["model"] != "OSRM (no model)"]
    best_baseline_row = baseline_no_osrm.sort_values("MAE").iloc[0]
    best_baseline_name = best_baseline_row["model"]
    print(f"  Best Part 6 baseline: {best_baseline_name} "
          f"(MAE={best_baseline_row['MAE']:.2f}s, "
          f"within15%={best_baseline_row['pct_within_15']:.1f}%)")
    HAVE_BASELINE = True
except FileNotFoundError:
    print("  ⚠ baseline_model_comparison.csv not found — run Part 6 first.")
    print("  Falling back to re-training a quick LightGBM baseline (non-graph) inline.")
    HAVE_BASELINE = False

    base_preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", BASELINE_NUMERIC),
                       ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        CATEGORICAL_FEATURES)]
    )
    base_pipe = Pipeline([("prep", base_preprocessor),
                           ("model", lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                                                        n_estimators=400, num_leaves=63,
                                                        learning_rate=0.05))])
    base_pipe.fit(train_df[BASELINE_NUMERIC + CATEGORICAL_FEATURES], y_train)
    base_preds = base_pipe.predict(test_df[BASELINE_NUMERIC + CATEGORICAL_FEATURES])
    best_baseline_row = pd.Series(evaluate(y_test, base_preds, "LightGBM (baseline, inline)"))
    best_baseline_name = "LightGBM (baseline, inline)"
    test_preds["pred_baseline_inline"] = base_preds


# =============================================================================
# SECTION 11 — QUANTIFY GRAPH ADVANTAGE
# =============================================================================
# "Graph advantage" = measured delta on the SAME test set, same metrics:
#   - Δ MAE  (seconds and %)
#   - Δ RMSE (seconds and %)
#   - Δ R²
#   - Δ % within 15%  (percentage points)

print("\nSECTION 11 — GRAPH ADVANTAGE (graph-enhanced vs baseline)")
print("=" * 65)

best_graph_row = graph_results_df[graph_results_df["model"] == best_graph_name].iloc[0]

advantage = {
    "baseline_model": best_baseline_name,
    "graph_model": best_graph_name,
    "baseline_MAE": best_baseline_row["MAE"],
    "graph_MAE": best_graph_row["MAE"],
    "MAE_improvement_pct": (best_baseline_row["MAE"] - best_graph_row["MAE"]) / best_baseline_row["MAE"] * 100,
    "baseline_RMSE": best_baseline_row["RMSE"],
    "graph_RMSE": best_graph_row["RMSE"],
    "RMSE_improvement_pct": (best_baseline_row["RMSE"] - best_graph_row["RMSE"]) / best_baseline_row["RMSE"] * 100,
    "baseline_R2": best_baseline_row["R2"],
    "graph_R2": best_graph_row["R2"],
    "R2_delta": best_graph_row["R2"] - best_baseline_row["R2"],
    "baseline_within15": best_baseline_row["pct_within_15"],
    "graph_within15": best_graph_row["pct_within_15"],
    "within15_delta_pp": best_graph_row["pct_within_15"] - best_baseline_row["pct_within_15"],
}
advantage_df = pd.DataFrame([advantage])

print(f"  {'Metric':<14} {'Baseline':>12} {'Graph':>12} {'Δ':>14}")
print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*14}")
print(f"  {'MAE (s)':<14} {advantage['baseline_MAE']:>12.2f} {advantage['graph_MAE']:>12.2f} "
      f"{advantage['MAE_improvement_pct']:>+12.2f}% better")
print(f"  {'RMSE (s)':<14} {advantage['baseline_RMSE']:>12.2f} {advantage['graph_RMSE']:>12.2f} "
      f"{advantage['RMSE_improvement_pct']:>+12.2f}% better")
print(f"  {'R²':<14} {advantage['baseline_R2']:>12.4f} {advantage['graph_R2']:>12.4f} "
      f"{advantage['R2_delta']:>+12.4f}")
print(f"  {'Within 15%':<14} {advantage['baseline_within15']:>11.1f}% {advantage['graph_within15']:>11.1f}% "
      f"{advantage['within15_delta_pp']:>+11.1f} pp")

VERDICT_THRESHOLD = 1.0  # % MAE improvement to call it a real "advantage"
if advantage["MAE_improvement_pct"] > VERDICT_THRESHOLD:
    verdict = (f"GRAPH ADVANTAGE CONFIRMED: Node2Vec embeddings reduce MAE by "
               f"{advantage['MAE_improvement_pct']:.2f}% and improve within-15% "
               f"accuracy by {advantage['within15_delta_pp']:.1f} percentage points.")
elif advantage["MAE_improvement_pct"] > -VERDICT_THRESHOLD:
    verdict = (f"GRAPH ADVANTAGE INCONCLUSIVE: MAE delta ({advantage['MAE_improvement_pct']:.2f}%) "
               f"is within noise. Graph features did not measurably help on this dataset.")
else:
    verdict = (f"NO GRAPH ADVANTAGE: graph-enhanced model is {-advantage['MAE_improvement_pct']:.2f}% "
               f"WORSE on MAE than baseline. Embeddings may be adding noise/overfitting risk "
               f"given dataset size — recommend trying lower EMBED_DIM or GraphSAGE (Part 8).")

print(f"\n  VERDICT: {verdict}")


# =============================================================================
# SECTION 12 — FEATURE IMPORTANCE (graph vs non-graph features)
# =============================================================================

print("\nSECTION 12 — FEATURE IMPORTANCE (graph-enhanced model)")
print("=" * 65)

ohe = best_graph_model.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = NUMERIC_FEATURES + cat_feature_names

importances = best_graph_model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}) \
                  .sort_values("importance", ascending=False).reset_index(drop=True)

# Group importance by feature type for the "is the graph pulling its weight" view
def feature_group(f):
    if f.startswith("src_n2v"): return "Node2Vec (source hub)"
    if f.startswith("dst_n2v"): return "Node2Vec (destination hub)"
    if f.startswith("hub_"):    return "Hub stats (non-graph)"
    if f in CATEGORICAL_FEATURES or any(f.startswith(c) for c in CATEGORICAL_FEATURES):
        return "Categorical"
    if f in ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]: return "Time"
    return "Distance/OSRM"

importance_df["group"] = importance_df["feature"].apply(feature_group)
group_importance = importance_df.groupby("group")["importance"].sum().sort_values(ascending=False)

print(importance_df.head(10).to_string(index=False))
print(f"\n  Importance by feature group:")
print(group_importance.to_string())


# =============================================================================
# SECTION 13 — VISUALIZATIONS
# =============================================================================

print("\nSECTION 13 — GENERATING VISUALIZATIONS")
print("=" * 65)

# ── Figure A: t-SNE of Node2Vec embeddings ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8))
n_nodes = len(embedding_df)
perplexity = min(30, max(5, n_nodes // 3))
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity, init="pca")
coords = tsne.fit_transform(embedding_df[embed_cols].values)

# Color by total trip volume (proxy for hub importance/centrality)
hub_volume = (
    pd.concat([
        train_df.groupby("source_center")["trip_uuid"].count(),
        train_df.groupby("destination_center")["trip_uuid"].count()
    ], axis=1).fillna(0).sum(axis=1)
)
colors_vals = [hub_volume.get(h, 0) for h in embedding_df.index]

sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors_vals, cmap="viridis",
                s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="Total trip volume (train)")
# Label top-5 busiest hubs
top5_hubs = hub_volume.sort_values(ascending=False).head(5).index
for h in top5_hubs:
    if h in embedding_df.index:
        idx = list(embedding_df.index).index(h)
        ax.annotate(h, coords[idx], fontsize=9, fontweight="bold",
                     xytext=(5, 5), textcoords="offset points")
ax.set_title(f"Node2Vec Hub Embeddings — t-SNE Projection\n({EMBED_DIM}-dim → 2D)")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("outputs/fig_node2vec_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_node2vec_tsne.png saved.")

# ── Figure B: Graph advantage comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f"Graph Advantage: {best_baseline_name} (baseline) vs "
              f"{best_graph_name} (Node2Vec-enhanced)", fontsize=13, fontweight="bold")

# B1: MAE / RMSE comparison
ax = axes[0]
metrics_compare = pd.DataFrame({
    "Baseline": [advantage["baseline_MAE"]/60, advantage["baseline_RMSE"]/60],
    "Graph-enhanced": [advantage["graph_MAE"]/60, advantage["graph_RMSE"]/60],
}, index=["MAE", "RMSE"])
metrics_compare.plot(kind="bar", ax=ax, color=[BLUE, GREEN], edgecolor="white", alpha=0.9)
ax.set_ylabel("Minutes")
ax.set_title("Error Metrics (lower = better)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(fontsize=8)

# B2: R² and within-15% comparison
ax = axes[1]
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [advantage["baseline_R2"], advantage["baseline_within15"]/100],
       width, label="Baseline", color=BLUE, alpha=0.9, edgecolor="white")
ax.bar(x + width/2, [advantage["graph_R2"], advantage["graph_within15"]/100],
       width, label="Graph-enhanced", color=GREEN, alpha=0.9, edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(["R²", "Within 15% (frac.)"])
ax.set_title("Fit Quality (higher = better)")
ax.legend(fontsize=8)

# B3: Feature group importance (graph model)
ax = axes[2]
group_importance.sort_values().plot(kind="barh", ax=ax, color=PURPLE, alpha=0.85, edgecolor="white")
ax.set_title("Importance Share by Feature Group\n(Graph-enhanced model)")
ax.set_xlabel("Summed Importance")

plt.tight_layout()
plt.savefig("outputs/fig_graph_advantage.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_graph_advantage.png saved.")

# ── Figure C: Top-15 feature importance (graph model) ───────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
top_imp = importance_df.head(15)
group_colors = {"Node2Vec (source hub)": RED, "Node2Vec (destination hub)": AMBER,
                "Hub stats (non-graph)": PURPLE, "Distance/OSRM": BLUE,
                "Time": GREEN, "Categorical": "#64748B"}
bar_colors = [group_colors.get(g, "#64748B") for g in top_imp["group"].values[::-1]]
ax.barh(range(len(top_imp)), top_imp["importance"].values[::-1],
        color=bar_colors, edgecolor="white", alpha=0.9)
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels(top_imp["feature"].values[::-1], fontsize=9)
ax.set_xlabel("Feature Importance")
ax.set_title(f"Top 15 Features — {best_graph_name} (Graph-Enhanced)")
import matplotlib.patches as mpatches
legend_el = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()
             if g in top_imp["group"].values]
ax.legend(handles=legend_el, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/fig_graph_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_graph_feature_importance.png saved.")


# =============================================================================
# SECTION 14 — EXPORT
# =============================================================================

print("\nSECTION 14 — EXPORT")
print("=" * 65)

graph_results_df.to_csv("outputs/graph_enhanced_model_comparison.csv", index=False)
advantage_df.to_csv("outputs/graph_advantage_summary.csv", index=False)
importance_df.to_csv("outputs/graph_feature_importance.csv", index=False)
joblib.dump(best_graph_model, "outputs/best_graph_model.pkl")
pd.DataFrame(test_preds).to_csv("outputs/graph_test_predictions.csv", index=False)

print("  ✓ graph_enhanced_model_comparison.csv")
print("  ✓ graph_advantage_summary.csv")
print("  ✓ graph_feature_importance.csv")
print("  ✓ graph_test_predictions.csv")
print("  ✓ best_graph_model.pkl")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"""
{'='*65}
NODE2VEC GRAPH-ENHANCED MODEL COMPLETE
{'='*65}

  BEST GRAPH MODEL: {best_graph_name}
{graph_results_df.to_string(index=False)}

  GRAPH ADVANTAGE (vs {best_baseline_name}):
    MAE:        {advantage['baseline_MAE']:.2f}s → {advantage['graph_MAE']:.2f}s ({advantage['MAE_improvement_pct']:+.2f}%)
    RMSE:       {advantage['baseline_RMSE']:.2f}s → {advantage['graph_RMSE']:.2f}s ({advantage['RMSE_improvement_pct']:+.2f}%)
    R²:         {advantage['baseline_R2']:.4f} → {advantage['graph_R2']:.4f}
    Within 15%: {advantage['baseline_within15']:.1f}% → {advantage['graph_within15']:.1f}% ({advantage['within15_delta_pp']:+.1f} pp)

  {verdict}

  NEXT → Part 8: GraphSAGE-enhanced model — benchmark against BOTH the
          Part 6 baseline AND this Node2Vec model.
{'='*65}
""")
