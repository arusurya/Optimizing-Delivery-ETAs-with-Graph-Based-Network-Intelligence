"""
=============================================================================
PART 3: GRAPH CONSTRUCTION — DELHIVERY LOGISTICS NETWORK
=============================================================================
Role    : Graph ML Expert
Scope   : Build a weighted directed graph where nodes = facilities,
          edges = corridors, edge weights = median delay ratio
          stratified by route_type × time-of-day bucket.

INPUT : delivery_data.csv
OUTPUT: outputs/graph.pkl
        outputs/edge_metrics.csv
        outputs/node_metrics.csv
        outputs/fig_graph_analytics.png
        outputs/fig_hub_centrality.png
=============================================================================
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("=" * 65)
print("PART 3: GRAPH CONSTRUCTION — DELHIVERY LOGISTICS NETWORK")
print("=" * 65)

# =============================================================================
# SECTION 1 — LOAD & CLEAN DATA FROM REAL CSV
# =============================================================================
# NOTE: delivery_data.csv does NOT have lat/lng coordinate columns.
# All graph layout is done topologically (spring/shell layout), not
# geographically. This is the correct approach for this dataset.

print("\nSECTION 1 — LOADING & CLEANING")
print("=" * 65)

df = pd.read_csv("delivery_data.csv")
raw_n = len(df)
print(f"  Raw rows: {raw_n:,}")

df = df[df["actual_time"] > 0]               # drop negative durations
df = df[df["osrm_time"] > 0]                 # prevent ÷0
df = df[df["source_center"] != df["destination_center"]]  # no self-loops
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
df = df.dropna(subset=["trip_creation_time", "source_center", "destination_center"])

print(f"  Raw rows:   {raw_n:>10,}")
print(f"  Clean rows: {len(df):>10,}  (dropped {raw_n - len(df):,})")


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING
# =============================================================================

print("\nSECTION 2 — FEATURE ENGINEERING")
print("=" * 65)

df["delay_ratio"] = df["actual_time"] / df["osrm_time"]

p99 = df["delay_ratio"].quantile(0.99)
df["delay_ratio_capped"] = df["delay_ratio"].clip(upper=p99)

df["hour"] = df["trip_creation_time"].dt.hour
df["dow"]  = df["trip_creation_time"].dt.dayofweek

def assign_tod(h):
    if h < 6:   return "night"
    if h < 12:  return "morn_peak"
    if h < 17:  return "afternoon"
    return "eve_peak"

df["time_bucket"] = df["hour"].map(assign_tod)

print(f"  delay_ratio  — median: {df['delay_ratio'].median():.3f}  "
      f"mean: {df['delay_ratio'].mean():.3f}  "
      f"P99 cap: {p99:.2f}")
print(f"\n  Time bucket distribution:")
print(df["time_bucket"].value_counts().to_string())


# =============================================================================
# SECTION 3 — NODE ATTRIBUTES
# =============================================================================
# NOTE: No lat/lng columns exist in delivery_data.csv — coordinate-based
# geographic layout is not possible. Graph layout uses spring_layout.

print("\nSECTION 3 — COMPUTING NODE ATTRIBUTES")
print("=" * 65)

src_stats = (
    df.groupby("source_center")
    .agg(
        out_degree          = ("destination_center", "nunique"),
        out_trips           = ("trip_uuid", "count"),
        avg_delay_as_source = ("delay_ratio", "mean"),
        sla_breach_src      = ("delay_ratio", lambda x: (x > 1.2).mean()),
    )
    .rename_axis("hub")
    .reset_index()
)

dst_stats = (
    df.groupby("destination_center")
    .agg(
        in_degree         = ("source_center", "nunique"),
        in_trips          = ("trip_uuid", "count"),
        avg_delay_as_dest = ("delay_ratio", "mean"),
    )
    .rename_axis("hub")
    .reset_index()
)

node_df = (
    src_stats
    .merge(dst_stats, on="hub", how="outer")
    .fillna(0)
)
node_df["total_load"]    = node_df["out_trips"] + node_df["in_trips"]
node_df["load_balance"]  = (node_df["out_trips"] / (node_df["in_trips"] + 1)).round(3)
node_df["sla_breach_rate"] = node_df["sla_breach_src"]

print(f"  Nodes (unique hubs): {len(node_df)}")
print(f"  Sample node attributes:")
print(node_df.head(3).to_string(index=False))


# =============================================================================
# SECTION 4 — EDGE ATTRIBUTE COMPUTATION
# =============================================================================

print("\nSECTION 4 — COMPUTING EDGE ATTRIBUTES")
print("=" * 65)

edge_base = (
    df.groupby(["source_center", "destination_center"])
    .agg(
        trip_count          = ("trip_uuid",          "count"),
        median_actual_time  = ("actual_time",         "median"),
        median_osrm_time    = ("osrm_time",           "median"),
        weight              = ("delay_ratio_capped",  "median"),
        delay_ratio_iqr     = ("delay_ratio_capped",  lambda x: x.quantile(0.75) - x.quantile(0.25)),
        pct_sla_breach      = ("delay_ratio",         lambda x: (x > 1.2).mean()),
        avg_osrm_distance   = ("osrm_distance",       "mean"),
        pct_ftl             = ("route_type",          lambda x: (x == "FTL").mean()),
    )
    .reset_index()
)

strat = (
    df.groupby(["source_center", "destination_center", "route_type", "time_bucket"])
    ["delay_ratio_capped"].median()
    .reset_index()
)
strat["col"] = "delay_" + strat["route_type"] + "_" + strat["time_bucket"]
strat_pivot = strat.pivot_table(
    index=["source_center", "destination_center"],
    columns="col",
    values="delay_ratio_capped",
    aggfunc="first"
).reset_index()
strat_pivot.columns.name = None

edge_df = edge_base.merge(strat_pivot, on=["source_center", "destination_center"], how="left")

MIN_SUPPORT = 10
edge_df = edge_df[edge_df["trip_count"] >= MIN_SUPPORT].reset_index(drop=True)
edge_df["is_chronic_delay"] = (edge_df["weight"] > 1.2).astype(int)

print(f"  Corridors (edges) after min-support filter: {len(edge_df):,}")
print(f"  Chronic delay corridors (weight > 1.2):     "
      f"{edge_df['is_chronic_delay'].sum():,} "
      f"({edge_df['is_chronic_delay'].mean()*100:.1f}%)")
print(f"\n  Edge weight summary:")
print(edge_df["weight"].describe().round(3).to_string())
strat_cols = [c for c in edge_df.columns if c.startswith("delay_")]
print(f"\n  Stratified columns available: {strat_cols}")


# =============================================================================
# SECTION 5 — BUILD THE NETWORKX GRAPH
# =============================================================================

print("\nSECTION 5 — BUILDING NETWORKX DIRECTED GRAPH")
print("=" * 65)

G = nx.DiGraph()

# Add nodes (no lat/lng — dataset does not have coordinates)
for _, row in node_df.iterrows():
    G.add_node(
        row["hub"],
        in_degree_trips     = int(row["in_trips"]),
        out_degree_trips    = int(row["out_trips"]),
        total_load          = int(row["total_load"]),
        in_degree           = int(row["in_degree"]),
        out_degree          = int(row["out_degree"]),
        load_balance        = float(row["load_balance"]),
        avg_delay_as_source = float(row["avg_delay_as_source"]),
        avg_delay_as_dest   = float(row["avg_delay_as_dest"]),
        sla_breach_rate     = float(row["sla_breach_rate"]),
    )

# Add edges
for _, row in edge_df.iterrows():
    attrs = {
        "weight"            : float(row["weight"]),
        "trip_count"        : int(row["trip_count"]),
        "median_actual_time": float(row["median_actual_time"]),
        "median_osrm_time"  : float(row["median_osrm_time"]),
        "delay_ratio_iqr"   : float(row["delay_ratio_iqr"]),
        "pct_sla_breach"    : float(row["pct_sla_breach"]),
        "avg_osrm_distance" : float(row["avg_osrm_distance"]),
        "pct_ftl"           : float(row["pct_ftl"]),
        "is_chronic_delay"  : int(row["is_chronic_delay"]),
    }
    for col in strat_cols:
        attrs[col] = float(row[col]) if not pd.isna(row[col]) else float(row["weight"])
    G.add_edge(row["source_center"], row["destination_center"], **attrs)

print(f"  Graph built:")
print(f"    Nodes (hubs):      {G.number_of_nodes():>6,}")
print(f"    Edges (corridors): {G.number_of_edges():>6,}")
print(f"    Is directed:       {G.is_directed()}")
print(f"    Is connected:      {nx.is_weakly_connected(G)}")
print(f"    Density:           {nx.density(G):.6f}")


# =============================================================================
# SECTION 6 — GRAPH VALIDATION
# =============================================================================

print("\nSECTION 6 — GRAPH VALIDATION")
print("=" * 65)

errors = []

missing_nodes = (
    set(edge_df["source_center"]) | set(edge_df["destination_center"])
) - set(G.nodes())
if missing_nodes:
    errors.append(f"Missing nodes: {missing_nodes}")

self_loops = list(nx.selfloop_edges(G))
if self_loops:
    errors.append(f"Self-loops found: {self_loops}")

zero_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] <= 0]
if zero_edges:
    errors.append(f"Zero-weight edges: {zero_edges}")

components = list(nx.weakly_connected_components(G))
if len(components) > 1:
    errors.append(f"Graph has {len(components)} disconnected components")

if errors:
    print("  ✗ VALIDATION ISSUES:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  ✓ All validation checks passed")

w_arr = np.array([d["weight"] for _, _, d in G.edges(data=True)])
print(f"\n  Edge weight sanity:")
print(f"    Min:    {w_arr.min():.3f}  (should be > 0)")
print(f"    Median: {np.median(w_arr):.3f}")
print(f"    Max:    {w_arr.max():.3f}  (P99 cap = {p99:.2f})")
print(f"    % edges with weight > 1.2:  {(w_arr > 1.2).mean()*100:.1f}%")

degrees = dict(G.degree())
deg_vals = list(degrees.values())
print(f"\n  Hub degree stats:")
print(f"    Min degree: {min(deg_vals)}  Max degree: {max(deg_vals)}  "
      f"Median: {np.median(deg_vals):.0f}")


# =============================================================================
# SECTION 7 — GRAPH METRICS
# =============================================================================

print("\nSECTION 7 — GRAPH METRICS")
print("=" * 65)

print("  Computing betweenness centrality (unweighted)...")
bc = nx.betweenness_centrality(G, normalized=True)
nx.set_node_attributes(G, bc, "betweenness_centrality")

print("  Computing weighted betweenness centrality...")
bc_w = nx.betweenness_centrality(G, weight="weight", normalized=True)
nx.set_node_attributes(G, bc_w, "betweenness_centrality_weighted")

print("  Computing PageRank...")
pr = nx.pagerank(G, alpha=0.85, weight="weight")
nx.set_node_attributes(G, pr, "pagerank")

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
nx.set_node_attributes(G, in_deg,  "graph_in_degree")
nx.set_node_attributes(G, out_deg, "graph_out_degree")

top_bc = sorted(bc.items(), key=lambda x: -x[1])[:10]
print(f"\n  Top 10 hubs by betweenness centrality:")
print(f"  {'Hub':<20} {'Betweenness':>12} {'PageRank':>10} {'SLA Breach%':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*12}")
for hub, val in top_bc:
    sla = G.nodes[hub].get("sla_breach_rate", 0) * 100
    pgr = pr[hub]
    print(f"  {hub:<20} {val:>12.4f} {pgr:>10.4f} {sla:>11.1f}%")


# =============================================================================
# SECTION 8 — BOTTLENECK IDENTIFICATION
# =============================================================================

print("\nSECTION 8 — BOTTLENECK IDENTIFICATION")
print("=" * 65)

metric_df = pd.DataFrame({
    "hub"            : list(bc.keys()),
    "betweenness"    : list(bc.values()),
    "pagerank"       : [pr[h] for h in bc.keys()],
    "sla_breach_rate": [G.nodes[h].get("sla_breach_rate", 0) for h in bc.keys()],
    "avg_delay"      : [G.nodes[h].get("avg_delay_as_source", 1) for h in bc.keys()],
    "total_load"     : [G.nodes[h].get("total_load", 0) for h in bc.keys()],
})

for col in ["betweenness", "sla_breach_rate", "avg_delay", "total_load"]:
    r = metric_df[col].max() - metric_df[col].min()
    metric_df[f"{col}_norm"] = (metric_df[col] - metric_df[col].min()) / (r if r else 1)

metric_df["bottleneck_score"] = (
    0.40 * metric_df["betweenness_norm"]
  + 0.35 * metric_df["avg_delay_norm"]
  + 0.25 * metric_df["total_load_norm"]
)

nx.set_node_attributes(G, metric_df.set_index("hub")["bottleneck_score"].to_dict(),
                       "bottleneck_score")

top5 = metric_df.nlargest(5, "bottleneck_score")
print(f"  TOP 5 BOTTLENECK HUBS (from real data):")
print(f"  {'Hub':<20} {'Score':>6} {'Betweenness':>12} {'Avg Delay':>10} "
      f"{'SLA Breach%':>12} {'Load':>8}")
print(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*10} {'-'*12} {'-'*8}")
for _, row in top5.iterrows():
    print(f"  {row['hub']:<20} {row['bottleneck_score']:>6.3f} "
          f"{row['betweenness']:>12.4f} {row['avg_delay']:>10.3f} "
          f"{row['sla_breach_rate']*100:>11.1f}% {row['total_load']:>8.0f}")

chronic = [(u, v, d) for u, v, d in G.edges(data=True) if d["is_chronic_delay"]]
print(f"\n  Chronic delay corridors (weight > 1.2): {len(chronic)}")


# =============================================================================
# SECTION 9 — VISUALIZATIONS (Topological — no lat/lng in dataset)
# =============================================================================

print("\nSECTION 9 — GENERATING VISUALIZATIONS")
print("=" * 65)

# ── Figure A: Topological Network Graph ───────────────────────────────────────
# Since delivery_data.csv has no coordinates, use spring layout.
# Visualise a subgraph of the most connected hubs to keep it readable.

top_hubs_by_load = metric_df.nlargest(60, "total_load")["hub"].tolist()
sub = G.subgraph(top_hubs_by_load)

print("  Computing spring layout (may take a moment)...")
pos = nx.spring_layout(sub, k=2.5, iterations=50, seed=42, weight="weight")

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0F172A")
ax.set_facecolor("#0F172A")

sub_edges = list(sub.edges(data=True))
if sub_edges:
    edge_weights  = np.array([d["weight"] for _, _, d in sub_edges])
    edge_counts   = np.array([d["trip_count"] for _, _, d in sub_edges])
    edge_widths   = 0.3 + (edge_counts / max(edge_counts.max(), 1)) * 2.0
    norm_e        = mcolors.Normalize(vmin=0.8, vmax=2.2)
    cmap_e        = plt.cm.RdYlGn_r
    edge_colors   = [cmap_e(norm_e(w)) for w in edge_weights]

    for (u, v, d), col, wid in zip(sub_edges, edge_colors, edge_widths):
        if u in pos and v in pos:
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            ax.plot(x_vals, y_vals, color=col, alpha=0.35, linewidth=wid, zorder=1)

load_vals  = np.array([G.nodes[n].get("total_load", 1) for n in sub.nodes()])
node_sizes = 40 + (load_vals / max(load_vals.max(), 1)) * 300
node_colors = [plt.cm.plasma(min(G.nodes[n].get("bottleneck_score", 0) * 1.5, 1.0))
               for n in sub.nodes()]

xs = [pos[n][0] for n in sub.nodes()]
ys = [pos[n][1] for n in sub.nodes()]
ax.scatter(xs, ys, s=node_sizes, c=node_colors, zorder=3, alpha=0.9,
           edgecolors="#FFFFFF", linewidths=0.3)

for _, row in top5.iterrows():
    if row["hub"] in pos:
        ax.annotate(row["hub"], xy=pos[row["hub"]], xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=8, color="#F97316", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B",
                              edgecolor="#F97316", alpha=0.85))

sm = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Edge Weight (Delay Ratio)", color="white", fontsize=10)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

legend_elements = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#F97316",
           markersize=10, label="Top Bottleneck Hub"),
    Line2D([0], [0], color="#16A34A", lw=2, label="Low Delay (<1.0x)"),
    Line2D([0], [0], color="#EAB308", lw=2, label="Marginal (1.0–1.2x)"),
    Line2D([0], [0], color="#DC2626", lw=2, label="Chronic Delay (>1.2x)"),
]
ax.legend(handles=legend_elements, loc="lower right",
          facecolor="#1E293B", edgecolor="#475569",
          labelcolor="white", fontsize=9)
ax.set_title("Delhivery Logistics Network (Top 60 Hubs by Load)\n"
             "Node size = Hub Load | Edge color = Delay Severity | Orange = Bottleneck",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="white")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(output_dir / "fig_graph_network.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓ Topological network map saved.")


# ── Figure B: Bottleneck Quadrant + Edge Weight Distribution ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Graph Analytics Dashboard", fontsize=14, fontweight="bold")

ax = axes[0]
sc = ax.scatter(metric_df["betweenness"], metric_df["avg_delay"],
                c=metric_df["bottleneck_score"], cmap="plasma",
                s=metric_df["total_load_norm"] * 200 + 20, alpha=0.75)
ax.axhline(1.2, color="red", lw=1.5, ls="--", alpha=0.7, label="SLA threshold")
ax.axvline(metric_df["betweenness"].quantile(0.8), color="gray",
           lw=1.5, ls=":", alpha=0.7)
plt.colorbar(sc, ax=ax, label="Bottleneck Score")
for _, row in top5.iterrows():
    ax.annotate(row["hub"], (row["betweenness"], row["avg_delay"]),
                fontsize=7, color="red", fontweight="bold",
                xytext=(4, 3), textcoords="offset points")
ax.set_title("Bottleneck Quadrant\n(Betweenness vs Delay | Size = Load)")
ax.set_xlabel("Betweenness Centrality")
ax.set_ylabel("Avg Delay Ratio")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(edge_df["weight"], bins=80, color="#2563EB", edgecolor="none", alpha=0.85)
ax.axvline(1.0, color="green", lw=2, ls="--", label="No delay (1.0x)")
ax.axvline(1.2, color="red",   lw=2, ls="--", label="SLA threshold (1.2x)")
ax.axvline(edge_df["weight"].median(), color="orange", lw=2, ls="-",
           label=f"Median ({edge_df['weight'].median():.2f}x)")
ax.set_title("Edge Weight Distribution\n(Median Delay Ratio per Corridor)")
ax.set_xlabel("Edge Weight (Delay Ratio)")
ax.set_ylabel("Corridor Count")
ax.legend(fontsize=8)

ax = axes[2]
strat_order = ["delay_FTL_night", "delay_FTL_morn_peak",
               "delay_FTL_afternoon", "delay_FTL_eve_peak",
               "delay_Carting_night", "delay_Carting_morn_peak",
               "delay_Carting_afternoon", "delay_Carting_eve_peak"]
avail = [c for c in strat_order if c in edge_df.columns]
means  = [edge_df[c].dropna().mean() for c in avail]
labels = [c.replace("delay_", "").replace("_", "\n") for c in avail]
colors = ["#16A34A" if "FTL" in c else "#DC2626" for c in avail]
bars = ax.bar(range(len(avail)), means, color=colors, edgecolor="white", alpha=0.85)
ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
ax.axhline(1.2, color="black", lw=1.5, ls="--")
ax.set_xticks(range(len(avail)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_title("Mean Edge Weight by\nRoute Type × Time Bucket")
ax.set_ylabel("Mean Delay Ratio")
legend_el = [mpatches.Patch(color="#16A34A", label="FTL"),
             mpatches.Patch(color="#DC2626", label="Carting")]
ax.legend(handles=legend_el, fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "fig_graph_analytics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Graph analytics dashboard saved.")


# ── Figure C: Hub centrality ranking ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Hub Centrality Rankings (Real Data)", fontsize=13, fontweight="bold")

top20_bc = metric_df.nlargest(20, "betweenness")
ax = axes[0]
ax.barh(range(len(top20_bc)), top20_bc["betweenness"].values[::-1],
        color="#2563EB", edgecolor="white", alpha=0.85)
ax.set_yticks(range(len(top20_bc)))
ax.set_yticklabels(top20_bc["hub"].values[::-1], fontsize=8)
ax.set_title("Top 20 Hubs: Betweenness Centrality")
ax.set_xlabel("Betweenness Centrality")

top20_pr = metric_df.nlargest(20, "pagerank")
ax = axes[1]
ax.barh(range(len(top20_pr)), top20_pr["pagerank"].values[::-1],
        color="#7C3AED", edgecolor="white", alpha=0.85)
ax.set_yticks(range(len(top20_pr)))
ax.set_yticklabels(top20_pr["hub"].values[::-1], fontsize=8)
ax.set_title("Top 20 Hubs: PageRank")
ax.set_xlabel("PageRank Score")

plt.tight_layout()
plt.savefig(output_dir / "fig_hub_centrality.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Hub centrality chart saved.")


# =============================================================================
# SECTION 10 — EXPORT GRAPH + FEATURE MATRICES
# =============================================================================

print("\nSECTION 10 — EXPORTING GRAPH DATA")
print("=" * 65)

# Save graph object
with open(output_dir / "graph.pkl", "wb") as f:
    pickle.dump(G, f)
print("  ✓ graph.pkl saved")

# Node metrics CSV
node_metrics = metric_df[["hub", "betweenness", "pagerank", "bottleneck_score",
                           "sla_breach_rate", "avg_delay", "total_load"]].merge(
    node_df[["hub", "in_degree", "out_degree", "load_balance",
             "avg_delay_as_source", "avg_delay_as_dest"]], on="hub", how="left"
)
node_metrics.to_csv(output_dir / "node_metrics.csv", index=False)
print(f"  ✓ node_metrics.csv  — {len(node_metrics)} rows")

# Edge metrics CSV
edge_df.to_csv(output_dir / "edge_metrics.csv", index=False)
print(f"  ✓ edge_metrics.csv  — {len(edge_df)} rows")

print(f"""
{'='*65}
GRAPH CONSTRUCTION COMPLETE
{'='*65}
  Nodes:             {G.number_of_nodes()}
  Edges:             {G.number_of_edges()}
  Top bottleneck:    {top5.iloc[0]['hub']}  (score={top5.iloc[0]['bottleneck_score']:.3f})
  Chronic corridors: {edge_df['is_chronic_delay'].sum()}  ({edge_df['is_chronic_delay'].mean()*100:.1f}%)
  Edge weight range: [{w_arr.min():.2f}, {w_arr.max():.2f}]
{'='*65}
""")
