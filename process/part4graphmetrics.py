"""
=============================================================================
PART 4: GRAPH METRICS, BOTTLENECK SCORING & CORRIDOR AUDIT
=============================================================================
Role  : Network Science Specialist
Scope : Compute all 7 centrality/graph metrics, rank bottleneck hubs,
        rank chronic corridors, and build a composite Bottleneck Risk Score.

INPUT : delivery_data.csv
OUTPUT: outputs/hub_metrics.csv
        outputs/top_bottleneck_hubs.csv
        outputs/network_metrics_dashboard.png
        outputs/bottleneck_dashboard.png
=============================================================================
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.stats import rankdata
from pathlib import Path
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

RED    = "#E63329"
BLUE   = "#2563EB"
GREEN  = "#16A34A"
AMBER  = "#D97706"
PURPLE = "#7C3AED"
DARK   = "#0F172A"
SLATE  = "#1E293B"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#CBD5E1",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})


# =============================================================================
# SECTION 0 — LOAD & CLEAN DATA FROM REAL CSV
# Build graph identically to Part 3.
# NOTE: delivery_data.csv has NO lat/lng columns — removed from pipeline.
# =============================================================================

print("=" * 65)
print("PART 4: GRAPH METRICS & BOTTLENECK SCORING")
print("=" * 65)

df = pd.read_csv("delivery_data.csv")

raw_n = len(df)
df = df[(df["actual_time"] > 0) & (df["osrm_time"] > 0)]
df = df[df["source_center"] != df["destination_center"]]
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
df = df.dropna(subset=["trip_creation_time", "source_center", "destination_center"])
df["delay_ratio"] = df["actual_time"] / df["osrm_time"]
p99 = df["delay_ratio"].quantile(0.99)
df["delay_ratio_capped"] = df["delay_ratio"].clip(upper=p99)
df["hour"] = df["trip_creation_time"].dt.hour

def tod(h):
    return ("night" if h < 6 else "morn_peak" if h < 12
            else "afternoon" if h < 17 else "eve_peak")
df["time_bucket"] = df["hour"].map(tod)
df["sla_breach"]  = (df["delay_ratio"] > 1.20).astype(int)

MIN_SUPPORT = 10

# Edge table
edge_df = (
    df.groupby(["source_center", "destination_center"])
    .agg(
        trip_count         = ("trip_uuid",          "count"),
        weight             = ("delay_ratio_capped",  "median"),
        pct_sla_breach     = ("sla_breach",          "mean"),
        median_actual_time = ("actual_time",         "median"),
        median_osrm_time   = ("osrm_time",           "median"),
        delay_iqr          = ("delay_ratio_capped",  lambda x: x.quantile(0.75)-x.quantile(0.25)),
        avg_distance       = ("osrm_distance",       "mean"),
    )
    .query(f"trip_count >= {MIN_SUPPORT}")
    .reset_index()
)
edge_df["is_chronic"] = (edge_df["weight"] > 1.20).astype(int)

# Node table (no lat/lng)
node_src = df.groupby("source_center").agg(
    out_trips   =("trip_uuid","count"),
    avg_delay_s =("delay_ratio","mean"),
    sla_rate    =("sla_breach","mean"),
).rename_axis("hub").reset_index()

node_dst = df.groupby("destination_center").agg(
    in_trips    =("trip_uuid","count"),
    avg_delay_d =("delay_ratio","mean"),
).rename_axis("hub").reset_index()

node_df = node_src.merge(node_dst, on="hub", how="outer").fillna(0)
node_df["total_load"] = node_df["out_trips"] + node_df["in_trips"]

# Build directed graph
G = nx.DiGraph()
for _, r in node_df.iterrows():
    G.add_node(r["hub"],
               total_load=r["total_load"],
               out_trips=r["out_trips"],
               in_trips=r["in_trips"],
               sla_rate=r["sla_rate"],
               avg_delay_src=r["avg_delay_s"],
               avg_delay_dst=r["avg_delay_d"])

for _, r in edge_df.iterrows():
    G.add_edge(r["source_center"], r["destination_center"],
               weight=r["weight"], trip_count=r["trip_count"],
               pct_sla_breach=r["pct_sla_breach"],
               median_actual_time=r["median_actual_time"],
               median_osrm_time=r["median_osrm_time"],
               delay_iqr=r["delay_iqr"], avg_distance=r["avg_distance"],
               is_chronic=r["is_chronic"])

print(f"  Graph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
print(f"  Clean rows: {len(df):,}  (dropped {raw_n - len(df):,})")


# =============================================================================
# SECTION 1 — COMPUTE ALL 7 GRAPH METRICS
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 1 — COMPUTING 7 GRAPH METRICS")
print("=" * 65)

print("\n  [1] Betweenness Centrality...")
bc_unw = nx.betweenness_centrality(G, normalized=True, weight=None)
bc_w   = nx.betweenness_centrality(G, normalized=True, weight="weight")
nx.set_node_attributes(G, bc_unw, "bc_unweighted")
nx.set_node_attributes(G, bc_w,   "bc_weighted")
print(f"      Max BC (unweighted): {max(bc_unw.values()):.4f}  "
      f"hub={max(bc_unw, key=bc_unw.get)}")
print(f"      Max BC (weighted):   {max(bc_w.values()):.4f}  "
      f"hub={max(bc_w, key=bc_w.get)}")

print("\n  [2] Degree Centrality...")
dc = nx.degree_centrality(G)
nx.set_node_attributes(G, dc, "degree_centrality")
print(f"      Max DC: {max(dc.values()):.4f}  hub={max(dc, key=dc.get)}")

print("\n  [3] In-Degree...")
in_deg  = dict(G.in_degree())
in_deg_c = {n: v/(G.number_of_nodes()-1) for n, v in in_deg.items()}
nx.set_node_attributes(G, in_deg,   "in_degree")
nx.set_node_attributes(G, in_deg_c, "in_degree_centrality")
print(f"      Max in-degree: {max(in_deg.values())}  "
      f"hub={max(in_deg, key=in_deg.get)}")

print("\n  [4] Out-Degree...")
out_deg  = dict(G.out_degree())
out_deg_c = {n: v/(G.number_of_nodes()-1) for n, v in out_deg.items()}
nx.set_node_attributes(G, out_deg,   "out_degree")
nx.set_node_attributes(G, out_deg_c, "out_degree_centrality")
print(f"      Max out-degree: {max(out_deg.values())}  "
      f"hub={max(out_deg, key=out_deg.get)}")

print("\n  [5] Closeness Centrality...")
cc = nx.closeness_centrality(G, wf_improved=True)
nx.set_node_attributes(G, cc, "closeness_centrality")
print(f"      Max CC: {max(cc.values()):.4f}  hub={max(cc, key=cc.get)}")

print("\n  [6] Clustering Coefficient...")
clust_dir = nx.clustering(G)
nx.set_node_attributes(G, clust_dir, "clustering")
avg_clust = np.mean(list(clust_dir.values()))
print(f"      Avg clustering (directed): {avg_clust:.4f}")

print("\n  [7] PageRank...")
pr_unw = nx.pagerank(G, alpha=0.85, weight=None)
pr_w   = nx.pagerank(G, alpha=0.85, weight="weight")
nx.set_node_attributes(G, pr_unw, "pagerank")
nx.set_node_attributes(G, pr_w,   "pagerank_weighted")
print(f"      Max PR (unweighted): {max(pr_unw.values()):.5f}  "
      f"hub={max(pr_unw, key=pr_unw.get)}")


# =============================================================================
# SECTION 2 — MASTER METRIC TABLE
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 2 — MASTER METRIC TABLE")
print("=" * 65)

metric_df = pd.DataFrame({
    "hub"              : list(bc_unw.keys()),
    "bc_unweighted"    : list(bc_unw.values()),
    "bc_weighted"      : [bc_w[h]       for h in bc_unw],
    "degree_centrality": [dc[h]         for h in bc_unw],
    "in_degree"        : [in_deg[h]     for h in bc_unw],
    "out_degree"       : [out_deg[h]    for h in bc_unw],
    "closeness"        : [cc[h]         for h in bc_unw],
    "clustering"       : [clust_dir[h]  for h in bc_unw],
    "pagerank"         : [pr_unw[h]     for h in bc_unw],
    "pagerank_weighted": [pr_w[h]       for h in bc_unw],
    "avg_delay"        : [G.nodes[h].get("avg_delay_src", 1) for h in bc_unw],
    "sla_breach_rate"  : [G.nodes[h].get("sla_rate", 0)     for h in bc_unw],
    "total_load"       : [G.nodes[h].get("total_load", 0)   for h in bc_unw],
})

print(f"\n  Metrics computed for {len(metric_df)} hubs.")
print(f"\n  Top 3 hubs by each metric:")
for col in ["bc_unweighted", "degree_centrality", "in_degree", "closeness", "pagerank"]:
    top3 = metric_df.nlargest(3, col)[["hub", col]].values
    row  = "  ".join([f"{h}({v:.4f})" for h, v in top3])
    print(f"    {col:<22}: {row}")


# =============================================================================
# SECTION 3 — COMPOSITE BOTTLENECK RISK SCORE (BRS)
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 3 — COMPOSITE BOTTLENECK RISK SCORE (BRS)")
print("=" * 65)

def minmax(series):
    r = series.max() - series.min()
    return (series - series.min()) / r if r > 0 else pd.Series(0.0, index=series.index)

metric_df["delay_norm"] = minmax(metric_df["avg_delay"])
metric_df["bc_norm"]    = minmax(metric_df["bc_weighted"])
metric_df["load_norm"]  = minmax(metric_df["total_load"])
metric_df["sla_norm"]   = minmax(metric_df["sla_breach_rate"])

WEIGHTS = {"delay": 0.35, "bc": 0.30, "load": 0.20, "sla": 0.15}

metric_df["BRS"] = (
    WEIGHTS["delay"] * metric_df["delay_norm"]
  + WEIGHTS["bc"]    * metric_df["bc_norm"]
  + WEIGHTS["load"]  * metric_df["load_norm"]
  + WEIGHTS["sla"]   * metric_df["sla_norm"]
)
metric_df["BRS_percentile"] = rankdata(metric_df["BRS"]) / len(metric_df) * 100
metric_df["risk_tier"] = pd.cut(
    metric_df["BRS_percentile"],
    bins=[0, 70, 85, 95, 100],
    labels=["Standard", "Elevated", "High", "Critical"]
)

nx.set_node_attributes(G, metric_df.set_index("hub")["BRS"].to_dict(), "BRS")
nx.set_node_attributes(G, metric_df.set_index("hub")["risk_tier"].astype(str).to_dict(),
                       "risk_tier")

print(f"\n  BRS Weight Schema: Delay(35%) | Betweenness(30%) | Volume(20%) | SLA(15%)")
print(f"\n  Risk Tier Distribution:")
print(metric_df["risk_tier"].value_counts().sort_index().to_string())


# =============================================================================
# SECTION 4 — TOP BOTTLENECK HUBS
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 4 — TOP 10 BOTTLENECK HUBS (REAL DATA)")
print("=" * 65)

top10 = metric_df.nlargest(10, "BRS")[[
    "hub", "BRS", "risk_tier", "avg_delay", "sla_breach_rate",
    "bc_weighted", "total_load", "in_degree", "out_degree",
    "closeness", "pagerank"
]].reset_index(drop=True)
top10.index += 1

print(f"\n  {'#':<3} {'Hub':<25} {'BRS':>5} {'Tier':<10} "
      f"{'Avg Delay':>9} {'SLA Breach':>10} {'Load':>7}")
print(f"  {'-'*3} {'-'*25} {'-'*5} {'-'*10} {'-'*9} {'-'*10} {'-'*7}")
for rank, row in top10.iterrows():
    tier_sym = {"Critical":"🔴","High":"🟠","Elevated":"🟡","Standard":"🟢"}.get(
        str(row["risk_tier"]), "⚪")
    print(f"  {rank:<3} {row['hub']:<25} {row['BRS']:>5.3f} "
          f"{tier_sym}{str(row['risk_tier']):<9} "
          f"{row['avg_delay']:>9.3f} {row['sla_breach_rate']*100:>9.1f}% "
          f"{int(row['total_load']):>7,}")

top10["sla_breach_trips_est"] = (top10["total_load"] * top10["sla_breach_rate"]).astype(int)
top10["revenue_at_risk_est"]  = (top10["sla_breach_trips_est"] * 100).astype(int)

print(f"\n  SLA BREACH IMPACT (Top 10 Hubs, ₹100/breach assumption):")
for _, row in top10.iterrows():
    print(f"  {row['hub']:<25} {int(row['sla_breach_trips_est']):>10,} trips  "
          f"₹{int(row['revenue_at_risk_est']):>10,}")
print(f"\n  Total estimated revenue at risk: ₹{top10['revenue_at_risk_est'].sum():,}")


# =============================================================================
# SECTION 5 — TOP 20 DELAYED CORRIDORS
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 5 — TOP 20 CHRONICALLY DELAYED CORRIDORS")
print("=" * 65)

edge_df["delay_severity_norm"]  = minmax(edge_df["weight"])
edge_df["volume_norm"]          = minmax(edge_df["trip_count"])
edge_df["sla_breach_corr_norm"] = minmax(edge_df["pct_sla_breach"])
edge_df["variability_norm"]     = minmax(edge_df["delay_iqr"])

edge_df["corridor_risk_score"] = (
    0.40 * edge_df["delay_severity_norm"]
  + 0.25 * edge_df["volume_norm"]
  + 0.25 * edge_df["sla_breach_corr_norm"]
  + 0.10 * edge_df["variability_norm"]
)

top20_corr = edge_df.nlargest(20, "corridor_risk_score")[[
    "source_center", "destination_center", "weight", "trip_count",
    "pct_sla_breach", "delay_iqr", "avg_distance", "corridor_risk_score"
]].reset_index(drop=True)
top20_corr["corridor"] = top20_corr["source_center"] + " → " + top20_corr["destination_center"]
top20_corr.index += 1

print(f"\n  {'#':<3} {'Corridor':<35} {'Delay Ratio':>11} {'Trips':>7} "
      f"{'SLA Breach':>10} {'Risk':>6}")
print(f"  {'-'*3} {'-'*35} {'-'*11} {'-'*7} {'-'*10} {'-'*6}")
for rank, row in top20_corr.iterrows():
    corr_label = row['corridor'][:33]
    print(f"  {rank:<3} {corr_label:<35} {row['weight']:>11.3f} "
          f"{int(row['trip_count']):>7,} {row['pct_sla_breach']*100:>9.1f}% "
          f"{row['corridor_risk_score']:>6.3f}")


# =============================================================================
# SECTION 6 — VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 6 — GENERATING VISUALIZATIONS")
print("=" * 65)

# ── Figure 1: Network Metrics Dashboard ───────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.suptitle("Delhivery Network — Graph Metrics & Bottleneck Risk Scoring",
             fontsize=15, fontweight="bold", color=DARK, y=1.01)
gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a: BRS bar chart top 20
ax = fig.add_subplot(gs_fig[0, 0])
top20_brs = metric_df.nlargest(20, "BRS")
tier_colors = {"Critical": RED, "High": AMBER, "Elevated": PURPLE, "Standard": BLUE}
bar_colors  = [tier_colors.get(str(t), BLUE) for t in top20_brs["risk_tier"].values[::-1]]
ax.barh(range(20), top20_brs["BRS"].values[::-1],
        color=bar_colors, edgecolor="white", alpha=0.9)
ax.set_yticks(range(20))
ax.set_yticklabels(top20_brs["hub"].values[::-1], fontsize=8)
ax.set_title("Top 20 Hubs: Composite BRS\n(🔴Critical 🟠High 🟡Elevated)")
ax.set_xlabel("Bottleneck Risk Score")
ax.axvline(metric_df["BRS"].quantile(0.95), color="black", lw=1.5,
           ls="--", alpha=0.6, label="P95 threshold")
ax.legend(fontsize=8)

# 1b: BRS component decomposition (stacked bar, top 10)
ax = fig.add_subplot(gs_fig[0, 1])
top10_plot = metric_df.nlargest(10, "BRS")
components = {
    "Delay (35%)":       WEIGHTS["delay"] * top10_plot["delay_norm"].values,
    "Betweenness (30%)": WEIGHTS["bc"]    * top10_plot["bc_norm"].values,
    "Volume (20%)":      WEIGHTS["load"]  * top10_plot["load_norm"].values,
    "SLA (15%)":         WEIGHTS["sla"]   * top10_plot["sla_norm"].values,
}
comp_colors = [RED, BLUE, GREEN, AMBER]
bottom = np.zeros(10)
for (label, vals), col in zip(components.items(), comp_colors):
    ax.barh(range(10), vals[::-1], left=bottom[::-1],
            label=label, color=col, alpha=0.85, edgecolor="white")
    bottom += vals
ax.set_yticks(range(10))
ax.set_yticklabels(top10_plot["hub"].values[::-1], fontsize=8)
ax.set_title("BRS Decomposition\n(Top 10 Hubs — Stacked by Component)")
ax.set_xlabel("BRS Component Contribution")
ax.legend(fontsize=7, loc="lower right")

# 1c: Betweenness vs Avg Delay (bottleneck quadrant)
ax = fig.add_subplot(gs_fig[0, 2])
sc = ax.scatter(metric_df["bc_weighted"], metric_df["avg_delay"],
                c=metric_df["BRS"], cmap="plasma",
                s=metric_df["load_norm"] * 200 + 20, alpha=0.75)
ax.axhline(1.2, color=RED, lw=1.5, ls="--", alpha=0.7, label="SLA threshold")
ax.axvline(metric_df["bc_weighted"].quantile(0.8), color="gray",
           lw=1.5, ls=":", alpha=0.7)
plt.colorbar(sc, ax=ax, label="BRS")
for _, row in metric_df.nlargest(5, "BRS").iterrows():
    ax.annotate(row["hub"], (row["bc_weighted"], row["avg_delay"]),
                fontsize=7, color=RED, fontweight="bold",
                xytext=(4, 3), textcoords="offset points")
ax.set_title("Bottleneck Quadrant\n(Betweenness vs Delay | Size = Load)")
ax.set_xlabel("Betweenness Centrality (weighted)")
ax.set_ylabel("Avg Delay Ratio")
ax.legend(fontsize=8)

# 1d: 7-metric radar for top 5 hubs
ax = fig.add_subplot(gs_fig[1, 0], polar=True)
radar_metrics = ["bc_unweighted", "degree_centrality", "closeness",
                 "clustering", "pagerank", "delay_norm", "sla_norm"]
radar_labels  = ["Betweenness", "Degree", "Closeness",
                 "Clustering", "PageRank", "Delay", "SLA"]
n_m = len(radar_metrics)
angles = [n / float(n_m) * 2 * np.pi for n in range(n_m)]
angles += angles[:1]

radar_cols = [RED, BLUE, GREEN, AMBER, PURPLE]
for i, (_, row) in enumerate(metric_df.nlargest(5, "BRS").iterrows()):
    vals = []
    for m in radar_metrics:
        col_max = metric_df[m].max()
        vals.append(row[m] / col_max if col_max > 0 else 0)
    vals += vals[:1]
    ax.plot(angles, vals, color=radar_cols[i], lw=2, alpha=0.8, label=row["hub"])
    ax.fill(angles, vals, color=radar_cols[i], alpha=0.08)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, size=8)
ax.set_ylim(0, 1)
ax.set_title("7-Metric Radar\n(Top 5 Bottleneck Hubs)", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)

# 1e: In-degree vs Out-degree flow balance
ax = fig.add_subplot(gs_fig[1, 1])
sc2 = ax.scatter(metric_df["in_degree"], metric_df["out_degree"],
                 c=metric_df["avg_delay"], cmap="RdYlGn_r",
                 s=metric_df["load_norm"] * 200 + 20, alpha=0.75,
                 vmin=0.9, vmax=2.2)
plt.colorbar(sc2, ax=ax, label="Avg Delay")
max_d = max(metric_df["in_degree"].max(), metric_df["out_degree"].max())
ax.plot([0, max_d], [0, max_d], "k--", lw=1.5, alpha=0.5, label="Balanced")
ax.set_title("Hub Flow Balance\n(In-Degree vs Out-Degree | Color=Delay)")
ax.set_xlabel("In-Degree (inbound corridors)")
ax.set_ylabel("Out-Degree (outbound corridors)")
ax.legend(fontsize=8)
imb = metric_df[(metric_df["out_degree"] / (metric_df["in_degree"] + 1) > 1.8) |
                (metric_df["in_degree"]  / (metric_df["out_degree"] + 1) > 1.8)]
for _, row in imb.head(5).iterrows():
    ax.annotate(row["hub"], (row["in_degree"], row["out_degree"]),
                fontsize=7, color=PURPLE, xytext=(3, 2), textcoords="offset points")

# 1f: Corridor risk score ranked
ax = fig.add_subplot(gs_fig[1, 2])
top15_c = edge_df.nlargest(15, "corridor_risk_score")
top15_c = top15_c.copy()
top15_c["corr_label"] = (top15_c["source_center"].str[:8] + "→"
                         + top15_c["destination_center"].str[:8])
ax.barh(range(15), top15_c["corridor_risk_score"].values[::-1],
        color=[RED if v > 1.5 else AMBER for v in top15_c["weight"].values[::-1]],
        edgecolor="white", alpha=0.88)
ax.set_yticks(range(15))
ax.set_yticklabels(top15_c["corr_label"].values[::-1], fontsize=8)
ax.set_title("Top 15 Corridors: Corridor Risk Score\n(Red = Delay > 1.5x)")
ax.set_xlabel("Corridor Risk Score")

plt.tight_layout()
plt.savefig(output_dir / "network_metrics_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ network_metrics_dashboard.png saved.")


# ── Figure 2: Metric Heatmap + BRS vs SLA scatter ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle("Hub Metric Heatmap — Top 30 Hubs by BRS",
             fontsize=13, fontweight="bold")

top30 = metric_df.nlargest(30, "BRS").set_index("hub")
heatmap_cols = ["bc_unweighted", "bc_weighted", "degree_centrality",
                "in_degree", "out_degree", "closeness", "clustering",
                "pagerank", "avg_delay", "sla_breach_rate", "total_load", "BRS"]
col_labels = ["BC (unw)", "BC (wtd)", "Degree", "In°", "Out°",
              "Closeness", "Cluster", "PageRank", "Avg Delay", "SLA%", "Load", "BRS"]

hm_data = top30[heatmap_cols].copy()
for col in heatmap_cols:
    hm_data[col] = minmax(hm_data[col])

ax = axes[0]
sns.heatmap(hm_data, ax=ax, cmap="YlOrRd",
            cbar_kws={"label": "Normalized Score"},
            xticklabels=col_labels, linewidths=0.3, linecolor="#E2E8F0",
            annot=False)
ax.set_title("Metric Heatmap (Normalized)\nTop 30 Hubs by BRS")
ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

ax = axes[1]
metric_df["sla_trips"] = (metric_df["total_load"] * metric_df["sla_breach_rate"]).astype(int)
sc = ax.scatter(metric_df["BRS"], metric_df["sla_trips"],
                c=metric_df["avg_delay"], cmap="RdYlGn_r",
                s=metric_df["load_norm"] * 200 + 20,
                alpha=0.75, vmin=0.9, vmax=2.2)
plt.colorbar(sc, ax=ax, label="Avg Delay Ratio")
for _, row in metric_df.nlargest(5, "BRS").iterrows():
    ax.annotate(row["hub"], (row["BRS"], row["sla_trips"]),
                fontsize=8, color=RED, fontweight="bold",
                xytext=(5, 3), textcoords="offset points")
ax.set_title("BRS vs SLA Breach Trip Count\n(Size = Volume | Color = Delay)")
ax.set_xlabel("Bottleneck Risk Score (BRS)")
ax.set_ylabel("Estimated SLA Breach Trips")

plt.tight_layout()
plt.savefig(output_dir / "bottleneck_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ bottleneck_dashboard.png saved.")


# =============================================================================
# SECTION 7 — EXPORT
# =============================================================================

print("\n" + "=" * 65)
print("SECTION 7 — EXPORT")
print("=" * 65)

out_cols = [
    "hub", "BRS", "risk_tier", "avg_delay", "sla_breach_rate", "total_load",
    "bc_unweighted", "bc_weighted", "degree_centrality", "in_degree",
    "out_degree", "closeness", "clustering", "pagerank", "pagerank_weighted",
    "delay_norm", "bc_norm", "load_norm", "sla_norm", "sla_trips"
]
hub_metrics = metric_df[out_cols].sort_values("BRS", ascending=False)
hub_metrics.to_csv(output_dir / "hub_metrics.csv", index=False)

top_bottleneck_hubs = top10.copy()
top_bottleneck_hubs.to_csv(output_dir / "top_bottleneck_hubs.csv", index=False)

edge_df.sort_values("corridor_risk_score", ascending=False).to_csv(
    output_dir / "corridor_risk_scores.csv", index=False)

print(f"  ✓ hub_metrics.csv               ({len(hub_metrics)} rows)")
print(f"  ✓ top_bottleneck_hubs.csv        ({len(top_bottleneck_hubs)} rows)")
print(f"  ✓ corridor_risk_scores.csv       ({len(edge_df)} rows)")


# Summary
critical = metric_df[metric_df["risk_tier"] == "Critical"]
high     = metric_df[metric_df["risk_tier"] == "High"]

print(f"""
{'='*65}
BOTTLENECK SCORING COMPLETE
{'='*65}
  NETWORK: {G.number_of_nodes()} hubs | {G.number_of_edges()} corridors

  RISK TIER BREAKDOWN:
    Critical  : {len(critical):>3} hubs
    High      : {len(high):>3} hubs
    Elevated  : {len(metric_df[metric_df['risk_tier']=='Elevated']):>3} hubs
    Standard  : {len(metric_df[metric_df['risk_tier']=='Standard']):>3} hubs

  TOP BOTTLENECK: {top10.iloc[0]['hub']}  (BRS={top10.iloc[0]['BRS']:.3f})
  CHRONIC CORRIDORS: {edge_df['is_chronic'].sum()} / {len(edge_df)} exceed 20% delay

  BRS = 0.35×delay + 0.30×betweenness + 0.20×volume + 0.15×SLA
{'='*65}
""")
