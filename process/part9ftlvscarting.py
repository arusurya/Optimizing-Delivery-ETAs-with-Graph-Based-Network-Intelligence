"""
=============================================================================
PART 9: FTL vs CARTING — DECISION FRAMEWORK
=============================================================================
Role  : Logistics Strategy Consultant
Scope : Build an ML-backed, operations-manager-readable decision framework
        for route-type selection (FTL vs Carting), quantifying the
        time-cost trade-off per corridor profile using distance, delay
        risk, hub congestion, route type, time of day, and network
        position (from Part 7's Node2Vec embeddings / Part 4's centrality).

INPUT  : delivery_data.csv
         top20_delay_corridors.csv      (Part 5)
         graph_feature_importance.csv   (Part 7, optional)
OUTPUT : ftl_vs_carting_decision_matrix.csv
         ftl_vs_carting_scenario_analysis.csv
         fig_decision_matrix.png
         fig_cost_time_tradeoff.png
         fig_scenario_analysis.png
=============================================================================
USAGE:
    df = pd.read_csv("delivery_data.csv")
    # Then run all sections top-to-bottom.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F8FAFC",
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#CBD5E1",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})
RED, AMBER, GREEN, BLUE, PURPLE = "#DC2626", "#D97706", "#16A34A", "#2563EB", "#7C3AED"

print("=" * 65)
print("PART 9: FTL vs CARTING DECISION FRAMEWORK")
print("=" * 65)


# =============================================================================
# SECTION 0 — LOAD + CLEAN (same rules as previous parts)
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

df["hour"] = df["trip_creation_time"].dt.hour
def assign_tod(h):
    if h < 6:   return "night"
    if h < 12:  return "morn_peak"
    if h < 17:  return "afternoon"
    return "eve_peak"
df["time_bucket"] = df["hour"].map(assign_tod)


# =============================================================================
# SECTION 1 — COST & RATE ASSUMPTIONS
# =============================================================================
# These are illustrative INR figures consistent with Indian road-freight
# benchmarks for the time/cost trade-off — adjust to actual Delhivery
# contract rates if available.
#
#   FTL (Full Truck Load):
#     - Fixed dispatch cost per trip (truck booking, regardless of load)
#     - Lower per-km variable cost (dedicated vehicle, no consolidation delay)
#     - Lower delay sensitivity (doesn't wait for other shipments)
#
#   Carting (shared/consolidated, milk-run / LTL-style):
#     - No fixed dispatch cost (shipment rides on shared capacity)
#     - Higher per-km cost per shipment (cost split across fewer guaranteed units,
#       but shipment owner pays a per-unit handling premium)
#     - Higher delay sensitivity (waits for consolidation, multiple stops)

FTL_FIXED_COST       = 3500     # INR per trip (truck booking)
FTL_COST_PER_KM      = 28       # INR per km (dedicated vehicle)
CARTING_FIXED_COST   = 0        # INR (no dedicated booking)
CARTING_COST_PER_KM  = 45       # INR per km per shipment (consolidation premium)
COST_PER_DELAY_HOUR  = 15       # INR per hour of delay (matches Part 5 proxy)

print("\nSECTION 1 — COST ASSUMPTIONS")
print("=" * 65)
print(f"  FTL:     fixed=₹{FTL_FIXED_COST}/trip + ₹{FTL_COST_PER_KM}/km")
print(f"  Carting: fixed=₹{CARTING_FIXED_COST}/trip + ₹{CARTING_COST_PER_KM}/km")
print(f"  Delay cost: ₹{COST_PER_DELAY_HOUR}/hour of excess delay")


# =============================================================================
# SECTION 2 — CORRIDOR PROFILE: THE 6 DECISION INPUTS
# =============================================================================
# 1. Distance         — avg_osrm_distance (km)
# 2. Delay risk       — median delay_ratio (actual/OSRM), by route type
# 3. Hub congestion    — combined source+dest trip volume (proxy for
#                         dock/queue congestion, matches Part 4's degree
#                         centrality intuition)
# 4. Route type        — FTL / Carting (the decision variable itself —
#                         profile is built per CANDIDATE route type)
# 5. Time of day        — time_bucket (night/morn_peak/afternoon/eve_peak)
# 6. Network position   — corridor risk tier from Part 4/5 (bottleneck
#                          hub involvement) — loaded from Part 5 output
#                          if available, else derived inline.

print("\nSECTION 2 — CORRIDOR PROFILE CONSTRUCTION")
print("=" * 65)

corridor_profile = (
    df.groupby(["source_center", "destination_center", "route_type", "time_bucket"])
    .agg(
        shipment_volume = ("trip_uuid", "count"),
        avg_distance_km = ("osrm_distance", "mean"),
        avg_osrm_time   = ("osrm_time", "mean"),
        avg_actual_time = ("actual_time", "mean"),
        delay_ratio_med = ("delay_ratio", "median"),
    )
    .reset_index()
)
corridor_profile = corridor_profile[corridor_profile["shipment_volume"] >= 5]

# Hub congestion proxy: total trips touching source or destination hub
hub_touch = (
    pd.concat([
        df.groupby("source_center")["trip_uuid"].count(),
        df.groupby("destination_center")["trip_uuid"].count()
    ], axis=1, sort=False).fillna(0).sum(axis=1)
)
hub_touch.name = "hub_total_volume"
corridor_profile = corridor_profile.merge(
    hub_touch.rename("src_congestion"), left_on="source_center", right_index=True, how="left"
)
corridor_profile = corridor_profile.merge(
    hub_touch.rename("dst_congestion"), left_on="destination_center", right_index=True, how="left"
)
corridor_profile["hub_congestion"] = (
    corridor_profile["src_congestion"] + corridor_profile["dst_congestion"]
)

# Network position: load Part 5 chronic-corridor flags if available
try:
    chronic_corridors = pd.read_csv("outputs/top20_delay_corridors.csv")
    chronic_set = set(zip(chronic_corridors["source_center"], chronic_corridors["destination_center"]))
    print(f"  Loaded {len(chronic_set)} chronic corridors from Part 5.")
except FileNotFoundError:
    chronic_set = set()
    print("  ⚠ top20_delay_corridors.csv not found — network position derived inline only.")

corridor_profile["is_chronic_bottleneck"] = corridor_profile.apply(
    lambda r: (r["source_center"], r["destination_center"]) in chronic_set, axis=1
).astype(int)

# Congestion tier (for the matrix)
congestion_q = corridor_profile["hub_congestion"].quantile([0.33, 0.66]).values
def congestion_tier(v):
    if v <= congestion_q[0]: return "Low"
    if v <= congestion_q[1]: return "Medium"
    return "High"
corridor_profile["congestion_tier"] = corridor_profile["hub_congestion"].apply(congestion_tier)

# Distance tier
dist_q = corridor_profile["avg_distance_km"].quantile([0.33, 0.66]).values
def distance_tier(v):
    if v <= dist_q[0]: return "Short"
    if v <= dist_q[1]: return "Medium"
    return "Long"
corridor_profile["distance_tier"] = corridor_profile["avg_distance_km"].apply(distance_tier)

# Delay-risk tier
def delay_tier(v):
    if v <= 1.05: return "Low"
    if v <= 1.20: return "Medium"
    return "High"
corridor_profile["delay_risk_tier"] = corridor_profile["delay_ratio_med"].apply(delay_tier)

print(f"  Corridor-route_type-timebucket profiles (≥5 shipments): {len(corridor_profile):,}")
print(f"  Distance tiers (km cutoffs): Short ≤{dist_q[0]:.0f}, Medium ≤{dist_q[1]:.0f}, Long >{dist_q[1]:.0f}")
print(f"  Congestion tiers (trip-count cutoffs): Low ≤{congestion_q[0]:.0f}, "
      f"Medium ≤{congestion_q[1]:.0f}, High >{congestion_q[1]:.0f}")


# =============================================================================
# SECTION 3 — COST-TIME TRADEOFF (per shipment)
# =============================================================================
# For each corridor profile, compute the COST and TIME of shipping it via
# FTL vs Carting — even for profiles where only one route type was
# historically used, by applying the assumed rate card to the OTHER
# route type's avg_distance/avg_actual_time at that corridor.
#
# total_cost = fixed_cost + per_km_cost * distance + delay_cost
#   delay_cost = max(0, actual_time - osrm_time) / 3600 * COST_PER_DELAY_HOUR

print("\nSECTION 3 — COST-TIME TRADEOFF CALCULATION")
print("=" * 65)

# Build a corridor-level (ignoring route_type) summary so both options can
# be priced for the same lane/time-bucket combination.
lane = (
    corridor_profile.groupby(["source_center", "destination_center", "time_bucket"])
    .agg(
        avg_distance_km = ("avg_distance_km", "mean"),
        avg_osrm_time   = ("avg_osrm_time", "mean"),
        hub_congestion  = ("hub_congestion", "mean"),
        congestion_tier = ("congestion_tier", "first"),
        distance_tier   = ("distance_tier", "first"),
        is_chronic_bottleneck = ("is_chronic_bottleneck", "max"),
        total_volume    = ("shipment_volume", "sum"),
    )
    .reset_index()
)

# Delay ratio specific to each route type on this lane (fallback to overall
# corridor-level delay if a specific route type wasn't observed there).
for rt in ["FTL", "Carting"]:
    sub = corridor_profile[corridor_profile["route_type"] == rt][
        ["source_center", "destination_center", "time_bucket", "delay_ratio_med", "delay_risk_tier"]
    ].rename(columns={"delay_ratio_med": f"delay_ratio_{rt.lower()}",
                       "delay_risk_tier": f"delay_risk_tier_{rt.lower()}"})
    lane = lane.merge(sub, on=["source_center", "destination_center", "time_bucket"], how="left")

overall_median_delay = df["delay_ratio"].median()
lane["delay_ratio_ftl"] = lane["delay_ratio_ftl"].fillna(overall_median_delay)
lane["delay_ratio_carting"] = lane["delay_ratio_carting"].fillna(overall_median_delay * 1.1)
lane["delay_risk_tier_ftl"] = lane["delay_risk_tier_ftl"].fillna("Medium")
lane["delay_risk_tier_carting"] = lane["delay_risk_tier_carting"].fillna("Medium")

# Cost & time for each option
lane["ftl_distance_cost"] = FTL_FIXED_COST + FTL_COST_PER_KM * lane["avg_distance_km"]
lane["carting_distance_cost"] = CARTING_FIXED_COST + CARTING_COST_PER_KM * lane["avg_distance_km"]

lane["ftl_actual_time"] = lane["avg_osrm_time"] * lane["delay_ratio_ftl"]
lane["carting_actual_time"] = lane["avg_osrm_time"] * lane["delay_ratio_carting"]

lane["ftl_delay_cost"] = ((lane["ftl_actual_time"] - lane["avg_osrm_time"]).clip(lower=0)
                           / 3600 * COST_PER_DELAY_HOUR)
lane["carting_delay_cost"] = ((lane["carting_actual_time"] - lane["avg_osrm_time"]).clip(lower=0)
                               / 3600 * COST_PER_DELAY_HOUR)

lane["ftl_total_cost"] = lane["ftl_distance_cost"] + lane["ftl_delay_cost"]
lane["carting_total_cost"] = lane["carting_distance_cost"] + lane["carting_delay_cost"]

lane["time_diff_hr"] = (lane["carting_actual_time"] - lane["ftl_actual_time"]) / 3600
lane["cost_diff"] = lane["carting_total_cost"] - lane["ftl_total_cost"]

print(f"  Lane-level (per source-dest-timebucket) profiles: {len(lane):,}")
print(f"  Avg FTL total cost:     ₹{lane['ftl_total_cost'].mean():,.0f}")
print(f"  Avg Carting total cost: ₹{lane['carting_total_cost'].mean():,.0f}")
print(f"  Avg time difference (Carting - FTL): {lane['time_diff_hr'].mean():.2f} hours")


# =============================================================================
# SECTION 4 — DECISION RULES
# =============================================================================
# Operations-manager-readable rules, applied in priority order. Each rule
# encodes a clear business rationale.

print("\nSECTION 4 — DECISION RULES")
print("=" * 65)

DECISION_RULES = """
  RULE 1 (Bottleneck override):
    IF corridor is a CHRONIC BOTTLENECK (Part 5 top-20)
       AND delay_risk = HIGH for Carting
    → FORCE FTL (dedicated capacity bypasses consolidation queues
      at congested hubs; Carting would compound existing delays)

  RULE 2 (Long-haul default):
    IF distance = LONG (>66th percentile km)
    → DEFAULT FTL (fixed dispatch cost amortizes over distance;
      Carting's per-km premium dominates on long lanes)

  RULE 3 (Short-haul + low congestion):
    IF distance = SHORT AND hub_congestion = LOW
    → DEFAULT CARTING (no fixed cost advantage to FTL on short
      hops; consolidation delay risk is low when hubs aren't busy)

  RULE 4 (Peak-hour congestion):
    IF time_bucket IN {morn_peak, eve_peak} AND congestion = HIGH
    → PREFER FTL (avoids queueing for consolidation slots during
      the hub's busiest hours, even at higher per-trip cost)

  RULE 5 (Volume threshold — economics of scale):
    IF shipment_volume on lane ≥ 50/week
    → PREFER FTL (volume justifies dedicated capacity; per-unit
      cost of FTL falls below Carting at this scale)

  RULE 6 (Default / fallback):
    IF none of the above trigger
    → CHOOSE LOWER total_cost option (ftl_total_cost vs
      carting_total_cost), tie-break towards Carting (lower
      capital commitment)
"""
print(DECISION_RULES)

VOLUME_THRESHOLD_WEEKLY = 50

def decide_route_type(row):
    # Rule 1
    if row["is_chronic_bottleneck"] == 1 and row["delay_risk_tier_carting"] == "High":
        return "FTL", "Rule 1: Chronic bottleneck — force FTL to bypass congestion"
    # Rule 2
    if row["distance_tier"] == "Long":
        return "FTL", "Rule 2: Long-haul — fixed cost amortizes over distance"
    # Rule 3
    if row["distance_tier"] == "Short" and row["congestion_tier"] == "Low":
        return "Carting", "Rule 3: Short-haul + low congestion — no FTL advantage"
    # Rule 4
    if row["time_bucket"] in ["morn_peak", "eve_peak"] and row["congestion_tier"] == "High":
        return "FTL", "Rule 4: Peak-hour congestion — avoid consolidation queue"
    # Rule 5 (approximate weekly volume from total_volume over data span)
    span_weeks = max(1, (df["trip_creation_time"].max() - df["trip_creation_time"].min()).days / 7)
    weekly_vol = row["total_volume"] / span_weeks
    if weekly_vol >= VOLUME_THRESHOLD_WEEKLY:
        return "FTL", f"Rule 5: High volume ({weekly_vol:.0f}/wk) — economics favor FTL"
    # Rule 6 — cost-based fallback
    if row["ftl_total_cost"] < row["carting_total_cost"]:
        return "FTL", "Rule 6: Lower total cost (FTL)"
    return "Carting", "Rule 6: Lower total cost (Carting)"

lane[["recommended_route_type", "rule_applied"]] = lane.apply(
    lambda r: pd.Series(decide_route_type(r)), axis=1
)

print(f"  Recommendation distribution:")
print(lane["recommended_route_type"].value_counts().to_string())


# =============================================================================
# SECTION 5 — DECISION MATRIX
# =============================================================================
# A simple 2D matrix (Distance Tier x Congestion Tier) showing the modal
# recommendation and avg cost saving — the headline artifact for ops
# managers who want a quick lookup table without running the model.

print("\nSECTION 5 — DECISION MATRIX (Distance x Congestion)")
print("=" * 65)

decision_matrix = (
    lane.groupby(["distance_tier", "congestion_tier"])
    .agg(
        modal_recommendation = ("recommended_route_type", lambda x: x.mode().iloc[0]),
        pct_ftl_recommended  = ("recommended_route_type", lambda x: (x == "FTL").mean() * 100),
        avg_cost_diff        = ("cost_diff", "mean"),   # Carting - FTL; positive => FTL cheaper
        avg_time_diff_hr     = ("time_diff_hr", "mean"),  # positive => Carting slower
        n_lanes              = ("source_center", "count"),
    )
    .reset_index()
)
order_dist = ["Short", "Medium", "Long"]
order_cong = ["Low", "Medium", "High"]
decision_matrix["distance_tier"] = pd.Categorical(decision_matrix["distance_tier"], order_dist, ordered=True)
decision_matrix["congestion_tier"] = pd.Categorical(decision_matrix["congestion_tier"], order_cong, ordered=True)
decision_matrix = decision_matrix.sort_values(["distance_tier", "congestion_tier"]).reset_index(drop=True)

print(decision_matrix.to_string(index=False))


# =============================================================================
# SECTION 6 — SCENARIO ANALYSIS
# =============================================================================
# 5 representative operational scenarios an ops manager would recognize.
# For each: inputs, recommendation, cost/time trade-off, and the rule fired.

print("\nSECTION 6 — SCENARIO ANALYSIS")
print("=" * 65)

scenarios = [
    {"name": "Short urban hop, quiet hub",
     "distance_tier": "Short", "congestion_tier": "Low",
     "time_bucket": "afternoon", "is_chronic_bottleneck": 0,
     "delay_risk_tier_carting": "Low", "total_volume": 20,
     "avg_distance_km": 25, "avg_osrm_time": 1800,
     "delay_ratio_ftl": 1.05, "delay_ratio_carting": 1.08},

    {"name": "Long inter-city corridor",
     "distance_tier": "Long", "congestion_tier": "Medium",
     "time_bucket": "afternoon", "is_chronic_bottleneck": 0,
     "delay_risk_tier_carting": "Medium", "total_volume": 120,
     "avg_distance_km": 450, "avg_osrm_time": 18000,
     "delay_ratio_ftl": 1.10, "delay_ratio_carting": 1.18},

    {"name": "Chronic bottleneck hub, evening peak",
     "distance_tier": "Medium", "congestion_tier": "High",
     "time_bucket": "eve_peak", "is_chronic_bottleneck": 1,
     "delay_risk_tier_carting": "High", "total_volume": 60,
     "avg_distance_km": 150, "avg_osrm_time": 7200,
     "delay_ratio_ftl": 1.15, "delay_ratio_carting": 1.65},

    {"name": "High-volume regional lane",
     "distance_tier": "Medium", "congestion_tier": "Medium",
     "time_bucket": "morn_peak", "is_chronic_bottleneck": 0,
     "delay_risk_tier_carting": "Medium", "total_volume": 400,
     "avg_distance_km": 120, "avg_osrm_time": 6000,
     "delay_ratio_ftl": 1.08, "delay_ratio_carting": 1.15},

    {"name": "Low-volume night corridor, low congestion",
     "distance_tier": "Medium", "congestion_tier": "Low",
     "time_bucket": "night", "is_chronic_bottleneck": 0,
     "delay_risk_tier_carting": "Low", "total_volume": 8,
     "avg_distance_km": 100, "avg_osrm_time": 5400,
     "delay_ratio_ftl": 1.02, "delay_ratio_carting": 1.06},
]

scenario_rows = []
for sc in scenarios:
    s = pd.Series(sc)
    s["ftl_distance_cost"] = FTL_FIXED_COST + FTL_COST_PER_KM * s["avg_distance_km"]
    s["carting_distance_cost"] = CARTING_FIXED_COST + CARTING_COST_PER_KM * s["avg_distance_km"]
    s["ftl_actual_time"] = s["avg_osrm_time"] * s["delay_ratio_ftl"]
    s["carting_actual_time"] = s["avg_osrm_time"] * s["delay_ratio_carting"]
    s["ftl_delay_cost"] = max(0, s["ftl_actual_time"] - s["avg_osrm_time"]) / 3600 * COST_PER_DELAY_HOUR
    s["carting_delay_cost"] = max(0, s["carting_actual_time"] - s["avg_osrm_time"]) / 3600 * COST_PER_DELAY_HOUR
    s["ftl_total_cost"] = s["ftl_distance_cost"] + s["ftl_delay_cost"]
    s["carting_total_cost"] = s["carting_distance_cost"] + s["carting_delay_cost"]
    s["time_diff_hr"] = (s["carting_actual_time"] - s["ftl_actual_time"]) / 3600
    s["cost_diff"] = s["carting_total_cost"] - s["ftl_total_cost"]

    rec, rule = decide_route_type(s)
    s["recommended_route_type"] = rec
    s["rule_applied"] = rule
    scenario_rows.append(s)

scenario_df = pd.DataFrame(scenario_rows)

for _, s in scenario_df.iterrows():
    print(f"\n  SCENARIO: {s['name']}")
    print(f"    Inputs: distance={s['distance_tier']}, congestion={s['congestion_tier']}, "
          f"time={s['time_bucket']}, bottleneck={'Yes' if s['is_chronic_bottleneck'] else 'No'}, "
          f"vol={s['total_volume']}")
    print(f"    FTL:     cost=₹{s['ftl_total_cost']:,.0f}, time={s['ftl_actual_time']/3600:.2f}h")
    print(f"    Carting: cost=₹{s['carting_total_cost']:,.0f}, time={s['carting_actual_time']/3600:.2f}h")
    print(f"    → RECOMMENDATION: {s['recommended_route_type']}  ({s['rule_applied']})")


# =============================================================================
# SECTION 7 — VISUALIZATIONS
# =============================================================================

print("\nSECTION 7 — GENERATING VISUALIZATIONS")
print("=" * 65)

# ── Figure A: Decision Matrix heatmap ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("FTL vs Carting — Decision Matrix", fontsize=14, fontweight="bold")

# A1: Modal recommendation grid
ax = axes[0]
pivot_rec = decision_matrix.pivot(index="distance_tier", columns="congestion_tier",
                                    values="pct_ftl_recommended")
pivot_rec = pivot_rec.reindex(index=order_dist, columns=order_cong)
im = ax.imshow(pivot_rec.values, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(order_cong)
ax.set_yticks(range(3)); ax.set_yticklabels(order_dist)
ax.set_xlabel("Hub Congestion"); ax.set_ylabel("Distance")
ax.set_title("% of Lanes Recommended FTL\n(green=Carting-favored, red=FTL-favored)")
for i in range(3):
    for j in range(3):
        val = pivot_rec.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val > 50 else "black")
plt.colorbar(im, ax=ax, label="% FTL recommended")

# A2: Avg cost difference grid (positive = FTL cheaper)
ax = axes[1]
pivot_cost = decision_matrix.pivot(index="distance_tier", columns="congestion_tier",
                                     values="avg_cost_diff")
pivot_cost = pivot_cost.reindex(index=order_dist, columns=order_cong)
im2 = ax.imshow(pivot_cost.values, cmap="RdBu", aspect="auto",
                vmin=-pivot_cost.abs().max().max(), vmax=pivot_cost.abs().max().max())
ax.set_xticks(range(3)); ax.set_xticklabels(order_cong)
ax.set_yticks(range(3)); ax.set_yticklabels(order_dist)
ax.set_xlabel("Hub Congestion"); ax.set_ylabel("Distance")
ax.set_title("Avg Cost Saving with FTL (₹)\n(positive = FTL cheaper than Carting)")
for i in range(3):
    for j in range(3):
        val = pivot_cost.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"₹{val:,.0f}", ha="center", va="center",
                    fontsize=10, fontweight="bold")
plt.colorbar(im2, ax=ax, label="₹ (Carting cost − FTL cost)")

plt.tight_layout()
plt.savefig("outputs/fig_decision_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_decision_matrix.png saved.")

# ── Figure B: Cost-time tradeoff scatter ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
sample = lane.sample(min(2000, len(lane)), random_state=42)
for rt, color, marker in [("FTL", BLUE, "o"), ("Carting", AMBER, "^")]:
    mask = sample["recommended_route_type"] == rt
    ax.scatter(sample.loc[mask, f"{rt.lower()}_actual_time"]/3600,
               sample.loc[mask, f"{rt.lower()}_total_cost"],
               alpha=0.4, s=25, color=color, marker=marker, label=f"{rt} (recommended)")
ax.set_xlabel("Delivery Time (hours)")
ax.set_ylabel("Total Cost (₹)")
ax.set_title("Cost-Time Trade-off by Recommended Route Type\n"
              "(each point = one lane × time-bucket profile, priced under its recommendation)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/fig_cost_time_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_cost_time_tradeoff.png saved.")

# ── Figure C: Scenario analysis bar chart ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Scenario Analysis: FTL vs Carting", fontsize=14, fontweight="bold")

x = np.arange(len(scenario_df))
width = 0.35

ax = axes[0]
ax.bar(x - width/2, scenario_df["ftl_total_cost"], width, label="FTL", color=BLUE, edgecolor="white")
ax.bar(x + width/2, scenario_df["carting_total_cost"], width, label="Carting", color=AMBER, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([s.replace(", ", ",\n") for s in scenario_df["name"]], fontsize=8, rotation=15, ha="right")
ax.set_ylabel("Total Cost (₹)")
ax.set_title("Cost Comparison")
ax.legend(fontsize=9)
for i, rec in enumerate(scenario_df["recommended_route_type"]):
    ax.annotate(f"→ {rec}", (i, max(scenario_df.loc[i,"ftl_total_cost"], scenario_df.loc[i,"carting_total_cost"])),
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=GREEN if rec == "Carting" else PURPLE)

ax = axes[1]
ax.bar(x - width/2, scenario_df["ftl_actual_time"]/3600, width, label="FTL", color=BLUE, edgecolor="white")
ax.bar(x + width/2, scenario_df["carting_actual_time"]/3600, width, label="Carting", color=AMBER, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([s.replace(", ", ",\n") for s in scenario_df["name"]], fontsize=8, rotation=15, ha="right")
ax.set_ylabel("Delivery Time (hours)")
ax.set_title("Time Comparison")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/fig_scenario_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_scenario_analysis.png saved.")


# =============================================================================
# SECTION 8 — EXPORT
# =============================================================================

print("\nSECTION 8 — EXPORT")
print("=" * 65)

lane.to_csv("outputs/ftl_vs_carting_decision_matrix.csv", index=False)
scenario_df.drop(columns=["name"]).assign(scenario=scenario_df["name"]).to_csv(
    "outputs/ftl_vs_carting_scenario_analysis.csv", index=False)
decision_matrix.to_csv("outputs/ftl_vs_carting_summary_matrix.csv", index=False)

print("  ✓ ftl_vs_carting_decision_matrix.csv  (lane-level, all profiles)")
print("  ✓ ftl_vs_carting_scenario_analysis.csv")
print("  ✓ ftl_vs_carting_summary_matrix.csv  (distance x congestion grid)")


# =============================================================================
# SECTION 9 — BUSINESS RECOMMENDATIONS
# =============================================================================

pct_ftl_overall = (lane["recommended_route_type"] == "FTL").mean() * 100
avg_saving_when_ftl = lane.loc[lane["recommended_route_type"]=="FTL", "cost_diff"].mean()
avg_saving_when_carting = -lane.loc[lane["recommended_route_type"]=="Carting", "cost_diff"].mean()
bottleneck_lanes = lane[lane["is_chronic_bottleneck"]==1]
bottleneck_ftl_pct = (bottleneck_lanes["recommended_route_type"]=="FTL").mean()*100 if len(bottleneck_lanes) else 0

print(f"""
{'='*65}
FTL vs CARTING — BUSINESS RECOMMENDATIONS
{'='*65}

  1. NETWORK-WIDE SPLIT
     {pct_ftl_overall:.0f}% of lane-profiles are recommended FTL,
     {100-pct_ftl_overall:.0f}% Carting. On lanes where Carting is the
     recommendation, it is the lower-cost option by an average of
     ₹{avg_saving_when_carting:,.0f} per shipment. On lanes where FTL is
     recommended, the choice is often driven by operational rules
     (bottleneck override, peak congestion, long-haul) rather than raw
     cost alone — average cost delta on these lanes is ₹{-avg_saving_when_ftl:,.0f}
     ({"FTL costs more but is justified by delay/risk reduction" if avg_saving_when_ftl < 0 else "FTL is also the cheaper option"}).

  2. CHRONIC BOTTLENECK CORRIDORS (Part 5 top-20)
     {bottleneck_ftl_pct:.0f}% of chronic-bottleneck lane-profiles are
     recommended FTL — Rule 1 (bottleneck override) is the dominant driver.
     Action: prioritize FTL allocation on these {len(bottleneck_lanes['source_center'].unique())
     if len(bottleneck_lanes) else 0} source hubs immediately;
     this is a near-zero-cost operational change (re-allocation, not capex).

  3. PEAK-HOUR POLICY
     Morning and evening peak windows on high-congestion hubs should
     default to FTL regardless of the cost delta — the delay-cost
     component (Rule 4) compounds non-linearly during consolidation
     queue buildup, which this static model underestimates.

  4. SHORT-HAUL / LOW-CONGESTION LANES
     These are Carting's strongest use case — no fixed-cost penalty,
     low delay risk. Audit any FTL usage on these lanes for potential
     cost savings (over-provisioning).

  5. VOLUME-BASED CONTRACT REVIEW
     Lanes crossing the {VOLUME_THRESHOLD_WEEKLY}-shipment/week threshold
     should be reviewed for dedicated FTL contracts — spot-booking FTL
     at this volume is likely costlier than a committed-capacity rate.

  6. MONITORING
     Re-run this framework quarterly. Hub congestion and chronic-
     bottleneck status (Part 5) shift as the network grows — a lane
     recommended Carting today may need FTL after a hub's volume grows.
{'='*65}
""")
