
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#CBD5E1",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})

RED, AMBER, GREEN, BLUE, PURPLE = "#DC2626", "#D97706", "#16A34A", "#2563EB", "#7C3AED"

# =============================================================================
# SECTION 0 — LOAD DATA
# =============================================================================

df = pd.read_csv("delivery_data.csv")

print("=" * 65)
print("PART 5: DELAY CORRIDOR AUDIT")
print("=" * 65)

# ── Cleaning (same rules as Part 3) ──────────────────────────────────────────
raw_n = len(df)
df = df[df["actual_time"] > 0]
df = df[df["osrm_time"] > 0]
df = df[df["source_center"] != df["destination_center"]]
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
df = df.dropna(subset=["trip_creation_time", "source_center", "destination_center"])
print(f"  Clean rows: {len(df):,}  (dropped {raw_n - len(df):,})")

# ── Core delay ratio ──────────────────────────────────────────────────────────
df["delay_ratio"] = df["actual_time"] / df["osrm_time"]
df["sla_breach"]  = (df["delay_ratio"] > 1.20).astype(int)


# =============================================================================
# SECTION 1 — REVENUE-AT-RISK ASSUMPTIONS
# =============================================================================
# Two proxies are used together:
#   (a) SLA PENALTY proxy   — fixed cost incurred per breached shipment
#       (contractual penalty / customer-credit cost). Conservative estimate
#       commonly used in logistics SLA contracts: ₹100 per breached shipment.
#   (b) DELAY-HOUR proxy    — operational cost of the EXCESS time itself
#       (extra fuel, driver hours, vehicle idling, demurrage at hubs).
#       Estimated at ₹15 per hour of delay beyond OSRM ETA.
#
# Revenue-at-risk per corridor = breach-trip penalty + excess-hour cost.
# This gives a $ figure ops leadership can compare against intervention cost.

SLA_PENALTY_PER_BREACH = 100      # INR per shipment that breaches SLA (>1.2x)
COST_PER_EXCESS_HOUR   = 15       # INR per hour of delay beyond OSRM time
CHRONIC_THRESHOLD      = 1.20     # PS-defined threshold: actual > OSRM by 20%
MIN_SUPPORT            = 10       # minimum trips for a stable corridor estimate


# =============================================================================
# SECTION 2 — CORRIDOR-LEVEL AGGREGATION
# =============================================================================

print("\nSECTION 2 — CORRIDOR AGGREGATION")
print("=" * 65)

df["excess_time_sec"] = (df["actual_time"] - df["osrm_time"]).clip(lower=0)

corr = (
    df.groupby(["source_center", "destination_center"])
    .agg(
        shipment_volume   = ("trip_uuid",       "count"),
        delay_ratio_mean  = ("delay_ratio",      "mean"),
        delay_ratio_med   = ("delay_ratio",      "median"),
        sla_breach_trips  = ("sla_breach",       "sum"),
        sla_breach_rate   = ("sla_breach",       "mean"),
        median_actual_sec = ("actual_time",      "median"),
        median_osrm_sec   = ("osrm_time",        "median"),
        total_excess_sec  = ("excess_time_sec",  "sum"),
        avg_excess_sec    = ("excess_time_sec",  "mean"),
        avg_osrm_distance = ("osrm_distance",    "mean"),
        pct_ftl           = ("route_type",       lambda x: (x == "FTL").mean()),
    )
    .reset_index()
)

# Minimum support filter — same logic as Part 3 (median needs ≥10 trips)
corr = corr[corr["shipment_volume"] >= MIN_SUPPORT].reset_index(drop=True)

# PS DEFINITION: corridor qualifies if ACTUAL > OSRM by >20%
# Use the median delay ratio (robust to single-trip outliers) as the
# qualifying criterion, consistent with Part 3's edge-weight definition.
chronic = corr[corr["delay_ratio_med"] > CHRONIC_THRESHOLD].copy()

print(f"  Total corridors (≥{MIN_SUPPORT} trips):        {len(corr):,}")
print(f"  Corridors with delay > 20% (chronic):  {len(chronic):,} "
      f"({len(chronic)/len(corr)*100:.1f}%)")


# =============================================================================
# SECTION 3 — PER-CORRIDOR METRICS
# =============================================================================
# 1. Delay ratio        — already computed (delay_ratio_med)
# 2. Shipment volume    — already computed (shipment_volume)
# 3. SLA breach contribution — this corridor's share of ALL SLA breaches
#    across the network. Tells ops "fixing this corridor alone reduces
#    total network breaches by X%".
# 4. Revenue-at-risk proxy — penalty cost + excess-time operational cost.

print("\nSECTION 3 — COMPUTING SLA BREACH CONTRIBUTION & REVENUE AT RISK")
print("=" * 65)

total_breach_trips_network = df["sla_breach"].sum()

chronic["sla_breach_contribution_pct"] = (
    chronic["sla_breach_trips"] / total_breach_trips_network * 100
)

# Revenue-at-risk proxy (₹)
chronic["penalty_cost"] = chronic["sla_breach_trips"] * SLA_PENALTY_PER_BREACH
chronic["excess_time_cost"] = (
    chronic["total_excess_sec"] / 3600 * COST_PER_EXCESS_HOUR
)
chronic["revenue_at_risk"] = (
    chronic["penalty_cost"] + chronic["excess_time_cost"]
).round(0)

print(f"  Network-wide SLA breach trips: {total_breach_trips_network:,}")
print(f"  Total revenue-at-risk across {len(chronic):,} chronic corridors: "
      f"₹{chronic['revenue_at_risk'].sum():,.0f}")


# =============================================================================
# SECTION 4 — SEVERITY RANKING / COMPOSITE SCORE
# =============================================================================
# Composite "Corridor Severity Score" — weights chosen to mirror the
# operational priorities in the PS: delay magnitude matters most (how
# broken is it), but a high-delay low-volume corridor is less urgent than
# a moderately delayed high-volume one. Revenue-at-risk folds in both
# penalty and operational cost, so it gets a meaningful weight too.
#
#   Severity = 0.35 × delay_ratio_norm
#            + 0.25 × volume_norm
#            + 0.20 × sla_breach_contribution_norm
#            + 0.20 × revenue_at_risk_norm

print("\nSECTION 4 — SEVERITY RANKING")
print("=" * 65)

def minmax(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r else s * 0

chronic["delay_norm"]   = minmax(chronic["delay_ratio_med"])
chronic["volume_norm"]  = minmax(chronic["shipment_volume"])
chronic["breach_norm"]  = minmax(chronic["sla_breach_contribution_pct"])
chronic["revenue_norm"] = minmax(chronic["revenue_at_risk"])

chronic["severity_score"] = (
    0.35 * chronic["delay_norm"]
  + 0.25 * chronic["volume_norm"]
  + 0.20 * chronic["breach_norm"]
  + 0.20 * chronic["revenue_norm"]
)

# Severity tiers (for the strategy memo / dashboard color-coding)
def tier(score, q75, q50, q25):
    if score >= q75: return "Critical"
    if score >= q50: return "High"
    if score >= q25: return "Moderate"
    return "Watch"

q75, q50, q25 = chronic["severity_score"].quantile([0.75, 0.50, 0.25])
chronic["severity_tier"] = chronic["severity_score"].apply(tier, args=(q75, q50, q25))


# =============================================================================
# SECTION 5 — TOP 20 DELAYED CORRIDORS
# =============================================================================

print("\nSECTION 5 — TOP 20 DELAYED CORRIDORS")
print("=" * 65)

top20 = chronic.nlargest(20, "severity_score").reset_index(drop=True)
top20.index += 1
top20["corridor"] = top20["source_center"] + " → " + top20["destination_center"]


# =============================================================================
# SECTION 6 — RECOMMENDED INTERVENTIONS
# =============================================================================
# Rule-based intervention mapping. Logic:
#   - High delay + high FTL share + high distance → PARALLEL ROUTE
#     (long-haul corridor systemically slower than OSRM; an alternate
#      routing option spreads load and avoids the structural bottleneck)
#   - High delay + high volume + low FTL share → FACILITY UPGRADE
#     (short-haul/urban corridor — congestion is likely at the hub/dock,
#      not on the road; capacity/dock-door upgrade at source or dest hub)
#   - High delay + high variability (IQR-like proxy via mean-median gap)
#     + moderate volume → ROUTE-TYPE SHIFT (FTL <-> Carting)
#     (mismatch between vehicle type and corridor profile)
#   - Else → MONITOR / PROCESS REVIEW

def recommend(row):
    long_haul = row["avg_osrm_distance"] > corr["avg_osrm_distance"].median()
    high_ftl  = row["pct_ftl"] > 0.5
    high_vol  = row["shipment_volume"] > corr["shipment_volume"].quantile(0.75)
    mean_med_gap = abs(row["delay_ratio_mean"] - row["delay_ratio_med"])

    if row["delay_ratio_med"] > 1.5 and long_haul and high_ftl:
        return "Parallel route (alternate corridor / load-split to relieve structural delay)"
    if high_vol and not high_ftl:
        return "Facility upgrade (dock/capacity expansion at source or destination hub)"
    if mean_med_gap > 0.3 and 0.3 <= row["pct_ftl"] <= 0.7:
        return "Route-type shift (re-evaluate FTL vs Carting allocation on this corridor)"
    if row["delay_ratio_med"] > 1.8:
        return "Facility upgrade + parallel route (dual intervention — severe chronic delay)"
    return "Process review / monitor (schedule audit, no capital intervention yet)"

top20["recommended_intervention"] = top20.apply(recommend, axis=1)


# ── Console report ───────────────────────────────────────────────────────────
print(f"\n  {'#':<3} {'Corridor':<22} {'Tier':<9} {'Delay':>6} {'Vol':>6} "
      f"{'SLA Contr.':>10} {'Rev@Risk':>10}")
print(f"  {'-'*3} {'-'*22} {'-'*9} {'-'*6} {'-'*6} {'-'*10} {'-'*10}")
for rank, row in top20.iterrows():
    tier_sym = {"Critical":"🔴","High":"🟠","Moderate":"🟡","Watch":"🟢"}.get(row["severity_tier"], "⚪")
    print(f"  {rank:<3} {row['corridor']:<22} {tier_sym}{row['severity_tier']:<8} "
          f"{row['delay_ratio_med']:>6.2f} {int(row['shipment_volume']):>6,} "
          f"{row['sla_breach_contribution_pct']:>9.2f}% ₹{row['revenue_at_risk']:>9,.0f}")

print(f"\n  RECOMMENDED INTERVENTIONS (Top 20):")
for rank, row in top20.iterrows():
    print(f"   {rank:>2}. {row['corridor']:<22} → {row['recommended_intervention']}")


# =============================================================================
# SECTION 7 — VISUALIZATIONS
# =============================================================================

print("\nSECTION 7 — GENERATING VISUALIZATIONS")
print("=" * 65)

# ── Figure A: Severity ranking bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
tier_colors = {"Critical": RED, "High": AMBER, "Moderate": PURPLE, "Watch": GREEN}
colors = [tier_colors[t] for t in top20["severity_tier"].values[::-1]]
bars = ax.barh(range(20), top20["severity_score"].values[::-1],
               color=colors, edgecolor="white", alpha=0.9)
ax.set_yticks(range(20))
ax.set_yticklabels(top20["corridor"].values[::-1], fontsize=8)
ax.set_xlabel("Corridor Severity Score")
ax.set_title("Top 20 Delayed Corridors — Severity Ranking\n"
              "(🔴 Critical  🟠 High  🟣 Moderate  🟢 Watch)")
legend_el = [mpatches.Patch(color=c, label=t) for t, c in tier_colors.items()]
ax.legend(handles=legend_el, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/fig_corridor_severity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_corridor_severity.png saved.")

# ── Figure B: 3-panel operational dashboard ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
fig.suptitle("Delay Corridor Operational Dashboard (Top 20)", fontsize=14, fontweight="bold")

# B1: Delay ratio vs Volume, sized by revenue-at-risk
ax = axes[0]
sc = ax.scatter(top20["shipment_volume"], top20["delay_ratio_med"],
                c=top20["revenue_at_risk"], cmap="Reds",
                s=top20["sla_breach_contribution_pct"] * 15 + 30,
                alpha=0.8, edgecolors="black", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="Revenue at Risk (₹)")
ax.axhline(CHRONIC_THRESHOLD, color="black", lw=1.5, ls="--", label="20% threshold")
for _, row in top20.head(5).iterrows():
    ax.annotate(row["corridor"], (row["shipment_volume"], row["delay_ratio_med"]),
                fontsize=7, fontweight="bold", xytext=(4, 3), textcoords="offset points")
ax.set_title("Delay Ratio vs Shipment Volume\n(Size = SLA breach contribution)")
ax.set_xlabel("Shipment Volume (trips)")
ax.set_ylabel("Delay Ratio (Actual / OSRM)")
ax.legend(fontsize=8)

# B2: SLA breach contribution bar chart
ax = axes[1]
ax.barh(range(20), top20["sla_breach_contribution_pct"].values[::-1],
        color=BLUE, edgecolor="white", alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels(top20["corridor"].values[::-1], fontsize=8)
ax.set_title("SLA Breach Contribution\n(% of all network breaches)")
ax.set_xlabel("% of Network SLA Breaches")

# B3: Revenue at risk bar chart
ax = axes[2]
ax.barh(range(20), top20["revenue_at_risk"].values[::-1],
        color=GREEN, edgecolor="white", alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels(top20["corridor"].values[::-1], fontsize=8)
ax.set_title("Revenue-at-Risk Proxy (₹)\n(SLA penalty + excess-time cost)")
ax.set_xlabel("Revenue at Risk (₹)")

plt.tight_layout()
plt.savefig("outputs/fig_corridor_severity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ fig_corridor_dashboard.png saved.")


# =============================================================================
# SECTION 8 — EXPORT
# =============================================================================

print("\nSECTION 8 — EXPORT")
print("=" * 65)

export_cols = [
    "corridor", "source_center", "destination_center", "severity_tier",
    "severity_score", "delay_ratio_med", "delay_ratio_mean",
    "shipment_volume", "sla_breach_trips", "sla_breach_rate",
    "sla_breach_contribution_pct", "revenue_at_risk", "penalty_cost",
    "excess_time_cost", "avg_osrm_distance", "pct_ftl",
    "recommended_intervention"
]
top20[export_cols].to_csv("outputs/top20_delay_corridors.csv", index=False)
print("  ✓ top20_delay_corridors.csv")


# =============================================================================
# SECTION 9 — OPERATIONAL RECOMMENDATIONS (TRANSLATED FOR OPS LEADERSHIP)
# =============================================================================

n_critical = (top20["severity_tier"] == "Critical").sum()
n_high     = (top20["severity_tier"] == "High").sum()
top3_rar   = top20.nlargest(3, "revenue_at_risk")["revenue_at_risk"].sum()
top3_breach= top20.nlargest(3, "sla_breach_contribution_pct")["sla_breach_contribution_pct"].sum()

print(f"""
{'='*65}
DELAY CORRIDOR AUDIT — OPERATIONAL SUMMARY
{'='*65}

  Chronic corridors (delay > 20% over OSRM): {len(chronic):,} of {len(corr):,}
  Top 20 account for: {top20['sla_breach_contribution_pct'].sum():.1f}% of all SLA breaches
                       ₹{top20['revenue_at_risk'].sum():,.0f} in revenue at risk

  SEVERITY MIX (Top 20):
    🔴 Critical : {n_critical}
    🟠 High     : {n_high}
    🟣 Moderate : {(top20['severity_tier']=='Moderate').sum()}
    🟢 Watch    : {(top20['severity_tier']=='Watch').sum()}

  TOP 3 CORRIDORS BY REVENUE AT RISK ALONE:
    ₹{top3_rar:,.0f}  ({top3_breach:.1f}% of all SLA breaches)
    → If fixed first, these 3 corridors deliver the highest
      revenue recovery per intervention dollar spent.

  INTERVENTION MIX (Top 20):
{top20['recommended_intervention'].value_counts().to_string()}

  NEXT → Part 6: Baseline ETA regression model
          (use median_actual_sec / median_osrm_sec as ground-truth
           reference for trip-level feature benchmarking)
{'='*65}
""")
