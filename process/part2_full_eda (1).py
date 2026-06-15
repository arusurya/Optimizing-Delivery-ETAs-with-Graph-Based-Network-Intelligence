"""
=============================================================================
PART 2: COMPLETE LOGISTICS EDA — DELHIVERY OPERATIONS LEADERSHIP REPORT
=============================================================================
Role: Senior Logistics Analytics Consultant
Client: Delhivery Network Operations
Scope: 10-section EDA covering shipment volumes, route/hub performance,
       delay patterns, SLA breaches, temporal patterns, route types,
       correlations, and outliers.

INPUT : delivery_data.csv
OUTPUT: outputs/fig1_volume_analysis.png  through  fig6_routetype_corr_outlier.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
import calendar

warnings.filterwarnings("ignore")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────
DELHIVERY_RED  = "#E63329"
DELHIVERY_DARK = "#1A1A2E"
ACCENT_BLUE    = "#2563EB"
ACCENT_GREEN   = "#16A34A"
ACCENT_AMBER   = "#D97706"
ACCENT_PURPLE  = "#7C3AED"
NEUTRAL        = "#64748B"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.color":       "#CBD5E1",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlepad":    10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# =============================================================================
# SECTION 0 — LOAD & CLEAN REAL DATA
# =============================================================================

print("=" * 70)
print("PART 2: COMPLETE LOGISTICS EDA — DELHIVERY")
print("=" * 70)

df_raw = pd.read_csv("delivery_data.csv")
print(f"  Raw rows loaded: {len(df_raw):,}")
print(f"  Columns: {list(df_raw.columns)}")

# ── Cleaning (same pipeline as Part 1) ────────────────────────────────────
df = df_raw.copy()
df = df[(df["actual_time"] > 0) & (df["osrm_time"] > 0)]
df = df[df["source_center"] != df["destination_center"]]
df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
df = df.dropna(subset=["trip_creation_time", "source_center", "destination_center"])
print("Rows after cleaning:", len(df))
# ── Feature Engineering ────────────────────────────────────────────────────
df["delay_ratio"]        = df["actual_time"] / df["osrm_time"]
p99                      = df["delay_ratio"].quantile(0.99)
df["delay_ratio_capped"] = df["delay_ratio"].clip(upper=p99)
df["actual_time_hrs"]    = df["actual_time"] / 3600
df["osrm_time_hrs"]      = df["osrm_time"] / 3600
df["hour"]               = df["trip_creation_time"].dt.hour
df["dow"]                = df["trip_creation_time"].dt.dayofweek
df["month"]              = df["trip_creation_time"].dt.month
df["week"]               = df["trip_creation_time"].dt.isocalendar().week.astype(int)
df["is_weekend"]         = (df["dow"] >= 5).astype(int)
df["sla_breach"]         = (df["delay_ratio"] > 1.20).astype(int)
df["corridor"]           = df["source_center"] + " → " + df["destination_center"]

print(f"\n  Clean dataset: {len(df):,} rows | {df['source_center'].nunique()} hubs | "
      f"{df['corridor'].nunique()} corridors")
print(f"  Date range: {df['trip_creation_time'].min().date()} to "
      f"{df['trip_creation_time'].max().date()}")
print(f"  Median delay ratio: {df['delay_ratio'].median():.3f}x")
print(f"  SLA breach rate:    {df['sla_breach'].mean()*100:.1f}%")


# =============================================================================
# FIGURE 1 — SHIPMENT VOLUME ANALYSIS (Section 1)
# =============================================================================

fig1, axes = plt.subplots(2, 3, figsize=(20, 11))
fig1.suptitle("SECTION 1 — SHIPMENT VOLUME ANALYSIS\nDelhivery Operations Intelligence",
              fontsize=15, fontweight="bold", color=DELHIVERY_DARK, y=1.01)

# 1a. Weekly shipment volume trend
ax = axes[0, 0]
weekly = df.groupby("week").size().reset_index(name="trips")
ax.fill_between(weekly["week"], weekly["trips"], alpha=0.25, color=ACCENT_BLUE)
ax.plot(weekly["week"], weekly["trips"], color=ACCENT_BLUE, lw=2.5)
ax.set_title("Weekly Shipment Volume Trend")
ax.set_xlabel("Week of Year")
ax.set_ylabel("Trip Count")
# Mark festive season if data spans Oct-Nov
if df["month"].isin([10, 11]).any():
    ax.axvspan(40, 47, alpha=0.12, color=DELHIVERY_RED, label="Festive Season (Oct-Nov)")
    ax.legend(fontsize=8)

# 1b. Monthly volume
ax = axes[0, 1]
monthly = df.groupby("month").size()
colors_m = [DELHIVERY_RED if m in [10, 11] else ACCENT_BLUE for m in monthly.index]
bars = ax.bar(monthly.index, monthly.values, color=colors_m, edgecolor="white", width=0.7)
ax.set_title("Monthly Shipment Volume")
ax.set_xlabel("Month")
ax.set_ylabel("Trip Count")
month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.legend(handles=[mpatches.Patch(color=DELHIVERY_RED, label="Festive spike"),
                   mpatches.Patch(color=ACCENT_BLUE, label="Normal")], fontsize=8)

# 1c. Volume by route type over months
ax = axes[0, 2]
rt_month = df.groupby(["month", "route_type"]).size().unstack(fill_value=0)
rt_month.plot(kind="bar", ax=ax, color=[ACCENT_GREEN, DELHIVERY_RED],
              edgecolor="white", width=0.7, stacked=False)
ax.set_title("Monthly Volume: FTL vs Carting")
ax.set_xlabel("Month")
ax.set_ylabel("Trip Count")
ax.set_xticklabels(
    [calendar.month_abbr[m] for m in monthly.index],
    rotation=0
)
ax.legend(fontsize=9)

# 1d. Hourly volume distribution
ax = axes[1, 0]
hourly_vol = df.groupby("hour").size()
ax.bar(hourly_vol.index, hourly_vol.values,
       color=[DELHIVERY_RED if h in [8,9,10,17,18,19,20] else ACCENT_BLUE
              for h in hourly_vol.index],
       edgecolor="white")
ax.set_title("Hourly Shipment Volume\n(Red = Peak Hours)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Trip Count")
ax.set_xticks(range(0, 24, 2))

# 1e. Daily volume (day of week)
ax = axes[1, 1]
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_vol = df.groupby("dow").size()
ax.bar(dow_labels[:len(dow_vol)], dow_vol.values,
       color=[DELHIVERY_RED if i >= 5 else ACCENT_BLUE for i in range(len(dow_vol))],
       edgecolor="white")
ax.set_title("Shipment Volume by Day of Week")
ax.set_ylabel("Trip Count")

# 1f. Trip duration distribution by route type
ax = axes[1, 2]
for rt, col in [("FTL", ACCENT_GREEN), ("Carting", DELHIVERY_RED)]:
    vals = df[df["route_type"] == rt]["actual_time_hrs"].clip(0, 72)
    if len(vals) > 0:
        ax.hist(vals, bins=80, alpha=0.55, color=col, label=rt, density=True)
ax.set_title("Actual Trip Duration Distribution\n(hours)")
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "fig1_volume_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Figure 1 saved: fig1_volume_analysis.png")


# =============================================================================
# FIGURE 2 — ROUTE-WISE ANALYSIS (Section 2)
# =============================================================================

fig2, axes = plt.subplots(2, 3, figsize=(20, 11))
fig2.suptitle("SECTION 2 — ROUTE-WISE ANALYSIS\nCorridor Performance & Delay Profiling",
              fontsize=15, fontweight="bold", color=DELHIVERY_DARK, y=1.01)

corridor_stats = (
    df.groupby("corridor")
    .agg(
        trips         =("trip_uuid", "count"),
        avg_delay     =("delay_ratio", "mean"),
        med_delay     =("delay_ratio", "median"),
        pct_sla_breach=("sla_breach", "mean"),
        avg_dist      =("osrm_distance", "mean"),
    )
    .query("trips >= 15")
    .reset_index()
)

# 2a. Top 20 corridors by volume
ax = axes[0, 0]
top_vol = corridor_stats.nlargest(20, "trips")
ax.barh(range(len(top_vol)), top_vol["trips"].values[::-1],
        color=ACCENT_BLUE, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top_vol)))
ax.set_yticklabels(top_vol["corridor"].values[::-1], fontsize=7)
ax.set_title("Top 20 Corridors by Trip Volume")
ax.set_xlabel("Trip Count")

# 2b. Top 20 corridors by avg delay
ax = axes[0, 1]
top_delay = corridor_stats.nlargest(20, "avg_delay")
colors_d = [DELHIVERY_RED if v > 1.5 else ACCENT_AMBER
            for v in top_delay["avg_delay"].values]
ax.barh(range(len(top_delay)), top_delay["avg_delay"].values[::-1],
        color=colors_d[::-1], alpha=0.85, edgecolor="white")
ax.axvline(1.2, color="black", lw=1.5, ls="--", label="SLA threshold (1.2x)")
ax.set_yticks(range(len(top_delay)))
ax.set_yticklabels(top_delay["corridor"].values[::-1], fontsize=7)
ax.set_title("Top 20 Corridors by Avg Delay Ratio")
ax.set_xlabel("Mean Delay Ratio")
ax.legend(fontsize=8)

# 2c. Corridor SLA breach rate vs volume (quadrant chart)
ax = axes[0, 2]
sc = ax.scatter(
    corridor_stats["trips"],
    corridor_stats["pct_sla_breach"] * 100,
    c=corridor_stats["avg_delay"],
    cmap="RdYlGn_r", s=20, alpha=0.6, vmin=0.9, vmax=2.5
)
plt.colorbar(sc, ax=ax, label="Avg Delay Ratio")
ax.axhline(20, color="black", lw=1.5, ls="--", alpha=0.6)
ax.axvline(corridor_stats["trips"].quantile(0.75), color="black", lw=1.5, ls=":", alpha=0.6)
ax.set_title("Corridor Quadrant Map\n(Volume vs SLA Breach Rate)")
ax.set_xlabel("Trip Volume")
ax.set_ylabel("SLA Breach Rate (%)")
ax.annotate("HIGH RISK\n(high vol, high breach)", xy=(0.65, 0.85), xycoords="axes fraction",
            fontsize=8, color=DELHIVERY_RED, fontweight="bold")

# 2d. Distance vs delay scatter
ax = axes[1, 0]
sample = corridor_stats.sample(min(500, len(corridor_stats)), random_state=42)
ax.scatter(sample["avg_dist"], sample["avg_delay"],
           c=sample["pct_sla_breach"], cmap="YlOrRd", s=30, alpha=0.7)
z = np.polyfit(sample["avg_dist"], sample["avg_delay"], 1)
x_r = np.linspace(sample["avg_dist"].min(), sample["avg_dist"].max(), 100)
ax.plot(x_r, np.poly1d(z)(x_r), color=DELHIVERY_RED, lw=2,
        label=f"Trend (slope={z[0]:.4f})")
ax.set_title("Corridor Distance vs Avg Delay Ratio")
ax.set_xlabel("Avg OSRM Distance (km)")
ax.set_ylabel("Avg Delay Ratio")
ax.legend(fontsize=8)

# 2e. SLA breach rate distribution across corridors
ax = axes[1, 1]
ax.hist(corridor_stats["pct_sla_breach"] * 100, bins=50, color=ACCENT_PURPLE,
        edgecolor="white", alpha=0.85)
ax.axvline(20, color=DELHIVERY_RED, lw=2, ls="--", label="20% breach threshold")
pct_above = (corridor_stats["pct_sla_breach"] > 0.2).mean() * 100
ax.set_title(f"SLA Breach Rate Distribution\n({pct_above:.0f}% corridors exceed 20%)")
ax.set_xlabel("SLA Breach Rate (%)")
ax.set_ylabel("Number of Corridors")
ax.legend(fontsize=8)

# 2f. Corridor delay reliability (IQR spread)
ax = axes[1, 2]
corr_iqr = (
    df.groupby("corridor")["delay_ratio"]
    .quantile([0.25, 0.75])
    .unstack()
    .assign(iqr=lambda x: x[0.75] - x[0.25])
    .query("iqr > 0")
    .nlargest(25, "iqr")
)
ax.barh(range(len(corr_iqr)), corr_iqr["iqr"].values[::-1],
        color=ACCENT_AMBER, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(corr_iqr)))
ax.set_yticklabels(corr_iqr.index[::-1], fontsize=7)
ax.set_title("Top 25 Corridors: Delay Variability (IQR)\nHigh IQR = Unreliable ETA")
ax.set_xlabel("Delay Ratio IQR")

plt.tight_layout()
plt.savefig(output_dir / "fig2_route_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 2 saved: fig2_route_analysis.png")


# =============================================================================
# FIGURE 3 — HUB-WISE ANALYSIS (Section 3)
# =============================================================================

fig3, axes = plt.subplots(2, 3, figsize=(20, 11))
fig3.suptitle("SECTION 3 — HUB-WISE ANALYSIS\nFacility Performance & Bottleneck Identification",
              fontsize=15, fontweight="bold", color=DELHIVERY_DARK, y=1.01)

src_stats = (
    df.groupby("source_center")
    .agg(out_trips    =("trip_uuid", "count"),
         avg_out_delay=("delay_ratio", "mean"),
         breach_rate  =("sla_breach", "mean"))
    .reset_index()
    .rename(columns={"source_center": "hub"})
)
dst_stats = (
    df.groupby("destination_center")
    .agg(in_trips    =("trip_uuid", "count"),
         avg_in_delay=("delay_ratio", "mean"))
    .reset_index()
    .rename(columns={"destination_center": "hub"})
)
hub_stats = src_stats.merge(dst_stats, on="hub", how="outer").fillna(0)
hub_stats["total_load"]   = hub_stats["out_trips"] + hub_stats["in_trips"]
hub_stats["load_balance"] = hub_stats["out_trips"] / (hub_stats["in_trips"] + 1)
hub_stats["avg_delay"]    = (hub_stats["avg_out_delay"] + hub_stats["avg_in_delay"]) / 2

# 3a. Top 20 hubs by total load
ax = axes[0, 0]
top_load = hub_stats.nlargest(20, "total_load")
ax.barh(range(len(top_load)), top_load["total_load"].values[::-1],
        color=ACCENT_BLUE, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(top_load)))
ax.set_yticklabels(top_load["hub"].values[::-1], fontsize=8)
ax.set_title("Top 20 Hubs by Total Load")
ax.set_xlabel("Total Trips (In + Out)")

# 3b. Hub avg delay ranking
ax = axes[0, 1]
top_delayed_hubs = hub_stats[hub_stats["total_load"] >= 50].nlargest(20, "avg_delay")
ax.barh(range(len(top_delayed_hubs)),
        top_delayed_hubs["avg_delay"].values[::-1],
        color=[DELHIVERY_RED if v > 1.5 else ACCENT_AMBER
               for v in top_delayed_hubs["avg_delay"].values[::-1]],
        alpha=0.85, edgecolor="white")
ax.axvline(1.2, color="black", lw=1.5, ls="--", label="Threshold")
ax.set_yticks(range(len(top_delayed_hubs)))
ax.set_yticklabels(top_delayed_hubs["hub"].values[::-1], fontsize=8)
ax.set_title("Top 20 Hubs by Avg Delay Ratio\n(min 50 trips)")
ax.set_xlabel("Avg Delay Ratio")
ax.legend(fontsize=8)

# 3c. Load balance scatter (in vs out)
ax = axes[0, 2]
ax.scatter(hub_stats["in_trips"], hub_stats["out_trips"],
           c=hub_stats["avg_delay"], cmap="RdYlGn_r",
           s=30, alpha=0.7, vmin=0.9, vmax=2.5)
max_t = max(hub_stats["in_trips"].max(), hub_stats["out_trips"].max())
ax.plot([0, max_t], [0, max_t], "k--", lw=1.5, alpha=0.5, label="Balanced")
ax.set_title("Hub Load Balance: In vs Out Trips\n(above line = net source hub)")
ax.set_xlabel("Inbound Trips")
ax.set_ylabel("Outbound Trips")

# 3d. Hub delay symmetry
ax = axes[1, 0]
ax.scatter(hub_stats["avg_in_delay"], hub_stats["avg_out_delay"],
           alpha=0.6, s=25, color=ACCENT_PURPLE)
lim_low  = min(hub_stats["avg_in_delay"].min(), hub_stats["avg_out_delay"].min()) * 0.9
lim_high = max(hub_stats["avg_in_delay"].max(), hub_stats["avg_out_delay"].max()) * 1.1
ax.plot([lim_low, lim_high], [lim_low, lim_high], "r--", lw=1.5, alpha=0.7,
        label="Symmetric (in=out)")
ax.set_title("Hub Delay Symmetry\nInbound vs Outbound Delay")
ax.set_xlabel("Avg Inbound Delay Ratio")
ax.set_ylabel("Avg Outbound Delay Ratio")
ax.legend(fontsize=8)

# 3e. SLA breach rate per hub (top 20)
ax = axes[1, 1]
top_breach_hubs = hub_stats[hub_stats["total_load"] >= 50].nlargest(20, "breach_rate")
ax.barh(range(len(top_breach_hubs)),
        (top_breach_hubs["breach_rate"] * 100).values[::-1],
        color=DELHIVERY_RED, alpha=0.85, edgecolor="white")
ax.axvline(20, color="black", lw=1.5, ls="--")
ax.set_yticks(range(len(top_breach_hubs)))
ax.set_yticklabels(top_breach_hubs["hub"].values[::-1], fontsize=8)
ax.set_title("Top 20 Hubs by SLA Breach Rate (%)")
ax.set_xlabel("SLA Breach Rate (%)")

# 3f. Bottleneck risk matrix (load vs delay, colored by breach rate)
ax = axes[1, 2]
hub_plot = hub_stats[hub_stats["total_load"] >= 30]
sc = ax.scatter(hub_plot["total_load"], hub_plot["avg_delay"],
                c=hub_plot["breach_rate"], cmap="RdYlGn_r",
                s=40, alpha=0.75, vmin=0, vmax=0.5)
plt.colorbar(sc, ax=ax, label="SLA Breach Rate")
ax.axhline(1.2, color="black", lw=1.5, ls="--", alpha=0.6)
ax.axvline(hub_plot["total_load"].quantile(0.8), color="black", lw=1.5, ls=":", alpha=0.6)
ax.set_title("Hub Bottleneck Risk Matrix\n(Top-right = highest priority)")
ax.set_xlabel("Total Load (trips)")
ax.set_ylabel("Avg Delay Ratio")

plt.tight_layout()
plt.savefig(output_dir / "fig3_hub_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 3 saved: fig3_hub_analysis.png")


# =============================================================================
# FIGURE 4 — DELAY + SLA BREACH ANALYSIS (Sections 4 & 5)
# =============================================================================

fig4, axes = plt.subplots(2, 3, figsize=(20, 11))
fig4.suptitle("SECTIONS 4 & 5 — DELAY ANALYSIS & SLA BREACH DEEP DIVE",
              fontsize=15, fontweight="bold", color=DELHIVERY_DARK, y=1.01)

# 4a. Delay ratio distribution with colour-coded zones
ax = axes[0, 0]
vals = df["delay_ratio"].clip(0, 5)
n, bins, patches = ax.hist(vals, bins=100, edgecolor="none")
for patch, left in zip(patches, bins[:-1]):
    if left < 0.9:   patch.set_facecolor(ACCENT_BLUE)
    elif left < 1.0: patch.set_facecolor(ACCENT_GREEN)
    elif left < 1.2: patch.set_facecolor(ACCENT_AMBER)
    else:            patch.set_facecolor(DELHIVERY_RED)
ax.axvline(1.0, color="black",      lw=2, ls="--")
ax.axvline(1.2, color=DELHIVERY_RED, lw=2, ls="--", label="SLA threshold")
ax.set_title("Delay Ratio Distribution\n(Blue=Early, Green=OnTime, Amber=Marginal, Red=Breach)")
ax.set_xlabel("Delay Ratio (actual/OSRM)")
ax.set_ylabel("Count")
ax.legend(fontsize=8)

zones = {
    "Early (<0.9)":       (df["delay_ratio"] < 0.9).mean() * 100,
    "On Time (0.9–1.0)":  ((df["delay_ratio"] >= 0.9) & (df["delay_ratio"] < 1.0)).mean() * 100,
    "Marginal (1.0–1.2)": ((df["delay_ratio"] >= 1.0) & (df["delay_ratio"] < 1.2)).mean() * 100,
    "Breach (>1.2)":      (df["delay_ratio"] >= 1.2).mean() * 100,
}

# 4b. Delay zone pie
ax = axes[0, 1]
ax.pie(zones.values(), labels=zones.keys(),
       colors=[ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, DELHIVERY_RED],
       autopct="%1.1f%%", startangle=90,
       wedgeprops=dict(edgecolor="white", linewidth=2))
ax.set_title("Trip Delay Zone Breakdown")

# 4c. Actual vs OSRM time scatter
ax = axes[0, 2]
sample = df.sample(min(8000, len(df)), random_state=42)
ax.scatter(sample["osrm_time_hrs"], sample["actual_time_hrs"],
           c=sample["delay_ratio"].clip(0.5, 3), cmap="RdYlGn_r",
           s=5, alpha=0.4, vmin=0.8, vmax=2.2)
lim = max(sample["osrm_time_hrs"].max(), sample["actual_time_hrs"].max())
ax.plot([0, lim], [0, lim],       "k--", lw=2,   label="Perfect (y=x)")
ax.plot([0, lim], [0, lim * 1.2], color=DELHIVERY_RED, lw=1.5, ls=":", label="+20% SLA line")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_title("Actual vs OSRM Time\n(Color = Delay Severity)")
ax.set_xlabel("OSRM Predicted (hrs)")
ax.set_ylabel("Actual Duration (hrs)")
ax.legend(fontsize=8)

# 4d. SLA breach rate by month
ax = axes[1, 0]
sla_monthly = df.groupby("month")["sla_breach"].mean() * 100
ax.bar(sla_monthly.index, sla_monthly.values,
       color=[DELHIVERY_RED if m in [10, 11, 12] else ACCENT_AMBER
              for m in sla_monthly.index],
       edgecolor="white")
ax.axhline(sla_monthly.mean(), color="black", lw=1.5, ls="--",
           label=f"Annual avg ({sla_monthly.mean():.1f}%)")
ax.set_title("SLA Breach Rate by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Breach Rate (%)")
ax.legend(fontsize=8)

# 4e. Delay percentile ladder
ax = axes[1, 1]
pcts      = [50, 60, 70, 80, 90, 95, 99]
pct_vals  = [df["delay_ratio"].quantile(p/100) for p in pcts]
colors_pct = [ACCENT_GREEN if v < 1.2 else ACCENT_AMBER if v < 1.5 else DELHIVERY_RED
              for v in pct_vals]
bars = ax.bar([f"P{p}" for p in pcts], pct_vals, color=colors_pct, edgecolor="white")
ax.axhline(1.2, color="black", lw=1.5, ls="--", label="SLA threshold (1.2x)")
ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
ax.set_title("Delay Ratio Percentile Ladder")
ax.set_ylabel("Delay Ratio")
ax.legend(fontsize=8)

# 4f. SLA breach % by route type and month
ax = axes[1, 2]
pivot = df.groupby(["month", "route_type"])["sla_breach"].mean().unstack() * 100
pivot.plot(kind="line", ax=ax, color=[ACCENT_GREEN, DELHIVERY_RED],
           marker="o", lw=2.5, markersize=6)
ax.set_title("SLA Breach Rate by Route Type & Month")
ax.set_xlabel("Month")
ax.set_ylabel("Breach Rate (%)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "fig4_delay_sla.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 4 saved: fig4_delay_sla.png")


# =============================================================================
# FIGURE 5 — TEMPORAL PATTERNS (Sections 6 & 7)
# =============================================================================

fig5, axes = plt.subplots(2, 3, figsize=(20, 11))
fig5.suptitle("SECTIONS 6 & 7 — TIME-OF-DAY & DAY-OF-WEEK PATTERNS",
              fontsize=15, fontweight="bold", color=DELHIVERY_DARK, y=1.01)

dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# 5a. Hourly delay ratio (with volume overlay)
ax1_twin = axes[0, 0]
ax2_twin = ax1_twin.twinx()
hourly = df.groupby("hour").agg(med_delay=("delay_ratio", "median"),
                                 vol=("trip_uuid", "count")).reset_index()
ax1_twin.bar(hourly["hour"], hourly["vol"], color="#CBD5E1", alpha=0.6, label="Volume")
ax2_twin.plot(hourly["hour"], hourly["med_delay"], color=DELHIVERY_RED, lw=2.5,
              marker="o", markersize=6, label="Median Delay")
ax2_twin.axhline(1.0, color="black", lw=1, ls="--")
ax2_twin.axhline(1.2, color=DELHIVERY_RED, lw=1, ls=":", alpha=0.6)
ax1_twin.set_title("Hourly: Volume vs Median Delay Ratio")
ax1_twin.set_xlabel("Hour of Day")
ax1_twin.set_ylabel("Trip Volume", color="#64748B")
ax2_twin.set_ylabel("Median Delay Ratio", color=DELHIVERY_RED)
ax1_twin.set_xticks(range(0, 24, 2))
lines1, labels1 = ax1_twin.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax1_twin.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

# 5b. Hourly SLA breach rate heatmap
ax = axes[0, 1]
heat_data = df.groupby(["hour", "route_type"])["sla_breach"].mean().unstack() * 100
sns.heatmap(heat_data.T, ax=ax, cmap="YlOrRd", annot=False,
            cbar_kws={"label": "Breach Rate (%)"}, linewidths=0)
ax.set_title("Hourly SLA Breach Rate\nby Route Type")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("")

# 5c. Day-of-week delay pattern
ax = axes[0, 2]
dow_delay = df.groupby("dow")["delay_ratio"].agg(["median", "mean"])
x = range(len(dow_delay))
ax.bar(x, dow_delay["median"].values,
       color=[DELHIVERY_RED if i >= 5 else ACCENT_BLUE for i in x],
       alpha=0.75, edgecolor="white", label="Median")
ax.plot(x, dow_delay["mean"].values, "ko--", lw=2, markersize=7, label="Mean")
ax.axhline(1.0, color="black", lw=1, ls="--")
ax.axhline(1.2, color=DELHIVERY_RED, lw=1, ls=":", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(dow_labels[:len(dow_delay)])
ax.set_title("Delay Ratio by Day of Week\n(Red = Weekends)")
ax.set_ylabel("Delay Ratio")
ax.legend(fontsize=8)

# 5d. Hour × Day heatmap
ax = axes[1, 0]
heat2 = df.groupby(["dow", "hour"])["delay_ratio"].median().unstack()
sns.heatmap(heat2, ax=ax, cmap="RdYlGn_r", vmin=0.85, vmax=1.6,
            cbar_kws={"label": "Median Delay Ratio"},
            xticklabels=2, yticklabels=dow_labels[:len(heat2)])
ax.set_title("Delay Heatmap: Day × Hour\n(Dark Red = Worst, Green = Best)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day of Week")

# 5e. Weekend vs weekday distribution
ax = axes[1, 1]
for label, val, col in [("Weekday", 0, ACCENT_BLUE), ("Weekend", 1, DELHIVERY_RED)]:
    vals = df[df["is_weekend"] == val]["delay_ratio"].clip(0, 5)
    ax.hist(vals, bins=60, alpha=0.55, color=col, label=label, density=True)
ax.axvline(1.2, color="black", lw=1.5, ls="--")
wkd = df[df["is_weekend"] == 0]["delay_ratio"]
wke = df[df["is_weekend"] == 1]["delay_ratio"]
if len(wkd) > 0 and len(wke) > 0:
    _, p = stats.mannwhitneyu(wkd, wke)
    ax.set_title(f"Weekend vs Weekday Delay Distribution\n(Mann-Whitney p={p:.4f})")
else:
    ax.set_title("Weekend vs Weekday Delay Distribution")
ax.set_xlabel("Delay Ratio")
ax.legend(fontsize=9)

# 5f. Hourly SLA breach rate by route type (area chart)
ax = axes[1, 2]
hourly_breach = df.groupby(["hour", "route_type"])["sla_breach"].mean().unstack() * 100
hourly_breach.plot(kind="area", ax=ax, alpha=0.5,
                   color=[ACCENT_GREEN, DELHIVERY_RED], stacked=False)
ax.set_title("Hourly SLA Breach Rate by Route Type")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Breach Rate (%)")
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "fig5_temporal_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 5 saved: fig5_temporal_patterns.png")


# =============================================================================
# FIGURE 6 — ROUTE TYPE, CORRELATION & OUTLIERS (Sections 8–10)
# =============================================================================

fig6, axes = plt.subplots(2, 3, figsize=(20, 11))
fig6.suptitle("SECTIONS 8–10 — ROUTE TYPE, CORRELATION & OUTLIER ANALYSIS",
              fontsize=15, fontweight="bold", color=DELHIVERY_DARK, y=1.01)

# 8a. FTL vs Carting: delay ratio box plot
ax = axes[0, 0]
ftl_vals  = df[df["route_type"] == "FTL"]["delay_ratio"].clip(0.5, 4)
cart_vals = df[df["route_type"] == "Carting"]["delay_ratio"].clip(0.5, 4)
bp = ax.boxplot([ftl_vals, cart_vals], tick_labels=["FTL", "Carting"],
                patch_artist=True, notch=True,
                medianprops={"color": "black", "lw": 2})
bp["boxes"][0].set_facecolor(ACCENT_GREEN)
bp["boxes"][1].set_facecolor(DELHIVERY_RED)
for patch in bp["boxes"]: patch.set_alpha(0.7)
ax.axhline(1.2, color="black", lw=1.5, ls="--", alpha=0.6, label="SLA threshold")
ax.set_title("Delay Ratio: FTL vs Carting\n(Box = IQR, Notch = 95% CI of Median)")
ax.set_ylabel("Delay Ratio")
ax.legend(fontsize=8)

# 8b. FTL vs Carting: SLA breach rate by time bucket
ax = axes[0, 1]
def assign_tod(h):
    if h < 6: return "Night"
    if h < 12: return "Morn Peak"
    if h < 17: return "Afternoon"
    return "Eve Peak"
df["time_bucket"] = df["hour"].map(assign_tod)
tod_order = ["Night", "Morn Peak", "Afternoon", "Eve Peak"]
tod_rt = (
    df.groupby(["time_bucket", "route_type"])["sla_breach"].mean() * 100
).unstack()
tod_rt = tod_rt.reindex([t for t in tod_order if t in tod_rt.index])
tod_rt.plot(kind="bar", ax=ax, color=[ACCENT_GREEN, DELHIVERY_RED],
            edgecolor="white", width=0.6)
ax.set_title("SLA Breach Rate by Time-of-Day & Route Type")
ax.set_xlabel("Time Bucket")
ax.set_ylabel("Breach Rate (%)")
ax.set_xticklabels(tod_rt.index, rotation=30)
ax.legend(fontsize=9)

# 8c. FTL vs Carting: distance distribution
ax = axes[0, 2]
for rt, col in [("FTL", ACCENT_GREEN), ("Carting", DELHIVERY_RED)]:
    vals = df[df["route_type"] == rt]["osrm_distance"].dropna().clip(0, 2000)
    if len(vals) > 0:
        ax.hist(vals, bins=80, alpha=0.55, color=col, label=rt, density=True)
ax.set_title("Trip Distance Distribution: FTL vs Carting")
ax.set_xlabel("OSRM Distance (km)")
ax.set_ylabel("Density")
ax.legend(fontsize=9)

# 9. Correlation heatmap
ax = axes[1, 0]
num_cols = [c for c in ["osrm_time_hrs", "osrm_distance", "actual_time_hrs",
                         "delay_ratio", "sla_breach", "hour", "dow", "is_weekend"]
            if c in df.columns]
corr_mat = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
short_labels = [c.replace("osrm_time_hrs","OSRM_t").replace("osrm_distance","OSRM_d")
                 .replace("actual_time_hrs","Act_t").replace("delay_ratio","delay_r")
                 .replace("sla_breach","SLA").replace("is_weekend","wknd")
                for c in num_cols]
sns.heatmap(corr_mat, ax=ax, mask=mask, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
            xticklabels=short_labels, yticklabels=short_labels)
ax.set_title("Correlation Matrix\n(Key Numerical Features)")

# 10a. Outlier detection: delay ratio Z-score
ax = axes[1, 1]
z_scores = np.abs(stats.zscore(df["delay_ratio"].dropna()))
df_z = df.dropna(subset=["delay_ratio"]).copy()
df_z["z_score"] = z_scores
ax.hist(df_z["delay_ratio"].clip(0, 8), bins=100, color="#CBD5E1",
        edgecolor="none", label="All trips")
for t, col_t in [(2, ACCENT_AMBER), (3, DELHIVERY_RED), (4, ACCENT_PURPLE)]:
    outliers = df_z[df_z["z_score"] > t]
    pct = len(outliers) / len(df_z) * 100
    threshold_val = outliers["delay_ratio"].min() if len(outliers) > 0 else 999
    if threshold_val < 8:
        ax.axvline(threshold_val, color=col_t, lw=2, ls="--",
                   label=f"Z>{t}: {len(outliers):,} trips ({pct:.1f}%)")
ax.set_title("Outlier Detection: Delay Ratio\n(Z-score thresholds)")
ax.set_xlabel("Delay Ratio")
ax.set_ylabel("Count")
ax.legend(fontsize=8)
ax.set_xlim(0, 8)

# 10b. IQR-based outlier corridors
ax = axes[1, 2]
q1_v, q3_v = df["delay_ratio"].quantile([0.25, 0.75])
iqr_v = q3_v - q1_v
df["is_outlier"] = (df["delay_ratio"] > q3_v + 3 * iqr_v).astype(int)
outlier_hubs = (
    df[df["is_outlier"] == 1]
    .groupby("source_center").size()
    .nlargest(20)
)
ax.barh(range(len(outlier_hubs)), outlier_hubs.values[::-1],
        color=DELHIVERY_RED, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(outlier_hubs)))
ax.set_yticklabels(outlier_hubs.index[::-1], fontsize=8)
ax.set_title("Top 20 Source Hubs: Extreme Delay Trips\n(IQR-based outliers, Q3 + 3×IQR)")
ax.set_xlabel("Outlier Trip Count")

plt.tight_layout()
plt.savefig(output_dir / "fig6_routetype_corr_outlier.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 6 saved: fig6_routetype_corr_outlier.png")


# =============================================================================
# OPERATIONS INTELLIGENCE SUMMARY
# =============================================================================

total       = len(df)
breach_pct  = df["sla_breach"].mean() * 100
med_delay   = df["delay_ratio"].median()
worst_hub   = hub_stats.loc[hub_stats["avg_delay"].idxmax(), "hub"]
worst_corr  = corridor_stats.loc[corridor_stats["avg_delay"].idxmax(), "corridor"]
ftl_breach  = df[df["route_type"] == "FTL"]["sla_breach"].mean() * 100
cart_breach = df[df["route_type"] == "Carting"]["sla_breach"].mean() * 100

festive_breach = df[df["month"].isin([10, 11])]["sla_breach"].mean() * 100 \
    if df["month"].isin([10, 11]).any() else 0
normal_breach  = df[~df["month"].isin([10, 11])]["sla_breach"].mean() * 100

print("\n" + "=" * 70)
print("  DELHIVERY OPERATIONS INTELLIGENCE SUMMARY")
print("=" * 70)
print(f"""
  Total trips analyzed:    {total:>10,}
  Unique hubs:             {df['source_center'].nunique():>10,}
  Unique corridors:        {df['corridor'].nunique():>10,}

  SLA breach rate:         {breach_pct:>9.1f}%
  Median delay ratio:      {med_delay:>10.3f}x
  Festive SLA breach:      {festive_breach:>9.1f}%
  Normal period breach:    {normal_breach:>9.1f}%

  FTL SLA breach rate:     {ftl_breach:>9.1f}%
  Carting SLA breach rate: {cart_breach:>9.1f}%

  Worst hub (avg delay):   {worst_hub}
  Worst corridor (delay):  {worst_corr[:50]}
""")

print("=" * 70)
print("OUTPUT FILES:")
for f in ["fig1_volume_analysis.png", "fig2_route_analysis.png",
          "fig3_hub_analysis.png", "fig4_delay_sla.png",
          "fig5_temporal_patterns.png", "fig6_routetype_corr_outlier.png"]:
    print(f"  outputs/{f}")
print("=" * 70)
