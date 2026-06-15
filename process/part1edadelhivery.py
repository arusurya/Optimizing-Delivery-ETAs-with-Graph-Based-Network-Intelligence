
# PART 1: DATA UNDERSTANDING & EDA

# SECTION 0: ENVIRONMENT SETUP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("PART 1: DATA UNDERSTANDING & EDA — DELHIVERY LOGISTICS")
print("=" * 70)


# SECTION 1: SCHEMA DEFINITION & COLUMN BUSINESS MEANING
# NOTE: origin_lat/lng and destination_lat/lng do NOT exist in delivery_data.csv
# They have been removed from the schema below. All analysis uses only real columns.

print("\n[1] COLUMN SCHEMA WITH BUSINESS MEANING")
print("-" * 50)

schema = {
    "data":                     ("Shipment Features", "Date of the trip segment record"),
    "trip_uuid":                ("Shipment Features", "Unique identifier for a full trip (source → dest)."),
    "route_type":               ("Route Features",    "FTL (Full Truck Load) or Carting. Determines vehicle type and SLA."),
    "trip_creation_time":       ("Time Features",     "Timestamp when the trip was created/booked."),
    "route_schedule_uuid":      ("Shipment Features", "UUID of the route schedule this trip belongs to."),

    "source_center":            ("Hub Features",      "Origin facility code. NODE in the logistics graph."),
    "source_name":              ("Hub Features",      "Human-readable name of the source facility."),
    "destination_center":       ("Hub Features",      "Destination facility code. NODE in the logistics graph."),
    "destination_name":         ("Hub Features",      "Human-readable name of the destination."),

    "od_start_time":            ("Time Features",     "Actual start time of the OD segment."),
    "od_end_time":              ("Time Features",     "Actual end time of the OD segment."),
    "start_scan_to_end_scan":   ("Time Features",     "Actual segment duration (seconds) from first scan to last scan."),
    "is_cutoff":                ("Route Features",    "Whether the shipment missed a cutoff window."),
    "cutoff_factor":            ("Route Features",    "Factor related to cutoff status."),
    "cutoff_timestamp":         ("Time Features",     "Timestamp of the cutoff window."),

    "actual_distance_to_destination": ("Route Features", "Actual GPS-traced distance (km). Ground truth."),
    "actual_time":              ("Time Features",     "Actual travel duration (seconds). TARGET VARIABLE."),
    "osrm_time":                ("Route Features",    "OSRM-predicted travel time (seconds). Delhivery's CURRENT baseline."),
    "osrm_distance":            ("Route Features",    "OSRM-predicted distance (km)."),
    "factor":                   ("Route Features",    "Overall trip-level delay factor = actual_time / osrm_time."),

    "segment_actual_time":      ("Route Features",    "Actual time for this specific segment."),
    "segment_osrm_time":        ("Route Features",    "OSRM time for this segment."),
    "segment_osrm_distance":    ("Route Features",    "OSRM distance for this segment."),
    "segment_factor":           ("Route Features",    "Ratio of actual_segment_time / osrm_segment_time. PRIMARY DELAY METRIC."),
}

col_df = pd.DataFrame(schema, index=["Category", "Business Meaning"]).T
col_df.index.name = "Column"
print(col_df.to_string())


# SECTION 2: COLUMN CATEGORIZATION
print("\n\n[2] COLUMN CATEGORIZATION BY USE CASE")
print("-" * 50)

categories = {}
for col, (cat, _) in schema.items():
    categories.setdefault(cat, []).append(col)

for cat, cols in categories.items():
    print(f"\n  ▸ {cat.upper()}:")
    for c in cols:
        print(f"    - {c}")


# SECTION 3: LOAD DATA
print("\n\n[3] LOADING delivery_data.csv")
print("-" * 50)

df = pd.read_csv("delivery_data.csv")
print(f"  Raw rows loaded: {len(df):,}")
print(f"  Columns found:   {list(df.columns)}")
print(f"\n  Basic info:")
print(df.dtypes.to_string())


# SECTION 4: DATA QUALITY AUDIT
print("\n\n[4] DATA QUALITY AUDIT")
print("-" * 50)

# 4a. Missing Values
print("\n  4a. Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Count": missing, "Pct (%)": missing_pct})
missing_df = missing_df[missing_df["Count"] > 0]
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("  No missing values found.")

# 4b. Duplicates
print("\n  4b. Duplicate trip_uuid:")
n_dups = df.duplicated(subset="trip_uuid").sum()
print(f"  Duplicate rows (same trip_uuid): {n_dups} ({n_dups/len(df)*100:.1f}%)")

# 4c. Negative Durations
print("\n  4c. Negative Durations:")
neg_time = (df["actual_time"] < 0).sum()
print(f"  Rows with actual_time < 0: {neg_time} ({neg_time/len(df)*100:.1f}%)")
if "actual_distance_to_destination" in df.columns:
    neg_dist = (df["actual_distance_to_destination"] < 0).sum()
    print(f"  Rows with actual_distance < 0: {neg_dist}")

# 4d. Zero OSRM time
print("\n  4d. Zero OSRM time (division by zero risk):")
zero_osrm = (df["osrm_time"] <= 0).sum()
print(f"  Rows with osrm_time ≤ 0: {zero_osrm}")

# 4e. Outlier Detection
print("\n  4e. Delay Factor Outliers:")
clean_tmp = df[(df["actual_time"] > 0) & (df["osrm_time"] > 0)].copy()
clean_tmp["delay_ratio"] = clean_tmp["actual_time"] / clean_tmp["osrm_time"]
q1, q3 = clean_tmp["delay_ratio"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 3 * iqr
outlier_count = (clean_tmp["delay_ratio"] > upper_fence).sum()
print(f"  IQR fence (Q3 + 3×IQR): {upper_fence:.2f}x")
print(f"  Outlier rows (delay_ratio > {upper_fence:.1f}): {outlier_count} ({outlier_count/len(clean_tmp)*100:.1f}%)")

# 4f. route_type validity
print("\n  4f. route_type validity check:")
valid_types = {"FTL", "Carting"}
invalid_rt = (~df["route_type"].isin(valid_types)).sum()
print(f"  Invalid route_type values: {invalid_rt}")
print(f"  Value counts:\n{df['route_type'].value_counts().to_string()}")


# SECTION 5: CLEANING PIPELINE
print("\n\n[5] CLEANING PIPELINE")
print("-" * 50)

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproducible cleaning pipeline.
    Each step is logged with row count delta for auditability.
    No synthetic data. No coordinate columns (not in dataset).
    """
    n0 = len(df)
    print(f"  Starting rows: {n0:,}")

    # Step 1: Drop rows where actual_time or osrm_time is invalid
    df = df[(df["actual_time"] > 0) & (df["osrm_time"] > 0)]
    print(f"  After dropping negative/zero times: {len(df):,} (removed {n0-len(df):,})")

    # Step 2: Drop self-loops (source == destination)
    n_before = len(df)
    df = df[df["source_center"] != df["destination_center"]]
    print(f"  After dropping self-loops: {len(df):,} (removed {n_before-len(df):,})")

    # Step 3: Compute delay_ratio
    df = df.copy()
    df["delay_ratio"] = df["actual_time"] / df["osrm_time"]

    # Step 4: Cap outliers at 99th percentile for modeling
    p99 = df["delay_ratio"].quantile(0.99)
    df["delay_ratio_capped"] = df["delay_ratio"].clip(upper=p99)
    print(f"  delay_ratio P99 cap: {p99:.2f}x (raw column preserved for bottleneck analysis)")

    # Step 5: Parse timestamps and extract time features
    df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"], errors="coerce")
    df["hour_of_day"] = df["trip_creation_time"].dt.hour
    df["day_of_week"] = df["trip_creation_time"].dt.dayofweek
    df["month"]       = df["trip_creation_time"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["sla_breach"]  = (df["delay_ratio"] > 1.20).astype(int)

    # Step 6: Drop rows with unparseable timestamps
    n_before = len(df)
    df = df.dropna(subset=["trip_creation_time"])
    print(f"  After dropping bad timestamps: {len(df):,} (removed {n_before-len(df):,})")

    print(f"\n  COLUMNS IN CLEAN DATASET:")
    print(df.columns.tolist())
    print(f"\n  Final clean rows: {len(df):,}")
    return df

df_clean = clean_pipeline(df.copy())

# Save cleaned dataset
df_clean.to_csv(output_dir / "delivery_data_clean.csv", index=False)
print(f"\n  ✓ Cleaned dataset saved to outputs/delivery_data_clean.csv")

# Summary stats
print("\n\n[5b] SUMMARY STATISTICS")
print("-" * 50)
numeric_cols = ["actual_time", "osrm_time", "delay_ratio", "osrm_distance"]
existing_cols = [c for c in numeric_cols if c in df_clean.columns]
print(df_clean[existing_cols].describe().round(3).to_string())


# SECTION 6: COLUMN USE-CASE MAPPING
print("\n\n[6] COLUMN → USE CASE MAPPING")
print("-" * 50)

use_cases = {
    "Build the directed graph (nodes + edges)": [
        "source_center (node)",
        "destination_center (node)",
        "route_type (edge attribute)",
        "delay_ratio → edge weight",
        "NOTE: No lat/lng coordinates in dataset — graph is topological only",
    ],
    "Compute delay ratios per corridor": [
        "actual_time (numerator)",
        "osrm_time (denominator)",
        "delay_ratio = actual_time / osrm_time",
        "Stratify by: route_type, hour_of_day, day_of_week",
    ],
    "Detect bottleneck hubs": [
        "source_center + destination_center → build graph → compute betweenness centrality",
        "delay_ratio > 1.2 (20% threshold per PS) → chronic delay flag",
        "in-degree / out-degree per hub → load imbalance indicator",
    ],
    "Train ETA prediction models": [
        "osrm_time (baseline feature)",
        "osrm_distance",
        "route_type (categorical)",
        "hour_of_day, day_of_week, is_weekend",
        "actual_time (target for regression)",
    ],
}

for use_case, cols in use_cases.items():
    print(f"\n  ▸ {use_case}:")
    for c in cols:
        print(f"    - {c}")


# SECTION 7: EXPLORATORY DATA ANALYSIS VISUALIZATIONS
print("\n\n[7] RUNNING EDA VISUALIZATIONS")
print("-" * 50)

df_plot = df_clean.copy()

fig = plt.figure(figsize=(22, 28))
fig.suptitle("Delhivery Logistics — Part 1: EDA Dashboard", fontsize=16, y=0.98)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

# 7.1: Delay Ratio Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df_plot["delay_ratio"].clip(0, 5), bins=80, color="#2563eb", edgecolor="white", alpha=0.85)
ax1.axvline(1.0, color="red", lw=2, ls="--", label="No delay (ratio=1)")
ax1.axvline(df_plot["delay_ratio"].median(), color="orange", lw=2, ls="-",
            label=f"Median={df_plot['delay_ratio'].median():.2f}")
ax1.axvline(1.2, color="purple", lw=1.5, ls=":", label="SLA threshold (1.2x)")
ax1.set_title("Delay Ratio Distribution\n(actual / OSRM time)")
ax1.set_xlabel("Delay Ratio")
ax1.set_ylabel("Count")
ax1.legend(fontsize=8)

# 7.2: Delay Ratio by Route Type
ax2 = fig.add_subplot(gs[0, 1])
for rt, color in [("FTL", "#16a34a"), ("Carting", "#dc2626")]:
    subset = df_plot[df_plot["route_type"] == rt]["delay_ratio"].clip(0, 5)
    if len(subset) > 0:
        ax2.hist(subset, bins=60, alpha=0.65, label=rt, color=color, edgecolor="white")
ax2.axvline(1.0, color="black", lw=1.5, ls="--")
ax2.set_title("Delay Ratio by Route Type")
ax2.set_xlabel("Delay Ratio")
ax2.set_ylabel("Count")
ax2.legend()

# 7.3: Delay Ratio by Hour of Day
ax3 = fig.add_subplot(gs[0, 2])
hourly = df_plot.groupby("hour_of_day")["delay_ratio"].median()
ax3.bar(hourly.index, hourly.values, color="#7c3aed", alpha=0.85, edgecolor="white")
ax3.axhline(1.0, color="red", lw=1.5, ls="--", label="Baseline (1.0)")
ax3.set_title("Median Delay Ratio by Hour of Day")
ax3.set_xlabel("Hour")
ax3.set_ylabel("Median Delay Ratio")
ax3.legend()

# 7.4: Actual vs OSRM Time scatter
ax4 = fig.add_subplot(gs[1, 0])
sample = df_plot.sample(min(5000, len(df_plot)), random_state=42)
ax4.scatter(sample["osrm_time"] / 3600, sample["actual_time"] / 3600,
            alpha=0.25, s=6, color="#0891b2")
max_val = max(sample["osrm_time"].max(), sample["actual_time"].max()) / 3600
ax4.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect prediction (y=x)")
ax4.plot([0, max_val], [0, max_val * 1.2], "purple", lw=1.5, ls=":", label="+20% threshold")
ax4.set_title("Actual vs OSRM Time\n(hours)")
ax4.set_xlabel("OSRM Time (hrs)")
ax4.set_ylabel("Actual Time (hrs)")
ax4.set_xlim(0, max_val)
ax4.set_ylim(0, max_val)
ax4.legend(fontsize=8)

# 7.5: Top 20 Source Hubs by Volume
ax5 = fig.add_subplot(gs[1, 1])
top_hubs = df_plot["source_center"].value_counts().head(20)
ax5.barh(top_hubs.index[::-1], top_hubs.values[::-1], color="#059669", alpha=0.85)
ax5.set_title("Top 20 Source Hubs by Trip Volume")
ax5.set_xlabel("Trip Count")
ax5.tick_params(axis='y', labelsize=8)

# 7.6: Top 20 Corridors by Avg Delay
ax6 = fig.add_subplot(gs[1, 2])
corridor_delay = (
    df_plot
    .assign(corridor=df_plot["source_center"] + " → " + df_plot["destination_center"])
    .groupby("corridor")["delay_ratio"]
    .agg(["mean", "count"])
    .query("count >= 20")
    .sort_values("mean", ascending=False)
    .head(20)
)
ax6.barh(corridor_delay.index[::-1], corridor_delay["mean"].values[::-1],
         color="#dc2626", alpha=0.85)
ax6.axvline(1.2, color="black", lw=1.5, ls="--", label="20% threshold")
ax6.set_title("Top 20 Corridors by Avg Delay Ratio\n(min 20 trips)")
ax6.set_xlabel("Mean Delay Ratio")
ax6.tick_params(axis='y', labelsize=7)
ax6.legend(fontsize=8)

# 7.7: Delay by Day of Week
ax7 = fig.add_subplot(gs[2, 0])
dow_labels_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_delay = df_plot.groupby("day_of_week")["delay_ratio"].median()
colors_dow = ["#dc2626" if i >= 5 else "#2563eb" for i in dow_delay.index]
ax7.bar(dow_labels_list[:len(dow_delay)], dow_delay.values,
        color=colors_dow, alpha=0.85, edgecolor="white")
ax7.axhline(1.0, color="black", lw=1.5, ls="--")
ax7.set_title("Median Delay Ratio by Day of Week")
ax7.set_ylabel("Median Delay Ratio")

# 7.8: Route Type Mix
ax8 = fig.add_subplot(gs[2, 1])
rt_counts = df_plot["route_type"].value_counts()
ax8.pie(rt_counts.values, labels=rt_counts.index, autopct="%1.1f%%",
        colors=["#16a34a", "#dc2626"], startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
ax8.set_title("Route Type Mix\n(FTL vs Carting)")

# 7.9: Delay Ratio vs OSRM Distance
ax9 = fig.add_subplot(gs[2, 2])
if "osrm_distance" in df_plot.columns:
    sample2 = df_plot.dropna(subset=["osrm_distance"]).sample(
        min(5000, len(df_plot)), random_state=7)
    ax9.scatter(sample2["osrm_distance"], sample2["delay_ratio"].clip(0, 5),
                alpha=0.2, s=5, color="#7c3aed")
    valid = sample2.dropna(subset=["osrm_distance"])
    if len(valid) > 2:
        z = np.polyfit(valid["osrm_distance"], valid["delay_ratio"].clip(0, 5), 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid["osrm_distance"].min(), valid["osrm_distance"].max(), 100)
        ax9.plot(x_line, p(x_line), "r-", lw=2, label=f"Trend (slope={z[0]:.4f})")
    ax9.axhline(1.0, color="black", lw=1, ls="--")
    ax9.set_title("Delay Ratio vs Corridor Distance")
    ax9.set_xlabel("OSRM Distance (km)")
    ax9.set_ylabel("Delay Ratio")
    ax9.legend(fontsize=8)
else:
    ax9.text(0.5, 0.5, "osrm_distance\nnot available", ha="center", va="center",
             transform=ax9.transAxes)
    ax9.set_title("Delay Ratio vs Distance")

# 7.10: Hub In-Degree vs Avg Delay (bottleneck precursor)
ax10 = fig.add_subplot(gs[3, 0:2])
hub_stats_plot = (
    df_plot.groupby("destination_center")
    .agg(in_degree=("source_center", "count"),
         avg_delay=("delay_ratio", "mean"))
    .reset_index()
    .query("in_degree >= 10")
)
sc = ax10.scatter(hub_stats_plot["in_degree"], hub_stats_plot["avg_delay"],
                  c=hub_stats_plot["avg_delay"], cmap="RdYlGn_r",
                  s=40, alpha=0.75, vmin=0.8, vmax=2.5)
plt.colorbar(sc, ax=ax10, label="Avg Delay Ratio")
ax10.axhline(1.2, color="black", lw=1.5, ls="--", label="20% delay threshold")
ax10.set_title("Hub In-Degree vs Avg Delay Ratio\n"
               "(Color = delay severity; top-right = bottleneck candidates)")
ax10.set_xlabel("In-Degree (number of incoming corridors)")
ax10.set_ylabel("Average Delay Ratio")
ax10.legend(fontsize=9)

# 7.11: Monthly Seasonality
ax11 = fig.add_subplot(gs[3, 2])
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly = df_plot.groupby("month")["delay_ratio"].median()
ax11.plot(monthly.index, monthly.values, "o-", color="#0891b2", lw=2, markersize=8)
ax11.fill_between(monthly.index, 1.0, monthly.values, alpha=0.2, color="#0891b2")
ax11.axhline(1.0, color="red", lw=1.5, ls="--")
ax11.set_xticks(range(1, len(monthly) + 1))
if len(monthly) == 12:
    ax11.set_xticklabels(month_labels, rotation=45, fontsize=8)
ax11.set_title("Monthly Seasonality in Delay")
ax11.set_ylabel("Median Delay Ratio")

plt.savefig(output_dir / "part1_eda_dashboard.png", dpi=150, bbox_inches="tight")
print("\n  ✓ EDA dashboard saved to outputs/part1_eda_dashboard.png")
plt.close()


# SECTION 8: STATISTICAL TESTS
print("\n\n[8] STATISTICAL VALIDATION")
print("-" * 50)

ftl  = df_clean[df_clean["route_type"] == "FTL"]["delay_ratio"].dropna()
cart = df_clean[df_clean["route_type"] == "Carting"]["delay_ratio"].dropna()

if len(ftl) > 0 and len(cart) > 0:
    u_stat, p_val = stats.mannwhitneyu(ftl, cart, alternative="two-sided")
    print(f"\n  Mann-Whitney U (FTL vs Carting delay_ratio):")
    print(f"  U-statistic: {u_stat:.0f}, p-value: {p_val:.4f}")
    print(f"  Median FTL: {ftl.median():.3f}x, Median Carting: {cart.median():.3f}x")
    if p_val < 0.05:
        print(f"  → SIGNIFICANT. Model MUST be stratified by route_type.")
    else:
        print(f"  → Not significant. Can pool route types if needed.")

hourly_corr = df_clean.groupby("hour_of_day")["delay_ratio"].mean()
lag1_corr = hourly_corr.autocorr(lag=1)
print(f"\n  Hourly delay ratio lag-1 autocorrelation: {lag1_corr:.3f}")
print(f"  → {'Strong' if abs(lag1_corr) > 0.5 else 'Moderate'} temporal persistence in delays.")

skew = df_clean["delay_ratio"].skew()
kurt = df_clean["delay_ratio"].kurtosis()
print(f"\n  delay_ratio skewness: {skew:.3f}, kurtosis: {kurt:.3f}")
print(f"  → {'Log-transform recommended' if skew > 1 else 'Distribution is roughly symmetric'}.")
print(f"  → {'Heavy tails present — robust loss functions recommended.' if kurt > 3 else 'Normal-ish tails.'}")


print("\n\n" + "=" * 70)
print("PART 1 COMPLETE.")
print("Outputs:")
print("  outputs/part1_eda_dashboard.png")
print("  outputs/delivery_data_clean.csv")
print("=" * 70)
