"""
review_180day_run.py

One-off go/no-go review of the 180-day training-data run.
Reads ml/data/quality_report.json + ml/data/history.csv, runs the checks
for the training-data run, prints PASS / WARN / FAIL per check,
and ends with a single recommendation block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ml" / "analysis"))

from sensor_coverage_check import (  # noqa: E402
    GRID_N,
    describe_empty_cells,
    grid_cell_counts,
)

HISTORY_CSV    = PROJECT_ROOT / "ml" / "data" / "history.csv"
QUALITY_REPORT = PROJECT_ROOT / "ml" / "data" / "quality_report.json"

SEVENDAY_DROPPED = [12969, 53365, 90785, 123409, 128645, 280474, 280940]

results: list[tuple[str, str, str]] = []  # (check_id, status, message)
flagged_sensors: list[str] = []


def record(check_id: str, status: str, msg: str) -> None:
    results.append((check_id, status, msg))
    print(f"  [{status:4}] {check_id}: {msg}")


def header(title: str) -> None:
    print("\n" + "-" * 72)
    print(f" {title}")
    print("-" * 72)


def main() -> None:
    print("=" * 72)
    print(" 180-DAY TRAINING DATA REVIEW")
    print("=" * 72)

    qr = json.loads(QUALITY_REPORT.read_text())
    df = pd.read_csv(HISTORY_CSV)

    final_rows  = qr["final_row_count"]
    dropped_ids = [int(s) for s in qr.get("sensors_dropped_ab_failure_ids", [])]
    discovered  = qr["sensors_discovered"]
    survivors   = discovered - qr["sensors_dropped_ab_failure"] - qr["sensors_dropped_no_data"]

    print(f"\nfinal_row_count    : {final_rows:,}")
    print(f"sensors_discovered : {discovered}")
    print(f"sensors_with_data  : {qr['sensors_with_data']}")
    print(f"sensors_dropped    : {qr['sensors_dropped_ab_failure']}  → {dropped_ids}")
    print(f"surviving sensors  : {survivors}")
    print(f"history.csv rows   : {len(df):,}  (cols: {len(df.columns)})")

    # -------------------------------------------------------------- 1
    header("SANITY-BOUND CHECKS")
    if final_rows < 30_000:
        record("1. row count", "FAIL",
               f"{final_rows:,} rows < 30,000 floor (no documented outage explanation)")
    elif final_rows <= 80_000:
        record("1. row count", "PASS",
               f"{final_rows:,} rows in target band 30k–80k")
    else:
        expected_max = qr["sensors_with_data"] * 24 * 180
        record("1. row count", "WARN",
               f"{final_rows:,} > 80k; theoretical ceiling sensors*24*180 = {expected_max:,}")

    # -------------------------------------------------------------- 2
    n_dropped = qr["sensors_dropped_ab_failure"]
    if n_dropped <= 12:
        record("2. sensors dropped at A/B", "PASS",
               f"{n_dropped} dropped (≤12 of {discovered})")
    else:
        record("2. sensors dropped at A/B", "WARN",
               f"{n_dropped} dropped (>12 of {discovered}) — revisit thresholds per handoff doc")

    # -------------------------------------------------------------- 3
    set_180 = set(dropped_ids)
    set_7   = set(SEVENDAY_DROPPED)
    entered = sorted(set_180 - set_7)
    left    = sorted(set_7 - set_180)
    consistent = sorted(set_180 & set_7)
    print(f"\n  7-day dropped list   : {sorted(set_7)}")
    print(f"  180-day dropped list : {sorted(set_180)}")
    print(f"  consistent (durable) : {consistent}")
    print(f"  entered (new at 180d): {entered}")
    print(f"  left   (gone at 180d): {left}")
    if entered or left:
        record("3. 7d↔180d drop comparison", "WARN",
               f"{len(entered)} new, {len(left)} dropped from list — threshold-sensitive")
    else:
        record("3. 7d↔180d drop comparison", "PASS",
               "drop list is identical across windows")

    # -------------------------------------------------------------- 4
    header("BORDERLINE SURVIVOR CHECK")
    borderline = qr.get("ab_failure_borderline", [])
    survivors_above_35 = [
        b for b in borderline
        if b["outcome"] == "survived" and b["failure_rate"] > 0.35
    ]
    if survivors_above_35:
        print()
        for b in survivors_above_35:
            print(f"  sensor {b['sensor_id']:>7}  "
                  f"failure_rate={b['failure_rate']:.2%}  "
                  f"rows={b['rows_total']:,}")
        # Per-sensor decisions:
        for b in survivors_above_35:
            sid = b["sensor_id"]
            rate = b["failure_rate"]
            if sid != 87721:
                flagged_sensors.append(
                    f"sensor {sid}: survived A/B at {rate:.1%} failure — tightening candidate"
                )
        record("4. survivors >35%", "WARN",
               f"{len(survivors_above_35)} survivor(s) above 35% — see flags")
    else:
        record("4. survivors >35%", "PASS",
               "no surviving sensor has >35% A/B failure rate")

    # 87721-specific clause
    rec_87721 = next((b for b in borderline if b["sensor_id"] == 87721), None)
    if rec_87721 is None:
        record("4b. 87721 verdict", "PASS",
               "87721 not in borderline list (clean)")
    else:
        rate = rec_87721["failure_rate"]
        outcome = rec_87721["outcome"]
        if outcome == "dropped":
            record("4b. 87721 verdict", "PASS",
                   f"87721 already dropped at A/B (failure_rate={rate:.1%})")
        elif rate > 0.40:
            flagged_sensors.append(
                f"sensor 87721: survived at {rate:.1%} — consider dropping wholesale"
            )
            record("4b. 87721 verdict", "FAIL",
                   f"87721 survived at {rate:.1%} > 40% — drop wholesale")
        elif rate < 0.30:
            record("4b. 87721 verdict", "PASS",
                   f"87721 at {rate:.1%} < 30% — 7-day reading was noise")
        else:
            record("4b. 87721 verdict", "WARN",
                   f"87721 in 30–40% grey zone ({rate:.1%}) — borderline")

    # -------------------------------------------------------------- 5
    header("DATA-QUALITY FLAG THRESHOLDS")
    pct_uncorr = 100 * qr["rows_uncorrected_humidity_missing"] / final_rows
    if pct_uncorr <= 1:
        record("5. uncorrected humidity %", "PASS",
               f"{pct_uncorr:.2f}% of rows lack humidity (≤1%)")
    elif pct_uncorr <= 3:
        record("5. uncorrected humidity %", "WARN",
               f"{pct_uncorr:.2f}% of rows lack humidity (1–3%) — document & keep")
    else:
        record("5. uncorrected humidity %", "FAIL",
               f"{pct_uncorr:.2f}% of rows lack humidity (>3%) — drop uncorrected rows")

    # -------------------------------------------------------------- 6
    pct_climate = 100 * qr["wind_hours_climate_fallback"] / final_rows
    if pct_climate <= 1:
        record("6. wind climate-fallback %", "PASS",
               f"{pct_climate:.2f}% of rows used climate fallback (≤1%)")
    elif pct_climate <= 3:
        record("6. wind climate-fallback %", "WARN",
               f"{pct_climate:.2f}% of rows used climate fallback (1–3%)")
    else:
        record("6. wind climate-fallback %", "FAIL",
               f"{pct_climate:.2f}% of rows used climate fallback (>3%)")

    # -------------------------------------------------------------- 7
    pct_oor = 100 * qr["rows_dropped_out_of_range"] / final_rows
    if pct_oor <= 2:
        record("7. out-of-range drop %", "PASS",
               f"{pct_oor:.2f}% of rows dropped as out-of-range (≤2%)")
    elif pct_oor <= 5:
        record("7. out-of-range drop %", "WARN",
               f"{pct_oor:.2f}% of rows dropped as out-of-range (2–5%)")
    else:
        record("7. out-of-range drop %", "FAIL",
               f"{pct_oor:.2f}% of rows dropped as out-of-range (>5%)")

    # -------------------------------------------------------------- 8
    header("DISTRIBUTION CHECKS (history.csv)")
    pm = df["pm25"]
    mean = pm.mean()
    mx   = pm.max()
    mn   = pm.min()
    pct_zero = 100 * (pm == 0).sum() / len(pm)
    print(f"\n  PM2.5 mean : {mean:.2f} µg/m³")
    print(f"  PM2.5 min  : {mn:.2f}")
    print(f"  PM2.5 max  : {mx:.2f}")
    print(f"  rows == 0  : {(pm == 0).sum():,}  ({pct_zero:.2f}%)")
    pm_ok = (5 <= mean <= 25) and (mx < 250) and (mn >= 0) and (pct_zero < 5)
    if pm_ok:
        record("8. PM2.5 distribution", "PASS",
               f"mean={mean:.1f}, min={mn:.1f}, max={mx:.1f}, zeros={pct_zero:.2f}%")
    else:
        violations = []
        if not (5 <= mean <= 25):
            violations.append(f"mean {mean:.1f} outside 5–25")
        if mx >= 250:
            violations.append(f"max {mx:.1f} ≥ 250")
        if mn < 0:
            violations.append(f"min {mn:.1f} < 0")
        if pct_zero >= 5:
            violations.append(f"zeros {pct_zero:.1f}% ≥ 5%")
        status = "FAIL" if (mn < 0 or mx >= 250 or mean < 5 or mean > 25) else "WARN"
        record("8. PM2.5 distribution", status, "; ".join(violations))

    # -------------------------------------------------------------- 9
    nan_pcts = (df.isna().mean() * 100).sort_values(ascending=False)
    worst_col, worst_pct = nan_pcts.index[0], nan_pcts.iloc[0]
    if worst_pct < 95:
        record("9. column NaN coverage", "PASS",
               f"worst column '{worst_col}' has {worst_pct:.2f}% NaN (<95%)")
    else:
        record("9. column NaN coverage", "FAIL",
               f"column '{worst_col}' is {worst_pct:.1f}% NaN")

    # ------------------------------------------------------------- 10
    rows_per_sensor = df.groupby("sensor_id").size()
    median_rows = rows_per_sensor.median()
    threshold = 0.5 * median_rows
    weak = rows_per_sensor[rows_per_sensor < threshold].sort_values()
    print(f"\n  median rows/sensor : {median_rows:,.0f}")
    print(f"  cutoff (50% median): {threshold:,.0f}")
    print(f"  rows/sensor min    : {rows_per_sensor.min():,}")
    print(f"  rows/sensor max    : {rows_per_sensor.max():,}")
    if weak.empty:
        record("10. per-sensor row count", "PASS",
               f"all {len(rows_per_sensor)} sensors have ≥50% of median ({median_rows:,.0f})")
    else:
        print("\n  Sensors below 50% of median:")
        for sid, n in weak.items():
            pct_med = 100 * n / median_rows
            print(f"    sensor {sid:>7}  {n:>5,} rows  ({pct_med:.1f}% of median)")
            flagged_sensors.append(
                f"sensor {sid}: only {n} rows ({pct_med:.0f}% of median) — significant downtime"
            )
        record("10. per-sensor row count", "WARN",
               f"{len(weak)} sensor(s) below 50% of median row count")

    # ------------------------------------------------------------- 11
    if qr["rows_dropped_out_of_range"] == 0:
        record("11. out-of-range concentration", "PASS",
               "0 out-of-range drops — no concentration to investigate")
    else:
        record("11. out-of-range concentration", "WARN",
               f"{qr['rows_dropped_out_of_range']} out-of-range drops; "
               "per-sensor breakdown not available in quality_report.json")

    # ------------------------------------------------------------- 12
    header("SPATIAL COVERAGE (180-DAY SURVIVORS)")
    sensors = (
        df[["sensor_id", "lat", "lon"]]
          .drop_duplicates(subset=["sensor_id"])
          .sort_values("sensor_id")
          .reset_index(drop=True)
    )
    counts, empty = grid_cell_counts(sensors)
    print(f"\n  surviving sensors    : {len(sensors)}  (7-day reference: 18)")
    print(f"  empty grid cells     : {len(empty)} of {GRID_N*GRID_N}  "
          f"({describe_empty_cells(empty) if empty else 'none'})")

    # Hypothetical: what if all 7-day-only drops (set_7 - set_180) were still here?
    # And: what cells does the 180-day-only-newly-dropped (set_180 - set_7) sit in
    # vs what would happen if those new drops had survived?
    # We don't have lat/lon for dropped sensors offline; rely on history.csv.
    # Instead compare 7-day surviving count vs 180-day surviving count.
    sevenday_survivor_count = 18
    delta = len(sensors) - sevenday_survivor_count
    if delta == 0:
        record("12. spatial coverage", "PASS",
               f"{len(sensors)} survivors, same count as 7-day; "
               f"{len(empty)} empty cell(s)")
    elif delta > 0 and len(empty) <= 4:
        record("12. spatial coverage", "PASS",
               f"{len(sensors)} survivors (+{delta} vs 7-day), {len(empty)} empty cells")
    elif len(empty) >= 5:
        record("12. spatial coverage", "WARN",
               f"{len(sensors)} survivors with {len(empty)} empty cells — "
               "geographic gaps")
    else:
        record("12. spatial coverage", "WARN",
               f"{len(sensors)} survivors (delta={delta:+d} vs 7-day's 18); "
               f"{len(empty)} empty cells")

    # ------------------------------------------------------------- 13
    header("HIGHWAY FEATURE SANITY")
    dh = df["dist_to_highway_m"]
    dh_min = dh.min()
    dh_max = dh.max()
    dh_nulls = dh.isna().sum()
    print(f"\n  dist_to_highway_m min : {dh_min:.1f} m")
    print(f"  dist_to_highway_m max : {dh_max:.1f} m")
    print(f"  nulls                 : {dh_nulls}")
    if dh_nulls > 0:
        record("13. dist_to_highway_m", "FAIL",
               f"{dh_nulls} null(s)")
    elif dh_min >= 50:
        record("13. dist_to_highway_m", "WARN",
               f"min {dh_min:.0f} m ≥ 50 — no sensor sits near a highway shoulder")
    elif dh_max >= 15_000:
        record("13. dist_to_highway_m", "FAIL",
               f"max {dh_max:.0f} m ≥ 15 km — implausibly far for DFW")
    else:
        record("13. dist_to_highway_m", "PASS",
               f"min={dh_min:.0f} m, max={dh_max:.0f} m, no nulls")

    # ----------------------------------------------------- final verdict
    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    fails = [r for r in results if r[1] == "FAIL"]
    warns = [r for r in results if r[1] == "WARN"]
    passes = [r for r in results if r[1] == "PASS"]
    print(f"\nPASS: {len(passes)}   WARN: {len(warns)}   FAIL: {len(fails)}")

    if flagged_sensors:
        print("\nFlagged sensors / items:")
        for s in flagged_sensors:
            print(f"  - {s}")

    print("\n" + "=" * 72)
    print(" RECOMMENDATION")
    print("=" * 72)

    high_severity_flag = any(
        ("consider dropping wholesale" in s) or ("drop wholesale" in s)
        for s in flagged_sensors
    )

    if not fails and len(warns) <= 2 and not high_severity_flag:
        print("\nGO — Dataset is ready for training.")
        if warns:
            print("Minor warnings noted above; none are blockers.")
        print()
    elif fails:
        print("\nNO-GO — Dataset has hard failures.")
        print("Failures:")
        for r in fails:
            print(f"  - {r[0]}: {r[2]}")
        print("\nRecommend deeper investigation or recollection before proceeding.")
        print()
    else:
        print("\nGO WITH CAVEATS — Dataset is usable but has open items.")
        print("\nWarnings:")
        for r in warns:
            print(f"  - {r[0]}: {r[2]}")
        if flagged_sensors:
            print("\nSpecific actions to consider before training:")
            for s in flagged_sensors:
                print(f"  - {s}")
        print()


if __name__ == "__main__":
    main()
