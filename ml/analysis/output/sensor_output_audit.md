# Raw Sensor Output Audit — Dallas Live Network

**Date:** 2026-05-12
**Scope:** Live PurpleAir + OpenAQ readings inside the project BBOX, raw vs. EPA-corrected.
**Question that prompted this:** Are the Cedar Hill / Floral Farms / other Dallas sensors returning plausible values right now?

---

## TL;DR

Two PurpleAir sensors in south Dallas are currently returning saturated, physically impossible values (~5000 µg/m³ raw, ~2600 µg/m³ EPA-corrected). The rest of the network — 24 PurpleAir sensors plus 8 OpenAQ reference stations — is healthy and self-consistent (median ≈ 7 µg/m³). The two bad sensors will dominate any IDW grid cell they touch, so they should be quarantined before the heatmap reads them.

There is **no PurpleAir sensor named "Cedar Hill"** inside the BBOX. The PurpleAir sensor named "FloralFarms1" exists and is one of the two malfunctioning ones. The closest OpenAQ reference monitors are TCEQ stations with different names ("Dallas Bexar Street", "Town Creek", etc.).

---

## What was tested

A one-shot inline test invoked the production ingestion paths:

- `data.ingestion.purpleair.fetch_sensors()` — live PurpleAir, EPA-corrected at source.
- `data.ingestion.openaq.fetch_openaq()` — live OpenAQ v3 reference-grade monitors.

For each row, the test printed `pm25_raw` (uncorrected CF=1) and `pm25` (post-Barkjohn 2021 EPA correction) so the correction itself could be visually sanity-checked.

---

## Findings

### 1. Two saturated PurpleAir sensors

| sensor_id | name         | lat       | lon        | pm25_raw | pm25 (corrected) | humidity |
|-----------|--------------|-----------|------------|---------:|-----------------:|---------:|
| 123409    | FloralFarms1 | 32.686947 | -96.737870 |   4991.0 |          2597.80 |       38 |
| 123453    | Cedar Crest  | 32.734040 | -96.792435 |   4920.7 |          2561.24 |       38 |

Both readings are 200×–500× the rest of the network. They sit ~6 km apart in south Dallas. Identical raw magnitudes (~5000) with identical humidity (38%) is the classic A/B-channel saturation signature — almost certainly hardware fault, not a real plume.

The EPA correction is doing its job (it scaled 4991 → 2598, applying the 0.52× factor + RH term as expected). The problem is upstream of the correction.

### 2. The rest of the PurpleAir network looks fine

Excluding the two bad sensors (n=24):

- raw range: 0.0 – 20.7 µg/m³
- corrected range: 1.8 – 13.3 µg/m³
- median corrected: ~7.4 µg/m³

The correction is moving values in the expected direction (corrected < raw in most cases, slightly above raw near zero where the +5.71 intercept dominates).

### 3. OpenAQ reference network (8 stations)

| sensor_id     | name                                            | pm25 |
|---------------|-------------------------------------------------|-----:|
| oaq-1867568   | Town Creek                                      | 12.2 |
| oaq-2027502   | Creekview                                       |  9.7 |
| oaq-2687422   | Plano (Central Plano)                           | 13.4 |
| oaq-2823991   | Royal Ln and Luna Rd                            | 13.6 |
| oaq-3305184   | North Irving TX                                 | 16.7 |
| oaq-4806780   | Dallas Bexar Street                             | 13.2 |
| oaq-6154898   | Intersection of Campbell rd and Willow wood ln  |  6.6 |
| oaq-6280493   | Dallas, TX                                      | 27.0 |

All values plausible. "Dallas, TX" at 27.0 is the highest and worth watching, but still well inside "moderate" AQI and consistent with normal urban background.

### 4. No sensors named "Cedar Hill" or "Floral Farms" on the OpenAQ side

Searched the OpenAQ result set for `cedar`, `floral`, `hinton`, `arlington`, `kaufman`, `frisco`, `denton`, `grapevine`. Zero matches. The TCEQ stations the user may have been thinking of are listed under different names (see table above).

On the PurpleAir side, "FloralFarms1" (sensor 123409) does exist — and is one of the two broken ones. "Cedar Crest" exists but is a south Dallas neighborhood, not Cedar Hill (which is a separate city ~15 km south).

---

## Recommendation

Add an upper-bound sanity filter in `data/ingestion/purpleair.py` alongside the existing `pm25 < 0` drop at line 107. A reasonable cap is 500 µg/m³ (already past EPA "hazardous" at 250.4), which would have caught both today's malfunctioning sensors without any risk of dropping a real-world Dallas reading. Currently nothing prevents these saturated values from feeding the IDW interpolation and producing a fake hot zone over south Dallas on the heatmap.

Implementation note: do the filter on `pm25_raw` (not `pm25`), because the EPA correction can pull a saturated 5000 down to 2600, but the upstream signal is still the fault indicator.
