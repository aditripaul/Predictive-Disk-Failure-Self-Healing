# Feature engineering: what the model sees, and why

Status as of 2026-10-09; the feature set itself has not changed since
2026-10-06. This describes the feature set in the code. The
attributes and features marked **new** were added that day and have since
been run on real data (Q1 + Q2 2026): **they made no measurable difference**.
The models with and without them are within 0.01 drive-level AUPRC of each
other on test, and removing the lifetime counters changed nothing either
(`docs/model_status_and_runbook.md` section 4.2). `active_defect_total` takes
about 40% of the model's split gain, but it restates what the rolling
counters already carried. The features stay in the pipeline because they cost
little and do no harm. Other results quoted in this document come from the
earlier feature set (five SMART attributes, Q1 2026, 14-day horizon); the
primary horizon is now 30 days.

**Update, 2026-10-09:** a sixth priority attribute (`reallocation_event_count`,
SMART 196) was added, and `feature_confidence`'s attribute-coverage term
changed from a single fleet-wide denominator to one scored per drive model -
see section 5 and
`docs/adr/0002-attribute-selection-and-vendor-aware-confidence.md`. Neither
change is reflected in the rest of this document's body text below, which
still describes the five-attribute, fleet-wide-denominator feature set as
"current"; read section 5 for what is actually true now.

Related: `docs/model_status_and_runbook.md` (results, goal status) and
`docs/pipeline_usage.md` (how to run).

## 1. The problem the features have to solve

Failures are rare. In the current two-quarter build, 621 of 353,949 drives
fail in the three-week test window - about 1 in 570. (Earlier numbers in this
document come from the one-quarter build, where it was about 300 of 345,000.)
At the goal's operating point the model may raise only a handful of false
alarms across the whole fleet, so a feature is useful only if it separates a
failing drive from the many healthy drives that also show errors.

Two facts from the measurements shape every choice below:

- **Most false alarms are healthy drives whose error counters stay
  elevated.** Requiring a score to persist for several days made precision
  worse, so the level of a counter is not enough. What separates the cases is
  whether a counter is *moving*.
- **A second model type and per-family models did not help.** XGBoost and
  one-model-per-drive-family both scored at or below the pooled LightGBM. The
  limit is in the inputs, not the learner.

## 2. Which SMART attributes are used

Backblaze publishes a raw and a normalized value for each SMART attribute. The
pipeline uses raw values only.

### 2.1 Core error counters (full feature family)

| SMART | Name in the pipeline | What it counts |
|---|---|---|
| 5 | `reallocated_sector_count` | Bad sectors found and remapped |
| 187 | `reported_uncorrectable_errors` | Errors hardware error-correction could not recover |
| 188 | `command_timeout` | Operations aborted on timeout |
| 197 | `current_pending_sector_count` | Bad sectors waiting to be remapped |
| 198 | `offline_uncorrectable` | Uncorrectable errors found in offline scans |
| 196 | `reallocation_event_count` | How many times a sector has been remapped - a count of *events*, not of currently-bad sectors |

**Why the first five:** Backblaze (2014) and Google (Pinheiro et al., 2007) both
found them the most consistent failure indicators across manufacturers, and
the paper we reviewed (Amram et al., 2021) found 5 and 187 the most important
in every model it built. In our model the top features by gain are rolling
statistics of 5 and 197.

**Why 196, added 2026-10-09:** a raw-attribute screen
(`scripts/screen_raw_attributes.py`) measured single-attribute drive-day AUC
for every SMART column Backblaze publishes, not just the ones the paper
named. 196 separates failing from healthy drive-days better than four of the
other five (|AUC - 0.5| = 0.306) and is reported at roughly double the
coverage of 187/188 (~66% against ~32%), reaching the non-Seagate drives
where those two are null. Full screen results, including the hypotheses it
refuted (a vault/pod spatial-clustering feature; SMART 22/helium), are in
`docs/adr/0002-attribute-selection-and-vendor-aware-confidence.md`.

### 2.2 Context attributes (light feature family), new

| SMART | Name | Why it is included |
|---|---|---|
| 7 | `seek_error_rate` | Head positioning errors; the paper's third most important attribute for long-term health |
| 9 | `power_on_hours` | The drive's real age (see section 4.3) |
| 194 | `temperature_celsius` | Operating temperature; a sudden change can accompany mechanical trouble |
| 199 | `udma_crc_error_count` | Interface/cable errors, which can explain command timeouts that are not the drive's fault |

These are ingested when a file has them. Some drive models do not report all
of them, so they are optional at ingest and excluded from the confidence
score (section 5).

### 2.3 Why two tiers

Each core counter produces about 34 feature columns. Giving the four context
attributes the same treatment would widen the table by roughly another 140
columns on top of what six core counters already cost, and the training
matrices would no longer fit in memory with two quarters of data. The context
attributes also do not behave like error counters: rolling statistics of
power-on hours carry almost no information.

So context attributes get the current value and its change over 7 and 30
observations, three columns each.

### 2.4 Not used, and why

- **Normalized values.** The paper kept both raw and normalized values and
  found some normalized ones useful (3 and 187). We have not ingested them;
  it is a candidate for a later step.
- **SMART 3 (spin-up time) and 190 (temperature difference).** Important in
  the paper, not ingested yet.
- **Lifetime counters 4, 12, 192, 193, 240-242.** The paper removed them
  because they mostly encode age. We ingest only power-on hours and test it
  with and without (section 6).

## 3. Feature families for the core counters

All windows are per drive. One drive's history never enters another drive's
features.

| Family | Columns per counter | What it captures |
|---|---|---|
| Current value | 1 | The level today |
| Rolling mean, median, min, max, std, range over 7, 14, 30 days | 18 | The recent level and how variable it is |
| Delta and slope over 7, 14, 30 observations | 6 | How much the counter moved, and how fast |
| Acceleration (7-day slope minus 30-day slope) | 1 | Whether growth is speeding up |
| Positive-day count over 7, 14, 30 days | 3 | How many recent days had any errors |
| Spike count over 7, 14, 30 days (5 and 197 only) | 3 | Days with a jump above a threshold |
| Zero-to-nonzero flag | 1 | The day a clean counter first shows an error |
| Days since last increase, new | 1 | How recently the counter moved (section 4.2) |
| Model-family z-score of the 7, 14, 30-day mean | 3 | The level relative to drives of the same family |

**Why rolling statistics:** a single day's reading is noisy, and a level that
has persisted for weeks is different from one that appeared yesterday.

**Why deltas, slopes and acceleration:** a drive with 50 reallocated sectors
that has been stable for a year is usually fine. One that went from 0 to 5 in
three days is not. Levels cannot tell these apart; rates can.

**Why z-scores within a drive family:** some models normally report higher
counts than others. The z-score asks whether a drive is unusual for its own
kind. `current_pending_sector_count_7d_model_zscore` has been the single
highest-gain feature in several runs.

## 4. Cross-attribute and drive-level features

### 4.1 Active defect velocity, new

`active_defect_total` is the sum of the four defect counters (5, 187, 197,
198) for a drive-day. `active_defect_velocity_{3,7,14,30}d` is its change per
observation over each window, and `active_defect_acceleration_7d_vs_30d`
compares the short and long rates.

**Why:** one counter can move for a benign reason. Several defect types rising
together is a stronger signal than any single one, and a tree model has to
discover that combination from many separate columns unless it is given
directly. The 3-observation window is there for fast-developing failures: the
median warning lead time we measured is 6 to 9 days.

### 4.2 Days since last increase, new

For each core counter, the number of days since it last went up, or 999 if it
has not gone up in the loaded data. A counter that was already above zero on
the drive's first loaded day counts as "never increased", because the increase
happened before the data starts.

**Why:** the false alarms are mostly drives with old, stable damage. Google
reported that after a drive's first scan error it is 39 times more likely to
fail within 60 days, which is a statement about recency, not level. This
feature states recency directly.

### 4.3 Age: `drive_age_days` and `power_on_days` (new)

`drive_age_days` counts days since the drive first appears in the loaded data.
For every drive already in service when the data starts, that is the same
number, so it measures the data window more than the drive.
`power_on_days` (`power_on_hours / 24`) is the drive's own lifetime counter.

Both are kept. The paper removed lifetime counters to avoid a model that just
learns "old drives fail", and an age feature can also leak the calendar in a
time-based split. Whether age helps here is an open question, answered by the
ablation in section 6, not assumed.

### 4.4 Ratios

| Feature | Meaning |
|---|---|
| `pending_to_reallocated_ratio` | Pending sectors relative to already-remapped ones: damage that is new rather than handled |
| `reallocated_per_capacity` | Remapped sectors relative to drive size |
| `uncorrectable_per_power_on_hour` | Uncorrectable errors relative to age. Defined before, but only active now that power-on hours is ingested |

### 4.5 Temperature spike, new

The highest temperature over a drive's last 3 observations minus its mean over
the last 30. **Why a difference and not the level:** Google found absolute
temperature a poor predictor, and drives in different chassis run at different
steady temperatures. A departure from the drive's own baseline is the part
that could mean something.

## 5. Telemetry quality and feature confidence

`telemetry_coverage_30d`, `days_since_last_telemetry`, the gap and stale flags,
and `feature_confidence` describe how much the features can be trusted.
`feature_confidence` = coverage x recency x attribute coverage, and the agent
will not take a destructive action below a minimum confidence (0.80,
`configs/agent.yaml`).

Attribute coverage is computed over the **core counters only**. If the context
attributes counted, a drive model that does not report temperature would have
lower confidence for a reason unrelated to its health, and its destructive
actions would be silently downgraded.

**Found broken fleet-wide, fixed 2026-10-09 (ADR 0002).** Two multiplicative
bugs each capped `feature_confidence` below the 0.80 floor, measured at **0%
of 62.6 million drive-days** clearing it before either fix:

1. The attribute-coverage denominator counted every core attribute for every
   drive, including the Seagate-only 187/188, so any non-Seagate drive
   capped at 3/5 = 0.60 regardless of how complete ITS OWN telemetry was.
   Fixed: `src/features/cross_vendor.py::model_expected_attributes` computes,
   per drive model, which attributes that model actually reports (present in
   at least half its drive-days, given at least 1,000 drive-days of
   evidence); `add_feature_confidence` then counts a drive down only for an
   attribute its own model is expected to report. A model with too little
   history to judge, or absent from the table, is scored against every
   attribute - the conservative direction, since it can only lower
   confidence, never raise it.
2. Recency charged the ordinary daily telemetry cadence itself as 24 hours of
   staleness, capping every drive-day at `exp(-24/48) = 0.6065` no matter how
   fresh its last reading was. Fixed: `expected_cadence_days`
   (`configs/features.yaml`, 1 for both Backblaze and SMART-Z) sets the
   cadence telemetry is expected to arrive at, and recency now charges only
   the time beyond it - a reading one day after the last is fresh and scores
   1.0; a three-day gap is charged two days' worth and scores `exp(-48/48) =
   0.37`.

Measured on the rebuilt gold table: **86.1%** of drive-days now clear the
floor, with 0.00% newly blocked (a per-model denominator is never stricter
than the global one it replaced). `build_gold_features` now fails the build
outright if NO drive-day can ever clear the floor, since no synthetic test
fixture would have caught either bug - every unit and chaos test builds
drives with every attribute already present.

## 6. How the choices are tested

Nothing new is kept on argument alone. On one data build:

1. Train with all features, and with each group removed: the new features, the
   age features, the context attributes.
2. Compare drive-level precision at 5, 10, 20 and 35% recall on the test split,
   with thresholds chosen on validation.
3. Keep a group only if removing it makes the result worse.

Earlier ablations on the old feature set: removing the identity features
(drive age, capacity) lowered AUPRC slightly, and restricting to windowed
features lowered it more. Differences below about 0.04 AUPRC between data
builds are not meaningful.

## 7. Known weaknesses

- **Windows count observations, not calendar days,** for deltas, velocity and
  the temperature spike. A drive with a telemetry gap gets a longer real window.
  Rolling statistics do use calendar days.
- **The first weeks of data have short history.** A 30-day window on day 5 sees
  5 days. Deltas over a window longer than the available history are null and
  become 0 in the model input.
- **Family z-scores use the whole data period,** including the validation and
  test weeks, for the family mean and standard deviation. This is a small leak
  of later information into training rows.
- **Raw values are vendor-specific.** Seagate's raw seek error rate and read
  error rate are composite numbers, not simple counts. Power-on hours is
  encoded differently by some vendors.
- **The label is partly defined by a feature.** Backblaze marks a drive failed
  when it is removed, and one removal reason is SMART 187 turning positive.
- **The vendor-aware denominator is a property of the fleet's CURRENT
  model-day distribution, computed fleet-wide the same way the model-family
  z-score mean/std are** - so it can shift slightly between builds even
  though no attribute's own ingestion changed, which is a small, acknowledged
  source of drift between a model card and a later evaluation of the same
  gold table.
- **The raw `failure` flag is excluded from the feature list**
  (`src/models/features.py::NON_FEATURE_COLUMNS`). It marks the failure day
  itself, so it is a label in disguise. It was all zeros in the assembled
  frame, so dropping it removed no signal - but it did change the model's
  column sampling, and so the fitted trees
  (`docs/model_status_and_runbook.md` section 1.1).

## 8. Where each piece lives

| Piece | Code |
|---|---|
| Which columns are ingested | `src/ingest/backblaze.py` (`REQUIRED_COLUMNS`, `OPTIONAL_SMART_COLUMNS`) |
| SMART id to name | `src/preprocess/smart_mapping.py` |
| Core and context lists, windows, spike thresholds | `configs/features.yaml` |
| Order of feature steps | `pipelines/build_gold_features.py::build_gold_features` |
| Rolling statistics | `src/features/windows.py` |
| Deltas, slopes, acceleration, light deltas | `src/features/derivatives.py` |
| Positive counts, spikes, zero-to-nonzero | `src/features/events.py` |
| Defect velocity, temperature spike, days since last increase | `src/features/velocity.py` |
| Age | `src/features/lifecycle.py` |
| Ratios and family z-scores | `src/features/cross_vendor.py` |
| Confidence | `src/features/confidence.py` (recency/coverage math), `src/features/cross_vendor.py::model_expected_attributes` (which attributes a model is expected to report) |
| The written list of every feature | `data/audit/feature_registry/v<version>.json` |

## 9. References

- Amram, Dunn, Toledano, Zhuo (2021). *Interpretable predictive maintenance for
  hard drives.* Machine Learning with Applications 5.
- Backblaze (2014). *Hard drive SMART stats.*
- Pinheiro, Weber, Barroso (2007). *Failure trends in a large disk drive
  population.* USENIX FAST.
