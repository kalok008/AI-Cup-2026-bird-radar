# Bird Radar Classification — AI Cup 2026

**Team: Trained on Rakia** · 🥉 **3rd / 89 in the Performance Track** · **7th overall**

Solution for the [AI Cup 2026](https://www.teamepoch.ai/ai-cup/) performance track: classifying bird groups from radar track data to support bird-collision mitigation around wind farms.

---

## The competition

AI Cup is a nationwide Dutch AI competition organised by [Team Epoch](https://www.teamepoch.ai/) (TU Delft's AI DreamTeam) and enabled by [AIC4NL](https://aic4nl.nl/), run over five weeks (13 Feb – 24 Mar 2026) with a €7,000 prize pool.

The 2026 challenge, provided by **TNO**: millions of birds migrate through the same airspace where wind energy is expanding. Teams built AI systems to identify bird species from radar data, so operators can curtail turbines intelligently — reducing collisions without needlessly sacrificing energy production.

Unlike most ML competitions, AI Cup scores two tracks:

| Track | Weight | Evaluated on |
|---|---|---|
| **Performance** | 60% | Kaggle leaderboard, quantitative only |
| **Implementation** | 40% | Written proposal: system design, responsible AI, data collection, mitigation strategy |

The overall nomination score is a rank-normalised blend of the two.

### Our results

| Leaderboard | Rank |
|---|---|
| **Performance (60%)** | **3rd** of 89 validated teams |
| Implementation (40%) | 26th |
| **Overall** | **7th**|

Top 5 overall were nominated to present at the Dutch AI Congress. We finished 3rd on pure model performance — ahead of four of the five finalists — but our implementation proposal ranked 26th, which pulled the weighted overall score down to 7th, one tie short of the nomination cutoff.

---

## The task

Multiclass classification of radar tracks into **9 bird groups**: Clutter, Cormorants, Pigeons, Ducks, Geese, Gulls, Birds of Prey, Waders, Songbirds.

**Metric:** macro-averaged Average Precision (macro-mAP) over per-class probabilities.

**Data:** 2,601 training tracks, 1,872 test tracks. Each track carries a 3D/4D WKB trajectory (lon, lat, altitude, RCS), a timestamp array, and radar-derived scalars (airspeed, min/max altitude, duration, radar bird size class).

The core difficulty is **extreme class imbalance**:

| Class | Share of train |
|---|---|
| Gulls | 57.8% |
| Songbirds | 18.6% |
| Pigeons | 4.7% |
| Waders | 4.6% |
| Birds of Prey | 4.2% |
| Clutter | 3.2% |
| Geese | 3.2% |
| Ducks | 2.2% |
| **Cormorants** | **1.5%** (40 samples) |

Macro-mAP weights all nine classes equally, so 40 Cormorant tracks matter as much as 1,503 Gull tracks.

---

## Our approach (`rakia-v21.ipynb`)

A CPU-only, feature-engineering-heavy gradient boosting ensemble with a Bayesian post-processing layer. No GPU required — in the spirit of the competition's "no GPU arms race" rule.

### 1. Feature engineering — 305 features per track

Trajectories are parsed from WKB and featurised into:

- **Kinematics** — per-segment ground speed, vertical speed, bearing, turn angle and turn rate; distributional stats over each.
- **Geometry** — total path length, net displacement, straightness ratio, convex-hull area, bounding-box extents.
- **Altitude profile** — min/max/mean/std of z, altitude-band occupancy fractions (e.g. `frac_z_lt_1000`), climb/descent asymmetry.
- **RCS statistics** — mean, std, and percentile spread of radar cross-section along the track.
- **Temporal** — hour of day, night flag, day-of-year, season interactions, migration-window encodings.
- **Spatial** — 0.2° lat/lon grid cell IDs, plus derived `night_id`, `cell_id`, `cell_night_id` context keys.

### 2. Hand-crafted separator features

Confusion analysis drove targeted features for the classes the model kept mixing up:

| Feature | Rationale |
|---|---|
| `songbird_score` | `max_z < 35` + `duration < 40` + `n_birds ≥ 2` — the strongest Gull→Songbird separator (Gull median duration 61s vs Songbird 28s) |
| `max_z_x_duration` | 4.3× separation between Gulls (3758) and Songbirds (873) |
| `bop_score` | `airspeed < 13` — Birds of Prey median 11.5 m/s vs Gulls 14.1 m/s |
| `duck_signature` | `airspeed > 18` AND `min_z < 5` — matches 60% of Ducks, 8% of everything else |
| `wader_flock_score` | `n_birds ≥ 10` (41% of Waders vs 12% of Gulls) combined with altitude band |

### 3. Fold-safe context aggregates

Group statistics over `night_id`, `cell_id` and `cell_night_id` (counts, mean/std airspeed and altitude, small-bird and flock ratios, RCS and straightness means), plus frequency encodings and rolling-night features. All aggregates are fit **only on the training half of each fold** and applied to validation and test — no leakage.

### 4. Ensemble — 12 models

| Slot | Model |
|---|---|
| 0–3 | CatBoost multiclass, full data, seeds 42/777 × variants A/B |
| 4–7 | CatBoost multiclass, Gull-undersampled (cap 200/fold), seeds 42/777 × A/B |
| 8–9 | LightGBM multiclass, seeds 42/777 |
| 10 | Binary CatBoost expert: Cormorants vs rest |
| 11 | Binary CatBoost expert: Waders vs rest |

- Loss: `MultiClassOneVsAll` with `auto_class_weights="Balanced"`.
- Minority classes oversampled to 150 samples with 1% Gaussian jitter on numeric features.
- Gull undersampling to 200/fold gives the balanced models room to learn Cormorant/Wader boundaries.
- CV: 5-fold `StratifiedGroupKFold` grouped on `primary_observation_id` — tracks from the same observation never straddle the split.

### 5. Bayesian post-processing

The main differentiator over a plain ensemble. Raw probabilities are reweighted by:

- **Size-conditional priors** — `P(class | radar_bird_size)` with Dirichlet smoothing.
- **Density likelihoods** — per-class Gaussian log-likelihood on airspeed, max altitude, mean RCS.
- **Migration boost** — night-time + burst signatures boost Songbirds/Waders, penalise Geese/Cormorants.
- **Adaptive context priors** — night / cell / cell-night priors with count-adaptive weighting.
- **Clutter heuristics** — airspeed > 28 m/s or altitude > 3000 m.

The four mixing weights (β_size, β_den, β_mig, β_ctx) are tuned with **Optuna over 200 trials** against OOF macro-mAP. Selected: `(0.231, 0.086, 0.063, 0.014)`.

### 6. Final calibration

- Binary experts blended into the Cormorant and Wader columns with per-class α tuned independently on that class's AP (both landed at 0.02 — the multiclass models were already strong).
- Logit-bias tuning on the three weakest classes, constrained so log-loss cannot degrade beyond tolerance.

---

## Results

**Final OOF macro-mAP: 0.6882** (log-loss 0.6025)

Pipeline contribution:

| Stage | OOF mAP |
|---|---|
| Raw ensemble mean | 0.6756 |
| + Bayesian post-processing (default β) | 0.6861 |
| + Optuna-tuned β | 0.6877 |
| + Binary expert blend | 0.6886 |
| + Logit bias (final) | 0.6882 |

Per-class Average Precision:

| Class | AP |
|---|---|
| Gulls | 0.9624 |
| Clutter | 0.8900 |
| Pigeons | 0.8687 |
| Songbirds | 0.8255 |
| Birds of Prey | 0.6502 |
| Ducks | 0.6303 |
| Geese | 0.6038 |
| Waders | 0.4477 |
| Cormorants | 0.3231 |

Per-fold raw mAP: 0.706 / 0.724 / 0.648 / 0.714 / — (mean **0.699 ± 0.027**).

### What we never fully solved

The dominant confusions, by OOF count:

| Confusion | Cases |
|---|---|
| Gulls → Songbirds | 93 |
| Songbirds → Gulls | 65 |
| Waders → Gulls | 45 |
| Gulls → Birds of Prey | 26 |
| Birds of Prey → Gulls | 24 |

**Cormorants (AP 0.32)** were the hard ceiling — 40 training samples, and 15 of them get absorbed into the Gull class. Neither oversampling, Gull undersampling, nor a dedicated binary expert moved it past 0.33. **Waders (0.45)** lose 45 cases to Gulls; they occupy a similar altitude band and only separate on flock size.

There is also a persistent **~0.13 OOF → leaderboard gap**, which points to distribution shift between the train and test splits rather than overfitting to CV.

---

## Implementation track

**`final_ai_cup_submission_v2.pdf`** — *Radar-first bird-group intelligence for precision wind-farm mitigation.*

The proposal turns the classifier into an operational system. Its core argument is **confidence-aware deployment**: the model informs decisions but is never the final authority, precisely because its per-class reliability is uneven.

- **Architecture** — four layers: radar ingestion → feature service (the same 305 features) → inference with Bayesian context calibration → a decision layer that aggregates track-level outputs into rolling risk states. Radar-first by design; cameras, acoustics and human observers are used only for labelling and validation, never as operational dependencies.
- **Decision logic** — a green/amber/red state machine over 10–30 minute windows, combining migration density, rotor-swept-zone overlap, bird-group probability mix and model confidence. Red recommends temporary curtailment at the turbine-cluster level rather than site-wide. Operator overrides are logged with reasons, creating an audit trail for later threshold tuning.
- **Data strategy** — targeted relabelling. A weekly review queue is built from low-confidence predictions, known confusion pairs and any amber/red event, so expert annotation effort goes where it fixes the actual weaknesses (Cormorants, Waders) instead of being spread uniformly.
- **Rollout** — three phases: shadow mode (silent logging), advisory mode (operators see states and uncertainty, retain control), then constrained automation only for narrow, validated scenarios.

The proposal is grounded in the radar aeroecology and wind-farm curtailment literature (Shamoun-Baranes et al. 2019; Cohen et al. 2022; Bradarić et al. 2024; McClure et al. 2022, on eagle-fatality reduction via automated curtailment).

It placed **26th of 89**. Our read on why: it's a careful, defensible design, but it stays close to what the model already justifies rather than proposing something ambitious about the wider system — and it leans on qualitative reasoning where the higher-ranked proposals brought concrete numbers on cost, false-alarm rates and energy lost to curtailment.

---

## Repository

```
rakia-v21.ipynb                    Full pipeline: EDA → features → CV → Bayes → submission
submission.csv                     Test-set probabilities (track_id + 9 class columns)
final_ai_cup_submission_v2.pdf     Implementation-track proposal
```

The notebook runs top to bottom on Kaggle CPU and emits diagnostic plots at every stage (class distribution, feature KDEs by group, sample trajectories, per-fold mAP, β search landscape, bias curves, test-vs-prior distribution).

**Dependencies:** `numpy`, `pandas`, `scikit-learn`, `catboost`, `lightgbm`, `optuna`, `shapely`, `matplotlib`, `seaborn`, `tqdm`.

---

## What we'd do differently

- **Sequence models.** We only ever fed aggregate statistics to the trees. An LSTM or small Transformer over the raw trajectory points is the obvious unexplored direction, especially for the Gull/Songbird split where the shape of the flight path — not its summary — is the discriminative signal.
- **Test-time prior matching / TTA** to attack the OOF→LB gap.
- **Invest earlier in the implementation track.** It's 40% of the score. Our proposal was sound but written late and largely as a narrative wrapper around the model we'd already built; a stronger version would have quantified the operational trade-off — false-alarm rate, MWh lost to curtailment, collisions avoided per curtailment hour — and let those numbers shape the model's objective rather than the other way round. Finishing 3rd on performance and 26th on implementation is precisely the imbalance AI Cup is designed to expose.

---

## Acknowledgements

Challenge and data: **TNO**. Organisation: **Team Epoch** (TU Delft) and **AIC4NL**.

