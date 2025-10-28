NH Preprocessing — Gold Feature Table (One Row per CCN)
======================================================

Objective
---------
Turn the uploaded nursing-home (NH) CSVs into a single, model-ready **Gold** table with exactly one row per CCN.
The pipeline cleans text/categoricals, handles missing values, builds derived features (citations, penalties,
survey recency, QM z-scores), and frequency-encodes high-cardinality categories.


Inputs (expected file names)
---------------------------
• NH_ProviderInfo_Sep2025.csv — roster (IDs, location, ratings, beds/residents, ownership, etc.).
• NH_HealthCitations_Sep2025.csv — deficiency events with scope/severity codes (A–L).
• NH_SurveySummary_Sep2025.csv — survey/complaint/revisit counts.
• NH_SurveyDates_Sep2025.csv — last known survey dates.
• NH_Penalties_Sep2025.csv — fines and payment-denial signals.
• FY_2025_SNF_VBP_Facility_Performance.csv — VBP incentive multiplier per facility.
• NH_QualityMsr_Claims_Sep2025.csv — claims-based QMs (measure_id, value, period).
• NH_QualityMsr_MDS_Sep2025.csv — MDS-based QMs (measure_id, value, period).
• NH_StateUSAverages_Sep2025.csv — optional: state/US benchmarks per measure (means/stds).

If a file is missing, the pipeline skips that domain gracefully.


How to Run
----------
Example CLI:

    python preprocess_nh.py \
      --data-dir "/mnt/data" \
      --out-dir "/mnt/data/outputs" \
      --months-window 36 \
      --min-count-rare 50 \
      --measure-direction-config "/mnt/data/measure_direction.yaml"   # optional

Key flags:
• --months-window: Look-back window in months for penalties/citations (default 36).
• --min-count-rare: Rare-category cutoff for high-card columns (default 50).
• --measure-direction-config: YAML with measures that are lower-is-better (see “Configuration” below).


What We Keep & How It’s Used
----------------------------

Keys & IDs
• ccn — strict, 6-digit string. Unique in roster. Used to join domains. Not a model feature.

Core Numerics
• number_of_certified_beds — facility scale; used directly; also normalizer for citations.
• average_number_of_residents_per_day — size/case-mix proxy; used directly.
→ Missingness indicators added (e.g., number_of_certified_beds_was_missing). Imputed by state median when possible.

Ratings (if present in roster)
• overall_rating, staffing_rating, rn_staffing_rating, qm_rating — cast to numeric and used as features.
→ Missingness indicators added and median-imputed (by state if available).

Categoricals (Low-Cardinality → One-Hot in training)
• state — standardized 2-letter code.
• ownership_type — canonicalized (“For-Profit”, “Non-Profit”, “Government”).
• in_hospital, sprinkler_status, resident_and_family_councils, chain_owner — normalized text, filled with "Unknown".

Categoricals (High-Cardinality → Frequency Encoding)
• county, zip — frequency-encoded into county_freq and zip_freq to avoid an explosion of sparse columns.
  Raw string columns are retained for traceability but the *_freq columns are the modeling features.

Derived Features
• severity_rate_per_100_beds — (sum of severity weights A–L over last N months) / beds × 100.
• penalty_events_36mo, total_fines_usd_36mo, had_any_payment_denial — penalty rollups over the last N months.
• last_health_survey_date, days_since_last_health_survey — survey recency signals derived from survey dates.
• vbp_incentive_multiplier — value-based purchasing multiplier (joined if present).
• qm_domain_z_state, qm_domain_z_us — mean z-scores across quality measures (claims + MDS), taking the latest
  period per measure per facility. If a measure is “lower is better,” its sign is flipped first so that
  higher is consistently better before z-scoring.

Missing Data Handling
• Numeric: add *_was_missing flags, then impute with medians (by state when available, else global).
• Categorical: fill missing with "Unknown".
• Dates: do not impute; use recency features (e.g., days_since_last_health_survey).

Encoding Strategy
• Low-card categoricals are normalized and ready for standard One-Hot Encoding in the model pipeline.
• High-card categoricals use frequency encoding columns (col_freq).


Outputs
-------
Written to the provided --out-dir:

Primary
• NH_Gold_Feature_Table.csv — model-ready features; one row per CCN.
• NH_Gold_Feature_Manifest.csv — list of columns in the Gold table.
• NH_Preprocessing_Report.txt — quick counts and high-level summary.

Cleaned/Derived Domain Tables (for traceability)
• clean_provider_info.csv
• clean_penalties_36mo.csv
• clean_survey_dates.csv
• clean_survey_summary.csv
• clean_health_citations_36mo.csv
• clean_vbp.csv
• clean_qm_long.csv
• clean_qm_facility_zscores.csv


End-to-End Steps (What the Script Does)
--------------------------------------
1) Read CSVs and normalize headers to snake_case.
2) Derive strict ccn as a 6-digit string; preserve leading zeros.
3) Provider roster: enforce one row per CCN; clean text; coerce numerics; standardize state codes.
4) Penalties: parse dates, filter to last N months, roll up counts/sums/flags per CCN.
5) Health citations: map A–L to numeric weights; window filter; sum weights per CCN; normalize per 100 beds.
6) Surveys: parse dates; compute last_health_survey_date and days_since_last_health_survey.
7) Quality measures: combine claims+MDS; pick latest period per measure; optional direction flip;
   compute state and US z-scores, then average across measures to facility-level domain scores.
8) VBP: merge vbp_incentive_multiplier if available.
9) Assemble Gold table by joining all domains on CCN.
10) Clean/normalize categoricals; bucket rare levels for high-card columns.
11) Add missingness flags; impute numerics (state median when available; otherwise global median); fill cats with "Unknown".
12) Frequency-encode county and zip (adds county_freq and zip_freq).
13) Save Gold table, manifest, report, and cleaned domain tables.


Configuration (Optional)
-----------------------
Create a YAML to define measures where lower is better. These will be sign-flipped before standardization so
“higher is better” is consistent across all measures.

Example: /mnt/data/measure_direction.yaml

    lower_is_better:
      - HOSP_READM
      - ED_VISITS
      - PRESSURE_ULCER
      - FALLS_WITH_MAJOR_INJURY


Data Quality & Validation Rules
-------------------------------
• CCN must be unique per facility row in the roster and exactly 6 digits (string).
• number_of_certified_beds ≥ 0; average_number_of_residents_per_day ≥ 0; residents ≤ beds when both present.
• Rating scales should remain within valid ranges (e.g., 1–5).
• Rates defined as proportions should lie in [0,1]; clip only if data dictionary confirms bounds.
• State must be a valid 2-letter code; otherwise set to NaN and rely on imputation logic.
• Dates must not be in the future; recency features (days_since_*) should be non-negative.


Reproducibility Notes
---------------------
• All joins are keyed by ccn only; no fuzzy joins across names/addresses.
• Missingness flags are created prior to imputation for transparency and model utility.
• Any column name variation is handled by token search; final outputs are documented in the manifest.


Next Steps for Modeling
-----------------------
• Keep ccn as an identifier only; exclude from model features.
• One-Hot Encode low-card categoricals; use *_freq features for high-card categories.
• Consider adding state-centered numeric features (value minus state median) for within-state comparisons.
• Validate with grouped CV by state or region to check generalization across geographies.
