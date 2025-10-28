#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess NH datasets into a single model-ready feature table (one row per CCN).
- Strict CCN handling (6-digit string). Unique per facility in roster.
- Text/categorical cleaning.
- Missing-value indicators and imputation.
- Derived features: penalty rollups, citation severity rate, recency of surveys.
- Quality measures: latest period per measure, optional direction flip, state/US standardization.
- Encodings: one-hot (low-card handled downstream) + frequency encoding (high-card).
- Outputs cleaned domain tables + Gold feature table.

USAGE (examples):
  python preprocess_nh.py --data-dir "/mnt/data" --out-dir "/mnt/data/outputs"
  python preprocess_nh.py --data-dir "/mnt/data" --out-dir "/mnt/data/outputs" --months-window 36 \
    --measure-direction-config "/mnt/data/measure_direction.yaml"

Notes:
  - Robust to column-name variation via fuzzy token matching.
  - If NH_StateUSAverages_Sep2025.csv lacks std columns, std is computed from your QM data.
  - Optional YAML (measure_direction.yaml) sets which QMs are lower-is-better (flip sign).
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import yaml
except Exception:
    yaml = None

# ------------------- Constants -------------------
NULL_TOKENS = {"", " ", "na", "n/a", "null", "none", "unknown", "nan", ".", "-", "--"}

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD",
    "ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
    "TN","TX","UT","VA","VT","WA","WI","WV","WY","PR","GU","VI","AS","MP"
}

# CMS scope/severity A–L → weights
SEVERITY_WEIGHTS = {
    "A":1, "B":1, "C":1,
    "D":2, "E":2, "F":2,
    "G":4, "H":4, "I":4,
    "J":8, "K":8, "L":8
}

LOW_CARD_DEFAULT = [
    "state","ownership_type","in_hospital","sprinkler_status",
    "resident_and_family_councils","chain_owner"
]
HIGH_CARD_DEFAULT = ["county","zip"]
TEXT_DEFAULT = ["provider_name","legal_business_name","parent_organization_name","address","city"]

# ------------------- Helpers -------------------
def snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.lower().strip("_")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]
    return df

def clean_text_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).map(lambda x: unicodedata.normalize("NFKD", x))
    s = s.str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"[“”]", '"', regex=True).str.replace(r"[‘’]", "'", regex=True)
    s = s.replace(NULL_TOKENS, np.nan)
    return s

def normalize_cats(s: pd.Series, mapping: dict = None, upper=False, title=False) -> pd.Series:
    out = clean_text_series(s)
    if upper:
        out = out.str.upper()
    elif title:
        out = out.str.title()
    if mapping:
        out = out.map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
    return out

def bucket_rare_levels(s: pd.Series, min_count: int = 50, other_label="Other") -> pd.Series:
    counts = s.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), other_label)

def fix_state_codes(s: pd.Series) -> pd.Series:
    s = normalize_cats(s, upper=True)
    s = s.where(s.isin(US_STATES), np.nan)
    return s

def derive_ccn_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Derive strict 6-digit CCN, preserving leading zeros."""
    CCN_CANDIDATES = [
        "cms_certification_number_(ccn)","cms_certification_number","provider_number",
        "facility_ccn","cms_ccn","ccn"
    ]
    df = df.copy()
    src_col = None
    for c in CCN_CANDIDATES:
        if c in df.columns:
            src_col = c
            break
    if src_col is None:
        for c in df.columns:
            if "ccn" in c:
                src_col = c
                break
    if src_col is not None:
        s = df[src_col].astype(str)
        digits = s.str.extract(r"(\d+)", expand=False)
        norm = digits.fillna("")
        norm = norm.apply(lambda x: x[-6:] if len(x) >= 6 else x.zfill(6) if len(x) > 0 else "")
        df["ccn"] = norm.replace("", pd.NA)
    else:
        if "ccn" not in df.columns:
            df["ccn"] = pd.NA
    return df

def parse_any_date(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(pd.Series([pd.NA]*len(s)), errors="coerce")

def read_csv_loose(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, low_memory=False)

# ------------------- Domain loaders -------------------
def load_provider_info(path: Path) -> pd.DataFrame:
    df = read_csv_loose(path)
    df = normalize_columns(df)
    df = derive_ccn_strict(df)

    # numeric coercions
    numeric_like = [
        "number_of_certified_beds","average_number_of_residents_per_day",
        "overall_rating","staffing_rating","rn_staffing_rating","qm_rating"
    ]
    for col in df.columns:
        if any(x in col for x in numeric_like):
            try:
                df[col] = pd.to_numeric(df[col].str.replace(r"[,$%]", "", regex=True), errors="coerce")
            except Exception:
                pass

    # state & text
    if "state" in df.columns:
        df["state"] = fix_state_codes(df["state"])
    for c in ["provider_name","legal_business_name","parent_organization_name","address","city",
              "ownership_type","chain_owner","in_hospital","sprinkler_status","resident_and_family_councils",
              "county","zip"]:
        if c in df.columns:
            df[c] = clean_text_series(df[c])

    # enforce one row per CCN in roster
    df = df.drop_duplicates(subset=["ccn"], keep="first")
    return df

def load_penalties(path: Path, months_window: int = 36) -> pd.DataFrame:
    df = read_csv_loose(path)
    df = normalize_columns(df)
    df = derive_ccn_strict(df)

    # date column
    date_cols = [c for c in df.columns if any(t in c for t in ["date","effective","from","start","issued"])]
    dcol = date_cols[0] if date_cols else None
    if dcol:
        df[dcol] = parse_any_date(df[dcol])

    # amount column
    amt_cols = [c for c in df.columns if any(t in c for t in ["fine","amount","penalty","cmp"])]
    ac = amt_cols[0] if amt_cols else None
    if ac:
        df[ac] = pd.to_numeric(df[ac].str.replace(r"[,$%]", "", regex=True), errors="coerce")

    # payment denial flag
    pd_cols = [c for c in df.columns if "denial" in c or "payment_denial" in c]
    pdf = pd_cols[0] if pd_cols else None
    if pdf:
        df[pdf] = df[pdf].astype(str).str.upper().isin(["Y","YES","TRUE","1"]).astype("int8")

    # window filter
    if dcol:
        cutoff = pd.Timestamp("today").normalize() - pd.Timedelta(days=30*months_window)
        df = df[df[dcol].isna() | (df[dcol] >= cutoff)]

    gp = df.groupby("ccn")
    out = pd.DataFrame({
        "ccn": gp.size().index,
        "penalty_events_36mo": gp.size().values
    })
    out["total_fines_usd_36mo"] = gp[ac].sum().values if ac else 0.0
    out["had_any_payment_denial"] = (gp[pdf].max()>0).astype("int8").values if pdf else 0
    return out

def load_survey_dates(path: Path) -> pd.DataFrame:
    df = read_csv_loose(path)
    df = normalize_columns(df)
    df = derive_ccn_strict(df)

    # candidate date columns
    candidates = [c for c in df.columns if ("health" in c and "survey" in c and "date" in c) or c.endswith("health_survey_date")]
    if not candidates:
        candidates = [c for c in df.columns if ("health" in c and "date" in c)]

    # cast to datetime
    for col in df.columns:
        if any(tok in col for tok in ["date","survey","start","end"]):
            dt = parse_any_date(df[col])
            if dt.notna().any():
                df[col] = dt

    out = df[["ccn"] + candidates].copy() if candidates else df[["ccn"]].copy()
    if candidates:
        out["last_health_survey_date"] = out[candidates].max(axis=1)
    else:
        out["last_health_survey_date"] = pd.NaT

    today = pd.Timestamp("today").normalize()
    out["days_since_last_health_survey"] = (today - out["last_health_survey_date"]).dt.days
    return out[["ccn","last_health_survey_date","days_since_last_health_survey"]]

def load_survey_summary(path: Path) -> pd.DataFrame:
    df = read_csv_loose(path)
    df = normalize_columns(df)
    df = derive_ccn_strict(df)
    for col in df.columns:
        if any(tok in col for tok in ["count","complaint","revisit","deficien","tag"]):
            df[col] = pd.to_numeric(df[col].str.replace(r"[,$%]", "", regex=True), errors="coerce")
    out = df.groupby("ccn", as_index=False).sum(numeric_only=True)
    return out

def load_health_citations(path: Path, months_window: int = 36) -> pd.DataFrame:
    df = read_csv_loose(path)
    df = normalize_columns(df)
    df = derive_ccn_strict(df)

    # severity code column
    sev_col = None
    for c in df.columns:
        if "scope" in c and "severity" in c:
            sev_col = c; break
    if sev_col is None:
        for c in df.columns:
            if re.fullmatch(r"[a-z_]*severity[a-z_]*", c or ""):
                sev_col = c; break

    # date column
    date_cols = [c for c in df.columns if any(tok in c for tok in ["survey","date","visit"])]
    dcol = date_cols[0] if date_cols else None
    if dcol:
        df[dcol] = parse_any_date(df[dcol])
        cutoff = pd.Timestamp("today").normalize() - pd.Timedelta(days=30*months_window)
        df = df[df[dcol].isna() | (df[dcol] >= cutoff)]

    # map severity A–L to weights
    if sev_col and sev_col in df.columns:
        code = df[sev_col].astype(str).str.upper().str.extract(r"([A-L])", expand=False)
        df["severity_weight"] = code.map(SEVERITY_WEIGHTS).fillna(0).astype(float)
    else:
        df["severity_weight"] = 0.0

    out = df.groupby("ccn", as_index=False)["severity_weight"].sum()
    out = out.rename(columns={"severity_weight":"severity_weight_sum_36mo"})
    return out

def load_vbp(path: Path) -> pd.DataFrame:
    df = read_csv_loose(path)
    df = normalize_columns(df)
    df = derive_ccn_strict(df)
    cand = [c for c in df.columns if "multiplier" in c or "incentive" in c]
    if cand:
        col = cand[0]
        df[col] = pd.to_numeric(df[col].str.replace(r"[,$%]", "", regex=True), errors="coerce")
        return df[["ccn", col]].rename(columns={col:"vbp_incentive_multiplier"}).drop_duplicates("ccn")
    return pd.DataFrame({"ccn":[], "vbp_incentive_multiplier":[]})

def load_state_us_benchmarks(path: Path) -> pd.DataFrame:
    df = read_csv_loose(path)
    return normalize_columns(df)

def load_qm_long(path: Path, source_label: str) -> pd.DataFrame:
    """Expect columns like: ccn, measure_id/name, value/rate/score, period_start, period_end (flexible)."""
    df = read_csv_loose(path)
    df = normalize_columns(df)
    df = derive_ccn_strict(df)

    # measure id
    mid = "measure_id" if "measure_id" in df.columns else None
    if mid is None:
        for c in df.columns:
            if "measure" in c and "id" in c:
                mid = c; break

    # value
    val = None
    for cand in ["value","rate","score","pct","percent","measure_value"]:
        if cand in df.columns:
            val = cand; break

    # period start / end
    ps = None; pe = None
    for c in df.columns:
        if ("period" in c or "start" in c) and "date" in c:
            ps = c; break
    for c in df.columns:
        if ("period" in c or "end" in c) and "date" in c:
            pe = c; break
    if ps is None:
        for c in df.columns:
            if c.endswith("start_date") or c.endswith("period_start"):
                ps = c; break
    if pe is None:
        for c in df.columns:
            if c.endswith("end_date") or c.endswith("period_end"):
                pe = c; break

    if ps: df[ps] = parse_any_date(df[ps])
    if pe: df[pe] = parse_any_date(df[pe])
    if val: df[val] = pd.to_numeric(df[val].str.replace(r"[,$%]", "", regex=True), errors="coerce")

    keep = ["ccn"]
    if mid: keep.append(mid)
    if val: keep.append(val)
    if ps: keep.append(ps)
    if pe: keep.append(pe)
    out = df[keep].copy()
    out["source"] = source_label

    rename_map = {}
    if mid: rename_map[mid] = "measure_id"
    if val: rename_map[val] = "value"
    if ps: rename_map[ps] = "period_start"
    if pe: rename_map[pe] = "period_end"
    out = out.rename(columns=rename_map)
    return out

# ------------------- Feature builders -------------------
def compute_citation_rate_per_bed(citations_sum: pd.DataFrame, provider_info: pd.DataFrame) -> pd.DataFrame:
    df = citations_sum.merge(provider_info[["ccn","number_of_certified_beds"]], on="ccn", how="left")
    df["number_of_certified_beds"] = pd.to_numeric(df["number_of_certified_beds"], errors="coerce")
    df["severity_rate_per_100_beds"] = df["severity_weight_sum_36mo"] / df["number_of_certified_beds"].replace(0, np.nan) * 100.0
    return df[["ccn","severity_rate_per_100_beds"]]

def latest_qm_per_facility(qm_long: pd.DataFrame) -> pd.DataFrame:
    if "period_end" in qm_long.columns:
        qm_long = qm_long.sort_values(["ccn","measure_id","period_end"])
        latest = qm_long.groupby(["ccn","measure_id"], as_index=False).tail(1)
    else:
        latest = qm_long.drop_duplicates(["ccn","measure_id"], keep="last")
    return latest

def load_measure_direction(config_path: Path = None):
    lower_is_better = set()
    if config_path and config_path.exists() and yaml is not None:
        try:
            cfg = yaml.safe_load(config_path.read_text())
            lower_is_better = set(cfg.get("lower_is_better", []))
        except Exception:
            pass
    return lower_is_better

def standardize_qm_by_state_us(qm_latest: pd.DataFrame, provider_info: pd.DataFrame, bench: pd.DataFrame = None):
    df = qm_latest.merge(provider_info[["ccn","state"]], on="ccn", how="left")

    if bench is not None and all(c in bench.columns for c in ["state","measure_id"]):
        # expected optional cols: state_mean,state_std,us_mean,us_std
        df = df.merge(bench, on=["state","measure_id"], how="left", suffixes=("","_bench"))
        state_mean = pd.to_numeric(df.get("state_mean"), errors="coerce")
        state_std  = pd.to_numeric(df.get("state_std"), errors="coerce")
        df["qm_z_state"] = (df["value"] - state_mean) / state_std.replace(0,np.nan)
        if "us_mean" in df.columns and "us_std" in df.columns:
            us_mean = pd.to_numeric(df["us_mean"], errors="coerce")
            us_std  = pd.to_numeric(df["us_std"], errors="coerce")
            df["qm_z_us"] = (df["value"] - us_mean) / us_std.replace(0,np.nan)
    else:
        # compute state stats from data
        g = df.groupby(["state","measure_id"])["value"]
        stats = g.agg(["mean","std"]).reset_index().rename(columns={"mean":"state_mean","std":"state_std"})
        df = df.merge(stats, on=["state","measure_id"], how="left")
        df["qm_z_state"] = (df["value"] - df["state_mean"]) / df["state_std"].replace(0,np.nan)
        # compute US stats
        gus = df.groupby(["measure_id"])["value"].agg(["mean","std"]).reset_index().rename(columns={"mean":"us_mean","std":"us_std"})
        df = df.merge(gus, on=["measure_id"], how="left")
        df["qm_z_us"] = (df["value"] - df["us_mean"]) / df["us_std"].replace(0,np.nan)

    fac = df.groupby("ccn", as_index=False).agg(qm_domain_z_state=("qm_z_state","mean"),
                                                qm_domain_z_us=("qm_z_us","mean"))
    return fac

def frequency_encode(df: pd.DataFrame, col: str) -> pd.Series:
    freqs = df[col].value_counts()
    return df[col].map(freqs).fillna(0)

# ------------------- Pipeline -------------------
def run_pipeline(data_dir: Path, out_dir: Path, months_window: int = 36, min_count_rare=50, measure_dir_config: Path = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs (adjust names if your files differ)
    p_provider     = data_dir / "NH_ProviderInfo_Sep2025.csv"
    p_penalties    = data_dir / "NH_Penalties_Sep2025.csv"
    p_survey_dates = data_dir / "NH_SurveyDates_Sep2025.csv"
    p_survey_sum   = data_dir / "NH_SurveySummary_Sep2025.csv"
    p_health_cit   = data_dir / "NH_HealthCitations_Sep2025.csv"
    p_vbp          = data_dir / "FY_2025_SNF_VBP_Facility_Performance.csv"
    p_bench        = data_dir / "NH_StateUSAverages_Sep2025.csv"
    p_qm_claims    = data_dir / "NH_QualityMsr_Claims_Sep2025.csv"
    p_qm_mds       = data_dir / "NH_QualityMsr_MDS_Sep2025.csv"

    provider  = load_provider_info(p_provider) if p_provider.exists() else pd.DataFrame()
    penalties = load_penalties(p_penalties, months_window) if p_penalties.exists() else pd.DataFrame(columns=["ccn","penalty_events_36mo","total_fines_usd_36mo","had_any_payment_denial"])
    surv_dates= load_survey_dates(p_survey_dates) if p_survey_dates.exists() else pd.DataFrame(columns=["ccn","last_health_survey_date","days_since_last_health_survey"])
    surv_sum  = load_survey_summary(p_survey_sum) if p_survey_sum.exists() else pd.DataFrame(columns=["ccn"])
    citations = load_health_citations(p_health_cit, months_window) if p_health_cit.exists() else pd.DataFrame(columns=["ccn","severity_weight_sum_36mo"])
    vbp       = load_vbp(p_vbp) if p_vbp.exists() else pd.DataFrame(columns=["ccn","vbp_incentive_multiplier"])
    bench     = load_state_us_benchmarks(p_bench) if p_bench.exists() else None

    sev_rate = compute_citation_rate_per_bed(citations, provider) if not citations.empty and not provider.empty else pd.DataFrame(columns=["ccn","severity_rate_per_100_beds"])

    # QMs
    qm_claims = load_qm_long(p_qm_claims, "claims") if p_qm_claims.exists() else pd.DataFrame(columns=["ccn","measure_id","value","period_start","period_end","source"])
    qm_mds    = load_qm_long(p_qm_mds, "mds") if p_qm_mds.exists() else pd.DataFrame(columns=["ccn","measure_id","value","period_start","period_end","source"])
    qm_all    = pd.concat([qm_claims, qm_mds], ignore_index=True) if not (qm_claims.empty and qm_mds.empty) else pd.DataFrame(columns=["ccn","measure_id","value","period_start","period_end","source"])

    if not qm_all.empty:
        lower_is_better = load_measure_direction(measure_dir_config)
        if lower_is_better:
            mask = qm_all["measure_id"].isin(lower_is_better)
            qm_all.loc[mask, "value"] = -qm_all.loc[mask, "value"]  # flip so "higher is better"
        qm_latest = latest_qm_per_facility(qm_all)
        qm_fac    = standardize_qm_by_state_us(qm_latest, provider, bench)
    else:
        qm_fac = pd.DataFrame(columns=["ccn","qm_domain_z_state","qm_domain_z_us"])

    # Assemble Gold
    base_cols = ["ccn","state","number_of_certified_beds","average_number_of_residents_per_day",
                 "overall_rating","staffing_rating","rn_staffing_rating","qm_rating"]
    base_cols = [c for c in base_cols if c in provider.columns] + ["ccn"]  # safe subset
    base_cols = list(dict.fromkeys(base_cols))  # unique, keep order

    gold = provider[base_cols].copy()
    for c in ["ownership_type","chain_owner","in_hospital","sprinkler_status",
              "resident_and_family_councils","city","county","zip"]:
        if c in provider.columns and c not in gold.columns:
            gold[c] = provider[c]

    for part in [penalties, surv_dates, sev_rate, vbp, qm_fac]:
        if not part.empty:
            gold = gold.merge(part, on="ccn", how="left")

    # Clean/normalize categoricals
    for c in TEXT_DEFAULT:
        if c in gold.columns:
            gold[c] = clean_text_series(gold[c])

    if "state" in gold.columns:
        gold["state"] = fix_state_codes(gold["state"])
    OWNERSHIP_MAP = {
        "Government - Federal":"Government",
        "Government - State":"Government",
        "Government - County":"Government",
        "For Profit":"For-Profit",
        "Non Profit":"Non-Profit"
    }
    if "ownership_type" in gold.columns:
        gold["ownership_type"] = normalize_cats(gold["ownership_type"], mapping=OWNERSHIP_MAP, title=True)
    for c in LOW_CARD_DEFAULT:
        if c in gold.columns and c not in ["ownership_type","state"]:
            gold[c] = normalize_cats(gold[c], title=True)
    for c in HIGH_CARD_DEFAULT:
        if c in gold.columns:
            gold[c] = normalize_cats(gold[c])
            gold[c] = bucket_rare_levels(gold[c], min_count=min_count_rare, other_label="Other")

    # Missingness + impute numerics (by-state if available)
    numeric_cols = [c for c in gold.columns if any(tok in c for tok in
                     ["number_of_certified_beds","average_number_of_residents_per_day",
                      "total_fines_usd","penalty_events","vbp_incentive_multiplier",
                      "qm_domain_z","severity_rate","days_since_last_health_survey"])]

    for c in numeric_cols:
        gold[f"{c}_was_missing"] = gold[c].isna().astype("int8")

    if "state" in gold.columns:
        for c in numeric_cols:
            gold[c] = gold.groupby("state")[c].transform(lambda s: s.fillna(s.median()))
            gold[c] = gold[c].fillna(gold[c].median())
    else:
        for c in numeric_cols:
            gold[c] = gold[c].fillna(gold[c].median())

    # Categorical missing
    cat_cols = [c for c in gold.columns if gold[c].dtype == "object" and c not in ["ccn"]]
    for c in cat_cols:
        gold[c] = gold[c].fillna("Unknown")

    # Frequency-encode high-card
    for c in HIGH_CARD_DEFAULT:
        if c in gold.columns:
            gold[c + "_freq"] = frequency_encode(gold, c)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    if not provider.empty:  provider.to_csv(out_dir / "clean_provider_info.csv", index=False)
    if not penalties.empty: penalties.to_csv(out_dir / "clean_penalties_36mo.csv", index=False)
    if not surv_dates.empty:surv_dates.to_csv(out_dir / "clean_survey_dates.csv", index=False)
    if not surv_sum.empty:  surv_sum.to_csv(out_dir / "clean_survey_summary.csv", index=False)
    if not citations.empty: citations.to_csv(out_dir / "clean_health_citations_36mo.csv", index=False)
    if not vbp.empty:       vbp.to_csv(out_dir / "clean_vbp.csv", index=False)
    if not qm_all.empty:    qm_all.to_csv(out_dir / "clean_qm_long.csv", index=False)
    if not qm_fac.empty:    qm_fac.to_csv(out_dir / "clean_qm_facility_zscores.csv", index=False)

    gold = gold.drop_duplicates(subset=["ccn"], keep="first")
    gold.to_csv(out_dir / "NH_Gold_Feature_Table.csv", index=False)

    manifest = pd.DataFrame({"feature": gold.columns})
    manifest.to_csv(out_dir / "NH_Gold_Feature_Manifest.csv", index=False)

    # Simple report
    report_lines = []
    report_lines.append("NH Preprocessing Report")
    report_lines.append(f"Facilities in provider roster: {len(provider)}")
    report_lines.append(f"Gold rows (unique CCN): {gold['ccn'].nunique()}")
    report_lines.append("Key derived: severity_rate_per_100_beds, penalties_36mo, days_since_last_health_survey, qm_domain_z_state/us, vbp_incentive_multiplier")
    (out_dir / "NH_Preprocessing_Report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    return gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Directory with input CSVs")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory to write outputs")
    ap.add_argument("--months-window", type=int, default=36, help="Window in months for penalties/citations")
    ap.add_argument("--min-count-rare", type=int, default=50, help="Rare-category cutoff for HIGH_CARD columns")
    ap.add_argument("--measure-direction-config", type=str, default="", help="YAML with list lower_is_better: [measure_id,...]")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    cfg = Path(args.measure_direction_config) if args.measure_direction_config else None

    run_pipeline(data_dir, out_dir,
                 months_window=args.months_window,
                 min_count_rare=args.min_count_rare,
                 measure_dir_config=cfg)

#----------handling outliers-------------
#def function to remove outliers 
def remove_outliers_iqr(df, cols, multiplier=1.5):
    df = df.copy()
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# identify numeric and categorical columns
num_cols = data.select_dtypes(include=[np.number]).columns
cat_cols = data.select_dtypes(exclude=[np.number]).columns
# handle missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
data[num_cols] = num_imputer.fit_transform(data[num_cols])
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
# remove outliers 
data = remove_outliers_iqr(data, num_cols)
# encode categorical variables
data = pd.get_dummies(data, drop_first=True)
# scale numeric features
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])


if __name__ == "__main__":
    main()
