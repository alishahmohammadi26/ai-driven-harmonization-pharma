"""
Bridging the Data Divide: AI-Driven Harmonization in Pharma
============================================================
Case study: Harmonizing multi-site clinical laboratory data to the public
LOINC standard using open Python libraries (pandas, difflib, numpy).

The raw dataset is SIMULATED to mirror a real pooling problem — three legacy
sources recording the same analytes under different local names, units, and
formats. No real patient data is used. The harmonization target (LOINC) is a
genuine public standard; all codes below are the real canonical codes.

Article:  https://alishahmohammadi22.github.io/blog/ai-driven-harmonization-pharma.html
Author:   Ali Shahmohammadi, Ph.D.
Date:     January 2026
"""
import string
from difflib import get_close_matches
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. PUBLIC REFERENCE STANDARD  (real LOINC codes -> canonical name + unit)
# ---------------------------------------------------------------------------
LOINC = {
    "ALT":        {"loinc": "1742-6", "name": "Alanine aminotransferase, S/P", "unit": "U/L"},
    "AST":        {"loinc": "1920-8", "name": "Aspartate aminotransferase, S/P", "unit": "U/L"},
    "GLUCOSE":    {"loinc": "2345-7", "name": "Glucose, S/P",                 "unit": "mg/dL"},
    "CREATININE": {"loinc": "2160-0", "name": "Creatinine, S/P",             "unit": "mg/dL"},
    "HBA1C":      {"loinc": "4548-4", "name": "Hemoglobin A1c/Hgb.total",    "unit": "%"},
    "CHOLESTEROL":{"loinc": "2093-3", "name": "Cholesterol, S/P",            "unit": "mg/dL"},
    "PLATELETS":  {"loinc": "777-3",  "name": "Platelets, Blood",            "unit": "10*3/uL"},
    "HEMOGLOBIN": {"loinc": "718-7",  "name": "Hemoglobin, Blood",           "unit": "g/dL"},
}

# Ontology-style synonym vocabulary: local term -> canonical concept.
SYNONYMS = {
    "ALT": ["ALT", "Alanine Aminotransferase", "SGPT", "ALT (SGPT)",
            "Alanine transaminase", "GPT", "ALAT"],
    "AST": ["AST", "Aspartate Aminotransferase", "SGOT", "AST (SGOT)",
            "Aspartate transaminase", "GOT", "ASAT"],
    "GLUCOSE": ["Glucose", "GLU", "Blood Glucose", "Fasting Glucose",
                "Glucose, serum", "Plasma Glucose", "FBG"],
    "CREATININE": ["Creatinine", "CREA", "Cr", "Serum Creatinine",
                   "Creat", "Creatinine (serum)"],
    "HBA1C": ["HbA1c", "A1C", "Hemoglobin A1c", "Glycated Hemoglobin",
              "HgbA1C", "Glycosylated Hemoglobin", "HbA1c (IFCC)"],
    "CHOLESTEROL": ["Cholesterol", "CHOL", "Total Cholesterol", "T. Chol",
                    "Cholesterol total", "Tot Chol"],
    "PLATELETS": ["Platelets", "PLT", "Platelet Count", "Plt Ct",
                  "Thrombocytes", "Platelet Ct"],
    "HEMOGLOBIN": ["Hemoglobin", "HGB", "Hgb", "Haemoglobin", "Hb", "Hemoglob"],
}

# Reverse lookup + normalized vocabulary for fuzzy matching
def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)

NORM_TO_CONCEPT = {}
VOCAB = []
for concept, terms in SYNONYMS.items():
    for t in terms:
        NORM_TO_CONCEPT[norm(t)] = concept
        VOCAB.append(norm(t))

# Reference physiological ranges (mean, sd) in canonical units, for value sim
DIST = {
    "ALT":         (30, 12), "AST": (28, 10), "GLUCOSE": (95, 18),
    "CREATININE":  (0.95, 0.25), "HBA1C": (5.6, 0.7), "CHOLESTEROL": (190, 35),
    "PLATELETS":   (255, 55), "HEMOGLOBIN": (13.8, 1.5),
}

# ---------------------------------------------------------------------------
# 2. SIMULATE THREE MESSY SOURCE SYSTEMS
# ---------------------------------------------------------------------------
# Each site uses a different naming style + unit system, mirroring real pooling.
SITES = {
    "Site A - Legacy EHR": {
        "n": 3200,
        "names": {c: SYNONYMS[c][0] for c in SYNONYMS},          # US short names
        "units": {"GLUCOSE": "mg/dL", "CREATININE": "mg/dL", "HEMOGLOBIN": "g/dL",
                  "CHOLESTEROL": "mg/dL", "HBA1C": "%", "PLATELETS": "10*3/uL",
                  "ALT": "U/L", "AST": "U/L"},
    },
    "Site B - CRO (SI units)": {
        "n": 3000,
        "names": {"ALT": "SGPT", "AST": "SGOT", "GLUCOSE": "GLU",
                  "CREATININE": "CREA", "HBA1C": "HbA1c (IFCC)",
                  "CHOLESTEROL": "CHOL", "PLATELETS": "PLT", "HEMOGLOBIN": "Hb"},
        "units": {"GLUCOSE": "mmol/L", "CREATININE": "umol/L", "HEMOGLOBIN": "g/L",
                  "CHOLESTEROL": "mmol/L", "HBA1C": "mmol/mol", "PLATELETS": "10*9/L",
                  "ALT": "IU/L", "AST": "IU/L"},
    },
    "Site C - Partner Lab": {
        "n": 2900,
        "names": {"ALT": "Alanine Aminotransferase", "AST": "Aspartate Aminotransferase",
                  "GLUCOSE": "Blood Glucose", "CREATININE": "Serum Creatinine",
                  "HBA1C": "Glycated Hemoglobin", "CHOLESTEROL": "Total Cholesterol",
                  "PLATELETS": "Platelet Count", "HEMOGLOBIN": "Haemoglobin"},
        "units": {"GLUCOSE": "mg/dL", "CREATININE": "mg/dL", "HEMOGLOBIN": "g/dL",
                  "CHOLESTEROL": "mg/dL", "HBA1C": "%", "PLATELETS": "10*3/uL",
                  "ALT": "U/L", "AST": "U/L"},
    },
}

# unit -> canonical converters (factor to canonical, or callable)
def to_canonical(concept, value, unit):
    u = unit.lower().replace(" ", "")
    if concept == "GLUCOSE" and u == "mmol/l":
        return value * 18.0182, "mg/dL"
    if concept == "CHOLESTEROL" and u == "mmol/l":
        return value * 38.67, "mg/dL"
    if concept == "CREATININE" and u == "umol/l":
        return value / 88.42, "mg/dL"
    if concept == "HEMOGLOBIN" and u == "g/l":
        return value / 10.0, "g/dL"
    if concept == "HBA1C" and u == "mmol/mol":
        return (value * 0.09148) + 2.152, "%"
    if concept == "PLATELETS" and u in ("10*9/l", "10^9/l"):
        return value, "10*3/uL"               # numerically identical
    if concept in ("ALT", "AST") and u in ("iu/l",):
        return value, "U/L"                    # label variant only
    return value, LOINC[concept]["unit"]

def raw_value(concept, canonical_unit_value, unit):
    """Express a canonical value back in the site's raw unit (inverse)."""
    u = unit.lower().replace(" ", "")
    v = canonical_unit_value
    if concept == "GLUCOSE" and u == "mmol/l":     return v / 18.0182
    if concept == "CHOLESTEROL" and u == "mmol/l": return v / 38.67
    if concept == "CREATININE" and u == "umol/l":  return v * 88.42
    if concept == "HEMOGLOBIN" and u == "g/l":     return v * 10.0
    if concept == "HBA1C" and u == "mmol/mol":     return (v - 2.152) / 0.09148
    return v

def messy(name: str) -> str:
    """Introduce realistic casing / whitespace / punctuation noise."""
    r = RNG.random()
    if r < 0.15:  name = name.upper()
    elif r < 0.30: name = name.lower()
    if RNG.random() < 0.10: name = "  " + name + " "
    if RNG.random() < 0.06: name = name.replace("a", "@", 1)   # rare OCR-style typo
    return name

rows = []
for site, cfg in SITES.items():
    for _ in range(cfg["n"]):
        concept = RNG.choice(list(SYNONYMS.keys()))
        mu, sd = DIST[concept]
        canon_val = max(0.1, RNG.normal(mu, sd))
        unit = cfg["units"][concept]
        rv = round(raw_value(concept, canon_val, unit), 2)
        nm = messy(cfg["names"][concept])
        rows.append({"source": site, "raw_test_name": nm, "raw_value": rv,
                     "raw_unit": unit, "_true_concept": concept})

# Inject a small fraction of genuinely un-mappable / ambiguous entries (honesty)
for _ in range(180):
    rows.append({"source": RNG.choice(list(SITES.keys())),
                 "raw_test_name": RNG.choice(["LFT panel", "Misc chemistry",
                                              "Other", "Unlisted analyte", "Chem-7"]),
                 "raw_value": round(RNG.random() * 100, 2),
                 "raw_unit": RNG.choice(["", "U/L", "mg/dL"]),
                 "_true_concept": "UNMAPPED"})

raw = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
raw.insert(0, "record_id", ["R%05d" % i for i in range(len(raw))])
raw.to_csv("data/raw_pooled_lab_data.csv", index=False)

# ---------------------------------------------------------------------------
# 3. HARMONIZATION PIPELINE
#    dictionary/ontology match  ->  fuzzy semantic match  ->  flag for review
# ---------------------------------------------------------------------------
def harmonize(name):
    key = norm(name)
    if key in NORM_TO_CONCEPT:
        return NORM_TO_CONCEPT[key], "exact-synonym"
    hit = get_close_matches(key, VOCAB, n=1, cutoff=0.82)
    if hit:
        return NORM_TO_CONCEPT[hit[0]], "fuzzy-semantic"
    return None, "flagged-for-review"

concepts, methods = [], []
for nm in raw["raw_test_name"]:
    c, m = harmonize(nm)
    concepts.append(c); methods.append(m)
raw["concept"] = concepts
raw["map_method"] = methods

def enrich(row):
    c = row["concept"]
    if c is None or (isinstance(c, float) and np.isnan(c)) or c not in LOINC:
        return pd.Series([None, None, None, None])
    cv, cu = to_canonical(c, row["raw_value"], row["raw_unit"])
    meta = LOINC[c]
    return pd.Series([meta["loinc"], meta["name"], round(cv, 2), meta["unit"]])

raw[["loinc_code", "loinc_name", "harmonized_value", "harmonized_unit"]] = \
    raw.apply(enrich, axis=1)

golden = raw[raw["concept"].notna()].copy()
golden_out = golden[["record_id", "source", "raw_test_name", "raw_value", "raw_unit",
                     "loinc_code", "loinc_name", "harmonized_value",
                     "harmonized_unit", "map_method"]]
golden_out.to_csv("data/harmonized_golden_dataset.csv", index=False)

# ---------------------------------------------------------------------------
# 4. METRICS  (before vs after)
# ---------------------------------------------------------------------------
total = len(raw)
mappable_truth = (raw["_true_concept"] != "UNMAPPED")
distinct_raw_terms = raw["raw_test_name"].str.strip().str.lower().nunique()
distinct_raw_units = raw["raw_unit"].replace("", np.nan).nunique()
mapped = raw["concept"].notna()
n_exact = (raw["map_method"] == "exact-synonym").sum()
n_fuzzy = (raw["map_method"] == "fuzzy-semantic").sum()
n_flag = (raw["map_method"] == "flagged-for-review").sum()

# Accuracy of automated mapping vs known truth (excluding truly-unmapped)
correct = ((raw["concept"] == raw["_true_concept"]) & mapped).sum()
auto_accuracy = correct / mappable_truth.sum() * 100

# Cross-site findability: a naive query for the canonical term "ALT", "AST"...
# BEFORE harmonization only matches rows whose raw name equals that exact term.
find_before, find_after = {}, {}
for concept in SYNONYMS:
    truth_rows = raw["_true_concept"] == concept
    n_truth = truth_rows.sum()
    canonical_term = concept.lower()
    naive_hits = (raw["raw_test_name"].apply(norm) == canonical_term) & truth_rows
    find_before[concept] = naive_hits.sum() / n_truth * 100
    find_after[concept] = (raw.loc[truth_rows, "concept"] == concept).mean() * 100

metrics = {
    "records_total": int(total),
    "sources": list(SITES.keys()),
    "analytes": len(SYNONYMS),
    "distinct_raw_terms": int(distinct_raw_terms),
    "harmonized_concepts": int(len(SYNONYMS)),
    "distinct_raw_units": int(distinct_raw_units),
    "harmonized_units": int(golden["harmonized_unit"].nunique()),
    "auto_mapped_pct": round(mapped.sum() / total * 100, 1),
    "exact_pct": round(n_exact / total * 100, 1),
    "fuzzy_pct": round(n_fuzzy / total * 100, 1),
    "flagged_pct": round(n_flag / total * 100, 1),
    "auto_mapping_accuracy_pct": round(auto_accuracy, 1),
    "manual_curation_before_pct": 100.0,
    "manual_curation_after_pct": round(n_flag / total * 100, 1),
    "findability_before_avg_pct": round(np.mean(list(find_before.values())), 1),
    "findability_after_avg_pct": round(np.mean(list(find_after.values())), 1),
    "find_before": {k: round(v, 1) for k, v in find_before.items()},
    "find_after": {k: round(v, 1) for k, v in find_after.items()},
}
with open("working/casestudy/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("=== HARMONIZATION CASE STUDY — RESULTS ===")
print(f"Pooled records:            {total:,} across {len(SITES)} sources, {len(SYNONYMS)} analytes")
print(f"Distinct raw term strings: {distinct_raw_terms}  ->  {len(SYNONYMS)} harmonized LOINC concepts")
print(f"Distinct raw units:        {distinct_raw_units}  ->  {golden['harmonized_unit'].nunique()} canonical units")
print(f"Auto-mapped to LOINC:      {metrics['auto_mapped_pct']}%  "
      f"(exact {metrics['exact_pct']}% + fuzzy {metrics['fuzzy_pct']}%)")
print(f"Flagged for human review:  {metrics['flagged_pct']}%")
print(f"Automated mapping accuracy:{metrics['auto_mapping_accuracy_pct']}%  (vs known truth)")
print(f"Cross-site findability:    {metrics['findability_before_avg_pct']}%  ->  "
      f"{metrics['findability_after_avg_pct']}%")
print("Wrote: raw_pooled_lab_data.csv, harmonized_golden_dataset.csv, metrics.json")
