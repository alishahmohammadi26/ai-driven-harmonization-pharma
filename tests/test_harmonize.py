"""
Tests for the harmonize pipeline — normalization, synonym lookup,
fuzzy matching, and unit conversion.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from harmonize import norm, harmonize, to_canonical, LOINC, SYNONYMS


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class TestNorm:
    def test_strips_whitespace(self):
        assert norm("  ALT  ") == "alt"

    def test_lowercases(self):
        assert norm("Hemoglobin") == "hemoglobin"

    def test_removes_punctuation(self):
        assert norm("ALT (SGPT)") == "alt sgpt"

    def test_collapses_spaces(self):
        assert norm("alanine  aminotransferase") == "alanine aminotransferase"


# ---------------------------------------------------------------------------
# Synonym dictionary lookup
# ---------------------------------------------------------------------------
class TestDictionaryLookup:
    @pytest.mark.parametrize("raw,expected_concept", [
        ("ALT",                   "ALT"),
        ("SGPT",                  "ALT"),
        ("Alanine Aminotransferase", "ALT"),
        ("ALAT",                  "ALT"),
        ("HbA1c",                 "HBA1C"),
        ("Glycated Hemoglobin",   "HBA1C"),
        ("PLT",                   "PLATELETS"),
        ("Platelet Count",        "PLATELETS"),
        ("Haemoglobin",           "HEMOGLOBIN"),
    ])
    def test_exact_synonym_match(self, raw, expected_concept):
        concept, method = harmonize(raw)
        assert concept == expected_concept, f"'{raw}' mapped to '{concept}'"
        assert method == "exact-synonym"

    def test_unmappable_returns_none(self):
        concept, method = harmonize("Chem-7 panel")
        assert concept is None
        assert method == "flagged-for-review"


# ---------------------------------------------------------------------------
# Fuzzy fallback
# ---------------------------------------------------------------------------
class TestFuzzyLookup:
    def test_fuzzy_near_miss(self):
        # "Haemoglbin" (typo) should fuzzy-match to HEMOGLOBIN
        concept, method = harmonize("Haemoglbin")
        assert concept == "HEMOGLOBIN"
        assert method == "fuzzy-semantic"


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
class TestUnitConversion:
    def test_glucose_mmol_to_mgdl(self):
        val, unit = to_canonical("GLUCOSE", 5.0, "mmol/L")
        assert abs(val - 90.09) < 0.1
        assert unit == "mg/dL"

    def test_glucose_canonical_passthrough(self):
        val, unit = to_canonical("GLUCOSE", 95.0, "mg/dL")
        assert val == 95.0
        assert unit == "mg/dL"

    def test_creatinine_umol_to_mgdl(self):
        val, unit = to_canonical("CREATININE", 88.42, "umol/L")
        assert abs(val - 1.0) < 0.01
        assert unit == "mg/dL"

    def test_hemoglobin_g_l_to_g_dl(self):
        val, unit = to_canonical("HEMOGLOBIN", 138.0, "g/L")
        assert abs(val - 13.8) < 0.01
        assert unit == "g/dL"

    def test_alt_iu_l_label_variant(self):
        val, unit = to_canonical("ALT", 32.0, "IU/L")
        assert val == 32.0
        assert unit == "U/L"

    def test_hba1c_mmol_mol_to_percent(self):
        val, unit = to_canonical("HBA1C", 42.0, "mmol/mol")
        assert abs(val - 5.998) < 0.01    # IFCC 42 mmol/mol ≈ 6.0%
        assert unit == "%"


# ---------------------------------------------------------------------------
# LOINC reference integrity
# ---------------------------------------------------------------------------
class TestLOINCReference:
    def test_all_synonyms_have_loinc_entry(self):
        for concept in SYNONYMS:
            assert concept in LOINC, f"No LOINC entry for {concept}"

    def test_loinc_codes_are_strings(self):
        for concept, meta in LOINC.items():
            assert isinstance(meta["loinc"], str)
            assert len(meta["loinc"]) > 0
