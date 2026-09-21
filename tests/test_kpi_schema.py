"""
Unit tests for KPI Pydantic schemas in backend/agents/kpi_extractor.py.

No LLM calls — tests the schema contracts directly:
- KPIValue defaults to not_found confidence with null value
- Valid KPIValue with value and unit passes validation
- FinancialKPIs has all required fields, all defaulting to not_found
- Invalid unit raises ValidationError
- Explicit "not found" state is always representable
- yoy_revenue_growth is optional (can be None)
- model_dump round-trips correctly
"""
from __future__ import annotations

import unittest

import pytest
from pydantic import ValidationError

from backend.agents.kpi_extractor import FinancialKPIs, KPIValue


class TestKPIValueDefaults(unittest.TestCase):
    def test_default_confidence_is_not_found(self):
        kpi = KPIValue()
        self.assertEqual(kpi.confidence, "not_found")

    def test_default_value_is_none(self):
        kpi = KPIValue()
        self.assertIsNone(kpi.value)

    def test_default_unit_is_none(self):
        kpi = KPIValue()
        self.assertIsNone(kpi.unit)

    def test_default_source_page_is_none(self):
        kpi = KPIValue()
        self.assertIsNone(kpi.source_page)

    def test_default_raw_text_snippet_is_none(self):
        kpi = KPIValue()
        self.assertIsNone(kpi.raw_text_snippet)


class TestKPIValueValidValues(unittest.TestCase):
    def test_valid_usd_millions(self):
        kpi = KPIValue(value=1500.0, unit="USD_millions", confidence="high", source_page=4)
        self.assertEqual(kpi.value, 1500.0)
        self.assertEqual(kpi.unit, "USD_millions")
        self.assertEqual(kpi.confidence, "high")
        self.assertEqual(kpi.source_page, 4)

    def test_valid_percent_unit(self):
        kpi = KPIValue(value=12.5, unit="percent", confidence="medium")
        self.assertEqual(kpi.unit, "percent")

    def test_valid_ratio_unit(self):
        kpi = KPIValue(value=2.3, unit="ratio", confidence="low")
        self.assertEqual(kpi.unit, "ratio")

    def test_valid_shares_unit(self):
        kpi = KPIValue(value=1_000_000, unit="shares", confidence="high")
        self.assertEqual(kpi.unit, "shares")

    def test_valid_usd_thousands(self):
        kpi = KPIValue(value=500_000.0, unit="USD_thousands", confidence="high")
        self.assertEqual(kpi.unit, "USD_thousands")

    def test_valid_usd(self):
        kpi = KPIValue(value=42.50, unit="USD", confidence="high")
        self.assertEqual(kpi.unit, "USD")

    def test_all_confidence_literals(self):
        for conf in ("high", "medium", "low", "not_found"):
            kpi = KPIValue(confidence=conf)
            self.assertEqual(kpi.confidence, conf)

    def test_raw_text_snippet_stored(self):
        kpi = KPIValue(value=100.0, unit="USD_millions", raw_text_snippet="Revenue was $100M")
        self.assertEqual(kpi.raw_text_snippet, "Revenue was $100M")

    def test_source_chunk_id_stored(self):
        kpi = KPIValue(source_chunk_id="chunk-abc-123")
        self.assertEqual(kpi.source_chunk_id, "chunk-abc-123")


class TestKPIValueInvalidInputs(unittest.TestCase):
    def test_invalid_unit_raises(self):
        with self.assertRaises(ValidationError):
            KPIValue(unit="euros")  # Not in the allowed Literal

    def test_invalid_confidence_raises(self):
        with self.assertRaises(ValidationError):
            KPIValue(confidence="very_high")  # Not in the allowed Literal


class TestFinancialKPIsDefaults(unittest.TestCase):
    """FinancialKPIs must default every field to not_found — the hallucination guard."""

    REQUIRED_FIELDS = [
        "revenue", "cogs", "gross_margin", "net_income",
        "eps", "total_assets", "total_liabilities", "total_equity",
    ]

    def test_all_required_fields_present(self):
        kpis = FinancialKPIs()
        for field in self.REQUIRED_FIELDS:
            self.assertTrue(hasattr(kpis, field), f"Missing field: {field}")

    def test_all_required_fields_default_to_not_found(self):
        kpis = FinancialKPIs()
        for field in self.REQUIRED_FIELDS:
            kpi_val: KPIValue = getattr(kpis, field)
            self.assertEqual(
                kpi_val.confidence, "not_found",
                f"{field} should default to not_found, got {kpi_val.confidence}",
            )

    def test_all_required_fields_default_value_is_none(self):
        kpis = FinancialKPIs()
        for field in self.REQUIRED_FIELDS:
            kpi_val: KPIValue = getattr(kpis, field)
            self.assertIsNone(kpi_val.value, f"{field}.value should default to None")

    def test_yoy_revenue_growth_is_optional(self):
        kpis = FinancialKPIs()
        self.assertIsNone(kpis.yoy_revenue_growth)

    def test_yoy_revenue_growth_can_be_set(self):
        kpis = FinancialKPIs(yoy_revenue_growth=KPIValue(value=5.2, unit="percent", confidence="medium"))
        self.assertIsNotNone(kpis.yoy_revenue_growth)
        self.assertEqual(kpis.yoy_revenue_growth.value, 5.2)


class TestFinancialKPIsValidConstruction(unittest.TestCase):
    def test_partial_population(self):
        """Only set revenue; others should remain not_found."""
        kpis = FinancialKPIs(
            revenue=KPIValue(value=250.0, unit="USD_millions", confidence="high", source_page=2)
        )
        self.assertEqual(kpis.revenue.value, 250.0)
        self.assertEqual(kpis.revenue.confidence, "high")
        # Others untouched
        self.assertEqual(kpis.net_income.confidence, "not_found")
        self.assertIsNone(kpis.net_income.value)

    def test_full_population(self):
        """Populate all fields and verify."""
        kpis = FinancialKPIs(
            revenue=KPIValue(value=500.0, unit="USD_millions", confidence="high"),
            net_income=KPIValue(value=50.0, unit="USD_millions", confidence="high"),
            eps=KPIValue(value=1.25, unit="USD", confidence="medium"),
            total_assets=KPIValue(value=2000.0, unit="USD_millions", confidence="high"),
            total_liabilities=KPIValue(value=800.0, unit="USD_millions", confidence="high"),
            total_equity=KPIValue(value=1200.0, unit="USD_millions", confidence="high"),
            gross_margin=KPIValue(value=45.0, unit="percent", confidence="medium"),
            cogs=KPIValue(value=275.0, unit="USD_millions", confidence="high"),
        )
        self.assertEqual(kpis.revenue.value, 500.0)
        self.assertEqual(kpis.eps.unit, "USD")

    def test_model_dump_round_trip(self):
        """model_dump() must produce a dict that can reconstruct the model."""
        original = FinancialKPIs(
            revenue=KPIValue(value=100.0, unit="USD_millions", confidence="high", source_page=1)
        )
        dumped = original.model_dump()
        reconstructed = FinancialKPIs(**dumped)
        self.assertEqual(reconstructed.revenue.value, 100.0)
        self.assertEqual(reconstructed.revenue.confidence, "high")

    def test_not_found_is_always_serializable(self):
        """Ensure a fully-default not_found KPIs dict is JSON-serializable."""
        import json
        kpis = FinancialKPIs()
        dumped = kpis.model_dump()
        json_str = json.dumps(dumped)
        self.assertIn("not_found", json_str)


if __name__ == "__main__":
    unittest.main()
