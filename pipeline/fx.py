"""Currency conversion with European Central Bank euro reference rates (free, official, dated)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .common import get_cached

ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"


@dataclass
class Rates:
    """Units of each currency per 1 EUR, as published by the ECB for one day."""
    per_eur: dict[str, float]
    date: str
    retrieved_at: str
    source: str = "European Central Bank euro foreign exchange reference rates"
    url: str = ECB_URL

    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> tuple[float, float]:
        """Return (converted amount, rate used) where rate = to_ccy units per from_ccy unit."""
        from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()
        for ccy in (from_ccy, to_ccy):
            if ccy not in self.per_eur:
                raise KeyError(f"No ECB reference rate for {ccy}")
        rate = self.per_eur[to_ccy] / self.per_eur[from_ccy]
        return round(amount * rate, 2), rate


def parse_ecb(xml: str, retrieved_at: str) -> Rates:
    date = re.search(r"time=['\"](\d{4}-\d{2}-\d{2})['\"]", xml)
    pairs = re.findall(r"currency=['\"]([A-Z]{3})['\"]\s+rate=['\"]([\d.]+)['\"]", xml)
    if not date or not pairs:
        raise ValueError("Unexpected ECB reference-rate format")
    return Rates({"EUR": 1.0, **{c: float(r) for c, r in pairs}}, date.group(1), retrieved_at)


def load_rates() -> Rates:
    xml, retrieved_at = get_cached(ECB_URL, max_age_s=12 * 3600, raw=True)
    return parse_ecb(xml, retrieved_at)
