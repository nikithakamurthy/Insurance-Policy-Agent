"""Country -> currency for the pricing locale. The agent hears the country; the currency follows."""

from __future__ import annotations

import difflib
import re
import unicodedata

EURO = ["Austria", "Belgium", "Croatia", "Cyprus", "Estonia", "Finland", "France", "Germany", "Greece", "Ireland",
        "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Portugal", "Slovakia", "Slovenia", "Spain"]

COUNTRY_CURRENCY: dict[str, str] = {
    **{name: "EUR" for name in EURO},
    "Argentina": "ARS", "Australia": "AUD", "Bangladesh": "BDT", "Brazil": "BRL", "Bulgaria": "BGN",
    "Canada": "CAD", "Chile": "CLP", "China": "CNY", "Colombia": "COP", "Czech Republic": "CZK",
    "Denmark": "DKK", "Egypt": "EGP", "Hong Kong": "HKD", "Hungary": "HUF", "Iceland": "ISK", "India": "INR",
    "Indonesia": "IDR", "Israel": "ILS", "Japan": "JPY", "Kenya": "KES", "Malaysia": "MYR", "Mexico": "MXN",
    "Morocco": "MAD", "Nepal": "NPR", "New Zealand": "NZD", "Nigeria": "NGN", "Norway": "NOK", "Pakistan": "PKR",
    "Peru": "PEN", "Philippines": "PHP", "Poland": "PLN", "Qatar": "QAR", "Romania": "RON", "Saudi Arabia": "SAR",
    "Singapore": "SGD", "South Africa": "ZAR", "South Korea": "KRW", "Sri Lanka": "LKR", "Sweden": "SEK",
    "Switzerland": "CHF", "Taiwan": "TWD", "Thailand": "THB", "Turkey": "TRY", "Ukraine": "UAH",
    "United Arab Emirates": "AED", "United Kingdom": "GBP", "United States": "USD", "Vietnam": "VND",
}

# ISO 3166-1 alpha-2 code for every country above; price lookups are keyed by these codes.
COUNTRY_CODES: dict[str, str] = {
    "Austria": "AT", "Belgium": "BE", "Croatia": "HR", "Cyprus": "CY", "Estonia": "EE", "Finland": "FI",
    "France": "FR", "Germany": "DE", "Greece": "GR", "Ireland": "IE", "Italy": "IT", "Latvia": "LV",
    "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT", "Netherlands": "NL", "Portugal": "PT", "Slovakia": "SK",
    "Slovenia": "SI", "Spain": "ES", "Argentina": "AR", "Australia": "AU", "Bangladesh": "BD", "Brazil": "BR",
    "Bulgaria": "BG", "Canada": "CA", "Chile": "CL", "China": "CN", "Colombia": "CO", "Czech Republic": "CZ",
    "Denmark": "DK", "Egypt": "EG", "Hong Kong": "HK", "Hungary": "HU", "Iceland": "IS", "India": "IN",
    "Indonesia": "ID", "Israel": "IL", "Japan": "JP", "Kenya": "KE", "Malaysia": "MY", "Mexico": "MX",
    "Morocco": "MA", "Nepal": "NP", "New Zealand": "NZ", "Nigeria": "NG", "Norway": "NO", "Pakistan": "PK",
    "Peru": "PE", "Philippines": "PH", "Poland": "PL", "Qatar": "QA", "Romania": "RO", "Saudi Arabia": "SA",
    "Singapore": "SG", "South Africa": "ZA", "South Korea": "KR", "Sri Lanka": "LK", "Sweden": "SE",
    "Switzerland": "CH", "Taiwan": "TW", "Thailand": "TH", "Turkey": "TR", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB", "United States": "US", "Vietnam": "VN",
}


def country_code(name: str) -> str:
    """ISO alpha-2 code for a country name from this table, or '' if unknown."""
    return COUNTRY_CODES.get(name, "")


ALIASES = {
    "uk": "United Kingdom", "great britain": "United Kingdom", "britain": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "wales": "United Kingdom", "northern ireland": "United Kingdom", "gb": "United Kingdom",
    "us": "United States", "usa": "United States", "united states of america": "United States", "america": "United States",
    "uae": "United Arab Emirates", "emirates": "United Arab Emirates", "dubai": "United Arab Emirates",
    "holland": "Netherlands", "the netherlands": "Netherlands", "deutschland": "Germany", "espana": "Spain",
    "czechia": "Czech Republic", "korea": "South Korea", "republic of korea": "South Korea", "turkiye": "Turkey",
    "bharat": "India", "nz": "New Zealand", "ksa": "Saudi Arabia",
    # ISO 3166 alpha-2 codes for the countries above.
    **{code: name for code, name in {
        "at": "Austria", "be": "Belgium", "hr": "Croatia", "cy": "Cyprus", "ee": "Estonia", "fi": "Finland",
        "fr": "France", "de": "Germany", "gr": "Greece", "ie": "Ireland", "it": "Italy", "lv": "Latvia",
        "lt": "Lithuania", "lu": "Luxembourg", "mt": "Malta", "nl": "Netherlands", "pt": "Portugal", "sk": "Slovakia",
        "si": "Slovenia", "es": "Spain", "ar": "Argentina", "au": "Australia", "br": "Brazil", "ca": "Canada",
        "cn": "China", "dk": "Denmark", "in": "India", "jp": "Japan", "mx": "Mexico", "no": "Norway",
        "pl": "Poland", "se": "Sweden", "ch": "Switzerland", "sg": "Singapore", "za": "South Africa", "kr": "South Korea",
    }.items()},
}


def _key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text)).strip()
    return re.sub(r"^the ", "", key)   # "the UK", "the Netherlands" as spoken


def resolve_country(text: str) -> tuple[str, str] | None:
    """(country name, ISO 4217 currency) for a spoken or typed country, or None if unknown.

    Accepts names, common aliases, two-letter codes and small misspellings ("Germnay").
    """
    key = _key(text)
    if not key:
        return None
    names = {_key(name): name for name in COUNTRY_CURRENCY}
    name = names.get(key) or ALIASES.get(key)
    if name is None:
        close = difflib.get_close_matches(key, list(names) + [a for a in ALIASES if len(a) > 3], n=1, cutoff=0.85)
        if close:
            name = names.get(close[0]) or ALIASES.get(close[0])
    return (name, COUNTRY_CURRENCY[name]) if name else None
