import os
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / "live_demo")]
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

import server as s  # noqa: E402
from live_tools import SYSTEM_INSTRUCTION, TOOL_NAMES, locale_instruction, tool_declarations  # noqa: E402
from locales import resolve_country  # noqa: E402


class ResolveCountryTest(unittest.TestCase):
    def test_names_aliases_codes_and_misspellings(self):
        for spoken, expected in [("Germany", ("Germany", "EUR")), ("the UK", ("United Kingdom", "GBP")),
                                 ("USA", ("United States", "USD")), ("in", ("India", "INR")),
                                 ("Deutschland", ("Germany", "EUR")), ("Germnay", ("Germany", "EUR")),
                                 ("  india ", ("India", "INR")), ("Holland", ("Netherlands", "EUR"))]:
            self.assertEqual(resolve_country(spoken), expected, spoken)

    def test_iso_codes_for_price_lookups(self):
        from locales import country_code
        self.assertEqual((country_code("Germany"), country_code("United Kingdom"), country_code("Atlantis")), ("DE", "GB", ""))

    def test_unknown_country(self):
        self.assertIsNone(resolve_country("Atlantis"))
        self.assertIsNone(resolve_country(""))


class SetCountryTest(unittest.TestCase):
    def test_currency_follows_country(self):
        session = s.IntakeSession("a")
        result = s.set_country(session, "germany", "USD")   # a stated currency never overrides a known country
        self.assertEqual((session.country, session.currency), ("Germany", "EUR"))
        self.assertEqual(result["currency_source"], "derived from country")

    def test_unknown_country_needs_stated_currency(self):
        session = s.IntakeSession("b")
        with self.assertRaises(ValueError) as caught:
            s.set_country(session, "Atlantis")
        self.assertIn("Ask which currency", str(caught.exception))
        result = s.set_country(session, "Atlantis", "eur")
        self.assertEqual((session.country, session.currency, result["currency_source"]), ("Atlantis", "EUR", "stated by policyholder"))


class LocaleEndpointTest(unittest.TestCase):
    def setUp(self):
        s.sessions.clear()
        self.client = TestClient(s.app, base_url="http://127.0.0.1:4188")
        self.headers = {"Origin": "http://127.0.0.1:4188"}

    def tearDown(self):
        self.client.close()
        s.sessions.clear()

    def test_country_only_derives_currency(self):
        sid = self.client.post("/api/sessions", headers=self.headers).json()["session_id"]
        response = self.client.put(f"/api/sessions/{sid}/locale", json={"country": "Japan"}, headers=self.headers)
        self.assertEqual(response.json()["locale"], {"country": "Japan", "currency": "JPY"})
        self.assertEqual(self.client.put(f"/api/sessions/{sid}/locale", json={"country": "Atlantis"}, headers=self.headers).status_code, 422)


class AgentPromptTest(unittest.TestCase):
    def test_agent_tools(self):
        expected = ["set_purpose", "set_country", "log_item", "analyse_books", "get_books", "exclude_book", "set_book_copies",
                    "set_book_count"]
        self.assertEqual([d.name for t in tool_declarations() for d in t.function_declarations], expected)
        self.assertEqual(TOOL_NAMES, expected)

    def test_prompt_is_the_contents_insurance_agent(self):
        for phrase in ("home-contents insurance", "make a claim", "add items", "set_purpose", "logs the objects it sees", "set_book_count",
                       "log_item", "books agent", "analyse_books", "get_books", "Row 1, the top shelf",
                       "Never state a book title", "Never promise coverage"):
            self.assertIn(phrase, SYSTEM_INSTRUCTION)
        for old in ("first notice of loss", "lookup_policy", "draw_incident_sketch"):
            self.assertNotIn(old, SYSTEM_INSTRUCTION)

    def test_books_are_handled_only_once_they_are_in_view(self):
        self.assertIn("When books are in view", SYSTEM_INSTRUCTION)
        self.assertIn("Never ask them to show books one at a time", SYSTEM_INSTRUCTION)

    def test_locale_notice_asks_for_country_when_unset(self):
        self.assertIn("set_country", locale_instruction("", ""))
        self.assertIn("derived from the country", locale_instruction("Germany", "EUR"))


class PurposeAndItemsTest(unittest.TestCase):
    def test_purpose(self):
        session = s.IntakeSession("p")
        self.assertEqual(s.set_purpose(session, "claim", "water damage last week")["purpose"], "claim")
        self.assertEqual(session.purpose_details, "water damage last week")
        with self.assertRaises(ValueError):
            s.set_purpose(session, "something else")

    def test_log_update_and_no_duplicates(self):
        session = s.IntakeSession("i")
        first = s.log_item(session, {"category": "Furniture", "description": "grey three-seat sofa", "where": "living room"})
        self.assertEqual((first["item_id"], first["items_logged"]), ("I001", 1))
        again = s.log_item(session, {"category": "furniture", "description": "Grey three-seat sofa"})
        self.assertTrue(again["already_logged"])
        s.log_item(session, {"item_id": "i001", "brand_model": "IKEA Kivik", "note": "bought 2021"})
        self.assertEqual((session.items[0]["brand_model"], session.items[0]["notes"]), ("IKEA Kivik", ["bought 2021"]))
        self.assertEqual(session.items[0]["category"], "furniture")
        with self.assertRaises(ValueError):
            s.log_item(session, {"item_id": "I999", "note": "x"})


if __name__ == "__main__":
    unittest.main()
