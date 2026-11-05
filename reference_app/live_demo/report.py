"""The readable report, built from claim_packet.json (packet.py): every figure on the page is a
value in the packet, so the two cannot disagree. Shown in the app and saved next to the packet.

Frame links are relative (frames/f00012.jpg), so they open from the saved file and from the app.
"""

from __future__ import annotations

import html
import time
from collections import Counter
from typing import Any

def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _money(amount: Any, currency: str) -> str:
    return "—" if amount is None else f"{amount:,.2f} {_e(currency)}"


def _link(url: Any, text: str) -> str:
    return f'<a href="{_e(url)}" target="_blank" rel="noopener noreferrer">{_e(text)}</a>' if url else _e(text)


def _price_cell(price: dict | None, currency: str) -> str:
    """Amount, then where it came from, when, and what was assumed."""
    if not price:
        return '<span class="muted">not priced</span>'
    if price.get("amount") is None:
        return f'<span class="muted">not priced: {_e(price.get("reason", "no price found"))}</span>'
    detail = [_link(price.get("url"), price.get("source") or "source")]
    if price.get("country"):
        detail.append(_e(price["country"]))
    if price.get("retrieved_at"):
        detail.append(_e(str(price["retrieved_at"])[:10]))
    if price.get("converted") and price.get("original"):
        fx = price.get("fx") or {}
        detail.append(f"converted from {price['original']['amount']:,.2f} {_e(price['original']['currency'])}"
                      + (f" at {fx.get('rate')} ({_e(fx.get('source', ''))}, {_e(fx.get('date', ''))})" if fx else ""))
    if price.get("condition_assumed"):
        detail.append(_e(price["condition_assumed"]))
    if price.get("instead_of"):
        detail.append(_e(f"no local price ({price['instead_of']})"))
    tag = '<span class="ebook-tag">EBOOK PRICE</span> ' if price.get("edition") == "eBook" else ""
    return f'{tag}<strong>{_money(price["amount"], price.get("currency") or currency)}</strong><div class="small">{" · ".join(detail)}</div>'


def _frame_link(path: str | None, text: str = "frame") -> str:
    return f'<a href="{_e(path)}" target="_blank">{_e(text)}</a>' if path else '<span class="muted">no frame</span>'


def _size_cell(book: dict) -> str:
    h, t = book.get("spine_height_cm"), book.get("spine_thickness_cm")
    if h is None and t is None:
        return '<span class="muted">not measured</span>'
    labels = {"card": "bank card", "ruler": "measured books as rulers", "model": "estimated (Depth Pro)"}
    label = labels.get(book.get("size_method"), "") if h is not None else "thickness only; height not measured"
    return (f'{"?" if h is None else h} × {"?" if t is None else t} cm'
            f'<div class="small">{_e(label)}{" · estimate" if book.get("size_estimated") else ""}</div>')


SERVICE_LABELS = {"gemini_live": "Gemini Live (voice agent)", "gemini_matcher": "Gemini (book matching)",
                  "google_books": "Google Books API", "open_library": "Open Library", "ecb_rates": "ECB exchange rates",
                  "ebay": "eBay Browse API"}


def _usage_section(usage: dict | None) -> str:
    """Cost per sweep (requests, tokens, USD where rates are set) and latency per stage."""
    if not usage:
        return ""
    money = usage.get("cost") or {}
    by_service = money.get("usd_by_service", {})
    call_rows = "".join(
        f"<tr><td>{_e(SERVICE_LABELS.get(name, name))}</td><td class=\"num\">{c['requests']}</td><td class=\"num\">{c['errors']}</td>"
        f"<td class=\"num\">{c['input_tokens']:,}</td><td class=\"num\">{c['output_tokens']:,}</td>"
        f"<td class=\"num\">{'—' if name not in by_service else f'${by_service[name]:.4f}'}</td></tr>"
        for name, c in sorted(usage.get("calls", {}).items()))
    stage_rows = "".join(
        f"<tr><td>{_e(name)}</td><td class=\"num\">{st['count']}</td><td class=\"num\">{st['mean_s']:.2f}</td>"
        f"<td class=\"num\">{st['p95_s']:.2f}</td><td class=\"num\">{st['max_s']:.2f}</td><td class=\"num\">{st['total_s']:.1f}</td></tr>"
        for name, st in sorted(usage.get("stages", {}).items()))
    total = money.get("usd_total")
    unpriced = money.get("unpriced_services", [])
    cost_note = (f"Estimated cost of this sweep: <strong>${total:.4f}</strong> from the configured rates."
                 if total is not None else "No rates configured, so no money figure.")
    if unpriced:
        cost_note += (" Not priced (no rate set): " + ", ".join(_e(SERVICE_LABELS.get(u, u)) for u in unpriced)
                      + f". Rates go in {_e(money.get('rates_file', 'usage_rates.json'))}.")
    if usage.get("gemini_live_running_total_tokens"):
        cost_note += (f" Gemini Live also reported a running total of {usage['gemini_live_running_total_tokens']:,} tokens; "
                      "if that is a session total, it is the true count rather than the per-message sum above.")
    if not usage.get("book_worker_reached", True):
        cost_note += " The book worker did not answer, so its stages and requests are missing."
    return f"""
  <h2>Cost and latency of this sweep</h2>
  <div class="note">{cost_note} Local models (detection, OCR, SAM, Depth Pro) run on this computer and cost nothing per call.</div>
  <div class="table-wrap"><table><thead><tr><th>Service</th><th class="num">Requests</th><th class="num">Errors</th>
    <th class="num">Input tokens</th><th class="num">Output tokens</th><th class="num">Cost</th></tr></thead>
  <tbody>{call_rows or '<tr><td colspan="6" class="muted">No outside requests.</td></tr>'}</tbody></table></div>
  <div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>Stage</th><th class="num">Runs</th><th class="num">Mean s</th>
    <th class="num">p95 s</th><th class="num">Max s</th><th class="num">Total s</th></tr></thead>
  <tbody>{stage_rows or '<tr><td colspan="6" class="muted">Nothing timed.</td></tr>'}</tbody></table></div>"""


def _room_section(room: dict) -> str:
    """Length and width from the corner measurement, or why they are empty."""
    if room.get("length_m") is None:
        return '<div class="note">Room length and width not measured in this sweep; the review queue says why.</div>'
    walls = " · ".join(f"wall {_e(w['wall'])}: {w['distance_m']} m ({w['measured']} frames)" for w in room.get("walls", []))
    frames = " ".join(_frame_link(f, f.split("/")[-1].removesuffix(".jpg")) for f in room.get("frames_used", []))
    rows = [("Length × width", f"<strong>{room['length_m']} × {room['width_m']} m</strong>"),
            ("Floor area", f"{room['floor_area_m2']} m² <div class=\"small\">length × width, a rectangular room assumed</div>"),
            ("How", f"{_e(room.get('scale_method'))}<div class=\"small\">camera to each wall: {walls}</div>"),
            ("Confidence", _e(room.get("confidence"))),
            ("Frames", frames or "—")]
    return '<div class="table-wrap"><table><tbody>' + "".join(f"<tr><td>{_e(k)}</td><td>{v}</td></tr>" for k, v in rows) + "</tbody></table></div>"


def build_report(packet: dict, usage: dict | None = None) -> str:
    """The page for one packet, plus cost and latency for the sweep."""
    sweep, totals = packet["sweep"], packet["totals"]
    currency = totals.get("currency") or sweep.get("currency") or ""
    books, items, review = packet["books"], packet["items"], packet["review_queue"]
    left_out = packet.get("excluded_books", [])

    by_type = Counter()
    for i in items:
        by_type[i.get("category") or "other"] += i.get("count", 1)
    type_rows = "".join(f'<tr><td>{_e(cat.replace("_", " ").capitalize())}</td><td class="num">{n}</td></tr>'
                        for cat, n in sorted(by_type.items()))
    if books:
        type_rows += f'<tr><td>Books</td><td class="num">{totals["book_count"]}</td></tr>'
    def item_row(i: dict) -> str:
        count = f" ×{i['count']}" if i.get("count", 1) > 1 else ""
        detected = '<div class="small">detected, not confirmed</div>' if (i.get("source") or "").startswith("object detector") else ""
        appraise = '<div class="flag">needs appraiser</div>' if i.get("status") == "needs_appraisal" else ""
        category = (i.get("category") or "other").replace("_", " ").capitalize()
        return (f"<tr><td>{_e(i['id'])}</td><td>{_e(category)}</td><td>{_e(i['description'])}{count}{detected}{appraise}</td>"
                f"<td>{_e(i.get('brand_model') or '')}</td><td>{_e(i.get('where') or '')}</td><td>{_e('; '.join(i.get('notes') or []))}</td>"
                f"<td>{_frame_link(i.get('frame_ref'))}</td></tr>")

    item_rows = "".join(item_row(i) for i in items)

    def book_row(b: dict) -> str:
        copies = f'<div class="small"><strong>{b["copies"]} copies</strong> (confirmed by the policyholder); prices are per copy</div>' \
            if b.get("copies", 1) > 1 else ""
        flag = '<div class="flag">needs appraiser</div>' if b["status"] == "needs_appraisal" else ""
        return (f"<tr><td>{_e(b['id'])}</td><td><strong>{_e(b['title'])}</strong><div class=\"small\">{_e(b.get('author') or '')}</div>"
                f"{copies}{flag}</td><td>{_e(b.get('shelf') or '')}</td><td>{_size_cell(b)}</td>"
                f"<td>{_price_cell(b.get('replacement_cost'), currency)}</td><td>{_price_cell(b.get('used_value'), currency)}</td>"
                f"<td class=\"num\">{'' if b.get('id_confidence') is None else b['id_confidence']}</td>"
                f"<td>{_frame_link(b.get('frame_ref'))}</td></tr>")

    left_out_rows = "".join(
        f"<tr><td><strong>{_e(b['title'])}</strong><div class=\"small\">{_e(b.get('author') or ('not identified in the sweep' if b.get('identified') is False else ''))}</div></td>"
        f"<td>{_e((b.get('excluded') or {}).get('reason') or 'policyholder asked to leave it out')}</td>"
        f"<td>{_e(str((b.get('excluded') or {}).get('at', ''))[:16].replace('T', ' '))}</td></tr>" for b in left_out)
    review_rows = "".join(f"<tr><td>{_e(r['ref_id'])}</td><td>{_e(r['reason'])}</td></tr>" for r in review)

    m_state = (packet.get("measurement") or {}).get("state")
    m_note = {"running": "Spine measurement was still running when this was made; reload for the final sizes.",
              "done": "", "off": "Spine measurement is off."}.get(m_state, "No spine measurement.")
    card_frames = (packet.get("measurement") or {}).get("card_frames", [])
    scale_note = (f"Scale from a bank card (85.60 × 53.98 mm) seen in {len(card_frames)} frame(s); other frames use books already "
                  "measured against it. Sizes marked estimate come from Depth Pro's depth and focal-length estimate."
                  if card_frames else "No bank card was seen, so any sizes are estimates from Depth Pro, not measurements.")
    priced_n = sum(b["copies"] for b in books if b["status"] == "identified" and b["replacement_cost"]["amount"] is not None)
    cards = "".join(f'<div class="card"><div class="label">{_e(k)}</div><div class="value">{v}</div></div>' for k, v in [
        ("Objects logged", str(len(items))),
        ("Books in the inventory", str(totals["book_count"])),
        ("Books left out", str(totals.get("books_left_out_by_policyholder", 0))),
        ("Books replacement cost", _money(totals["books_replacement_cost"], currency) if totals["books_replacement_cost"] is not None else "none priced"),
        ("Books used value", _money(totals["books_used_value"], currency) if totals["books_used_value"] is not None else "—"),
        ("Review items", str(len(review))),
    ])
    generated = time.strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy inventory report</title>
<meta name="report-state" content="{"in-progress" if packet.get("pending") else "complete"}">
<style>
  :root {{ --ink:#1d2433; --muted:#667085; --line:#e4e7ec; --bg:#ffffff; --soft:#f6f7f9; --accent:#2f5bd3; --flag:#b42318; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width:1180px; margin:0 auto; padding:28px 16px 48px; }}
  h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 10px; }}
  .meta {{ color:var(--muted); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:10px; margin-top:20px; }}
  .card {{ background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .label {{ color:var(--muted); font-size:13px; }} .value {{ font-size:20px; font-weight:600; margin-top:2px; }}
  .table-wrap {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; }}
  th, td {{ text-align:left; vertical-align:top; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ font-size:13px; color:var(--muted); font-weight:600; background:var(--soft); }}
  td.num, th.num {{ text-align:right; }}
  .small {{ font-size:12px; color:var(--muted); margin-top:2px; }} .muted {{ color:var(--muted); }}
  .flag {{ color:var(--flag); font-size:12px; font-weight:600; }}
  .ebook-tag {{ display:inline-block; padding:1px 6px; border:1px solid var(--flag); border-radius:4px; color:var(--flag);
               font-size:11px; font-weight:700; letter-spacing:0.04em; }}
  .note {{ background:var(--soft); border-left:3px solid var(--accent); padding:8px 12px; margin:10px 0; font-size:14px; }}
  a {{ color:var(--accent); }}
  @media print {{ .note {{ break-inside:avoid; }} }}
</style></head>
<body><main>
  <h1>Policy inventory report</h1>
  <div class="meta">Adding items to the policy
    · {_e(sweep.get("country") or "country not set")}{(" (" + _e(currency) + ")") if currency else ""}
    · captured {_e(str(sweep.get("captured_at") or "")[:16].replace("T", " "))}
    {(" · " + _e(sweep["duration_s"]) + " s") if sweep.get("duration_s") is not None else ""}
    · report {generated} · sweep {_e(sweep.get("id") or "")} · data: claim_packet.json</div>
  {f'<div class="note">Still working on: {_e(", ".join(packet["pending"]))}. This report updates by itself.</div>' if packet.get("pending") else ""}
  <div class="cards">{cards}</div>
  {'<div class="note">The book worker did not answer; books are missing from this report.</div>' if not packet.get("book_worker_reached") else ""}

  <h2>Objects by type</h2>
  <div class="table-wrap"><table><thead><tr><th>Type</th><th class="num">Count</th></tr></thead>
  <tbody>{type_rows or '<tr><td colspan="2" class="muted">Nothing logged.</td></tr>'}</tbody></table></div>

  <h2>Objects</h2>
  <div class="note">Listed as logged during the call. Not priced in this version; art needs an appraiser unless it is a print.</div>
  <div class="table-wrap"><table><thead><tr><th>Id</th><th>Type</th><th>Description</th><th>Brand / model</th><th>Where</th><th>Notes</th><th>Evidence</th></tr></thead>
  <tbody>{item_rows or '<tr><td colspan="7" class="muted">Nothing logged.</td></tr>'}</tbody></table></div>

  <h2>Books ({totals["book_count"]})</h2>
  <div class="note">{priced_n} of {totals["book_count"]} priced; books without a price or needing an appraiser are not in the total.
    Replacement cost is a new copy (marked EBOOK PRICE where the country has no eBay site and the eBook edition was
    priced instead); used value is the median asking price of used copies. {_e(m_note)}
    {len(packet.get("unread_spine_readings", []))} unread spine reading(s) are not counted (see the review queue).
</div>
  <div class="table-wrap"><table><thead><tr><th>Id</th><th>Book</th><th>Shelf</th><th>Spine (height × thickness)</th><th>Replacement cost</th>
    <th>Used value</th><th class="num">Id conf.</th><th>Evidence</th></tr></thead>
  <tbody>{"".join(book_row(b) for b in books) or '<tr><td colspan="8" class="muted">No books identified.</td></tr>'}</tbody></table></div>

  <h2>Left out at the policyholder's request</h2>
  <div class="table-wrap"><table><thead><tr><th>Book</th><th>Reason</th><th>When</th></tr></thead>
  <tbody>{left_out_rows or '<tr><td colspan="3" class="muted">None.</td></tr>'}</tbody></table></div>

  <h2>Room</h2>
  {_room_section(packet["room"])}

  <h2>Review queue ({len(review)})</h2>
  <div class="table-wrap"><table><thead><tr><th>Line</th><th>Reason</th></tr></thead>
  <tbody>{review_rows or '<tr><td colspan="2" class="muted">Nothing to review.</td></tr>'}</tbody></table></div>

  <h2>How these figures were made</h2>
  <div class="note">Totals are added up in code from the lines in claim_packet.json; books left out, unpriced books, books
    needing an appraiser and unpriced objects are not in them. {_e(scale_note)} Shelf run:
    {_e(totals["shelf_run_m"]) if totals.get("shelf_run_m") is not None else "—"} m (upright identified books only).</div>
{_usage_section(usage)}
</main></body></html>
"""
