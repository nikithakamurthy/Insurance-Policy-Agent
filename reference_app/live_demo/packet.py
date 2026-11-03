"""claim_packet.json: the sweep's structured output, in the brief's output contract.

Built in code from the items the voice agent logged and the book worker's results (books,
prices, measurements). The rules:
- A field the system does not know stays empty (None). Nothing is filled without evidence.
- Every frame_ref is a path, relative to the packet file, of a saved frame.
- Totals are added up here from the lines; no language model writes a number.
- Anything not confident goes to review_queue with the reason.

Fields beyond the contract: books[].copies, books[].evidence, excluded_books, unread_spine_readings,
items[].where/notes, and per-price detail (country, condition, conversion, edition matched).
"""

from __future__ import annotations

import time
from typing import Any

PACKET_VERSION = 1
LOW_ID_CONFIDENCE = 0.8


def _price(price: dict | None, keys: tuple[str, ...]) -> dict:
    """A price in the contract's shape; empty amount when there is no sourced price."""
    price = price or {}
    out = {k: price.get(k) for k in keys}
    out["amount"] = price.get("amount")
    for extra in ("currency", "country", "condition_assumed", "converted", "original", "fx", "edition_matched", "reason",
                  "listings_considered"):
        if price.get(extra) is not None and extra not in out:
            out[extra] = price[extra]
    return out


def _frame(ref: str | None) -> str | None:
    return f"frames/{ref}.jpg" if ref else None


# Words in an object the agent logged that mean the detector's category is already covered.
COVERED_BY = {"sofa": ("sofa", "couch", "settee"), "chair": ("chair", "armchair", "stool"), "table": ("table",),
              "desk": ("desk",), "lamp": ("lamp", "light"), "framed_art": ("painting", "picture", "print", "art", "frame", "poster"),
              "mirror": ("mirror",), "rug": ("rug", "carpet"), "electronics": ("tv", "television", "monitor", "laptop", "speaker"),
              "appliance": ("machine", "kettle", "microwave", "appliance"), "decor": ("plant", "vase", "clock", "decor"),
              "cabinet": ("cabinet", "cupboard", "sideboard", "dresser"), "bookshelf": ("bookcase", "bookshelf", "shelf", "shelving")}
DETECTOR_LABELS = {"framed_art": "painting or framed picture", "electronics": "electronic device", "decor": "decor item",
                   "bookshelf": "bookcase"}


def detected_items(detected: dict, logged: list[dict]) -> list[dict]:
    """Object types the detector confirmed during the sweep that the agent did not log. The
    count is the most seen in one frame: a lower bound when the camera never had all in view."""
    said = " ".join(f"{i.get('category', '')} {i.get('description', '')}" for i in logged).lower()
    out = []
    for n, (category, seen) in enumerate(sorted(detected.items()), 1):
        if any(word in said for word in COVERED_BY.get(category, (category,))):
            continue
        name = DETECTOR_LABELS.get(category, category.replace("_", " "))
        out.append({"id": f"D{n:02d}", "category": category, "description": name, "count": seen.get("count", 1),
                    "brand_model": None, "where": None, "notes": [], "frame_ref": seen.get("frame_ref"),
                    "source": "object detector (Grounding DINO), not confirmed by the policyholder",
                    "detector_score": seen.get("score")})
    return out


ROOM_MIN_WALLS = 2
ROOM_MIN_CONFIDENCE = 0.5


def room_problems(room: dict) -> list[str]:
    """Why a room measurement is not good enough to report; empty when it is. Below these bars the
    numbers are mostly guesses (one sweep gave a floor area with no wall found at all), so the
    packet leaves the room empty rather than report them."""
    problems = []
    if room.get("scale_estimated"):
        problems.append("scale only estimated by Depth Pro (no bank card in the reconstructed frames)")
    if not room.get("floor_plane_found"):
        problems.append("floor not found")
    if (room.get("walls_found") or 0) < ROOM_MIN_WALLS:
        problems.append(f"{room.get('walls_found') or 0} wall(s) found, at least {ROOM_MIN_WALLS} needed")
    if (room.get("confidence") or 0) < ROOM_MIN_CONFIDENCE:
        problems.append(f"confidence {room.get('confidence')} below {ROOM_MIN_CONFIDENCE}")
    return problems


def build_packet(meta: dict, items: list[dict], books: dict | None, detected: dict | None = None) -> dict:
    """meta: sweep_id, captured_at, device, duration_s, country, currency, purpose, purpose_details.
    items: the voice agent's logged objects. books: the worker's GET /books/{sweep}/report, or None.
    detected: object types the live detector confirmed (category -> count, score, frame_ref)."""
    books = books or {}
    measurement = (books.get("measurement") or {}).get("report") or {}
    sizes = {(m["title"], m.get("author", "")): m for m in measurement.get("books", [])}
    shelf_by_work = books.get("shelf_by_work", {})
    review: list[dict] = []

    claim_books, excluded = [], []
    for line in books.get("books", []):
        size = sizes.get((line["title"], line.get("author", ""))) or {}
        replacement = line.get("replacement_cost") or {}
        book = {
            "id": line["id"], "shelf": shelf_by_work.get(line["work_key"]), "position": None,
            "frame_ref": _frame(line.get("frame_ref")), "frame_refs": [_frame(r) for r in line.get("frame_refs", [])],
            "status": "needs_appraisal" if line.get("status") == "needs_appraisal" else "identified",
            "title": line["title"], "author": line.get("author") or None,
            "edition": None, "isbn": None,           # a spine rarely shows the edition; the priced edition is in replacement_cost
            "spine_height_cm": size.get("spine_height_cm"), "spine_thickness_cm": size.get("spine_thickness_cm"),
            "size_method": size.get("scale_method"), "size_estimated": size.get("estimated"),
            "id_confidence": line.get("id_confidence"),
            "copies": line.get("copies", 1), "copies_confirmed": line.get("copies_confirmed", False),
            "replacement_cost": _price(replacement, ("source", "url", "retrieved_at", "converted")),
            "used_value": _price(line.get("used_value"), ("source", "url", "retrieved_at", "condition_assumed")),
        }
        if line.get("excluded"):
            excluded.append({**book, "excluded": line["excluded"]})
            continue
        claim_books.append(book)
        if book["status"] == "needs_appraisal":
            review.append({"ref_id": book["id"], "reason": f"needs a human appraiser: {line.get('appraisal_reason', '')}".strip(": ")})
        elif replacement.get("amount") is None:
            review.append({"ref_id": book["id"], "reason": f"no price: {replacement.get('reason') or 'price lookup failed'}; excluded from the total"})
        elif replacement.get("converted"):
            review.append({"ref_id": book["id"], "reason": f"price converted from {replacement.get('original', {}).get('currency')} "
                                                           f"({replacement.get('country')})"})
        used = line.get("used_value") or {}
        if used.get("amount") is not None and (used.get("listings_considered") or 0) < 3:
            review.append({"ref_id": book["id"], "reason": f"used value from only {used.get('listings_considered')} eBay listing(s)"})
        if (book["id_confidence"] or 0) < LOW_ID_CONFIDENCE:
            review.append({"ref_id": book["id"], "reason": f"identification confidence {book['id_confidence']}"})
        if book["spine_height_cm"] is None and book["spine_thickness_cm"] is None:
            review.append({"ref_id": book["id"], "reason": "spine not measured"})
        elif book["size_estimated"]:
            review.append({"ref_id": book["id"], "reason": "spine size estimated from depth, no bank card in view"})

    for copy in books.get("possible_copies", []):
        match = next((b for b in claim_books if b["title"] == copy["title"]), None)
        review.append({"ref_id": match["id"] if match else copy["title"],
                       "reason": f"read on {copy['spines_in_one_frame']} spines in one frame; counted once until the policyholder confirms copies"})
    for suggestion in books.get("suggested", []):
        review.append({"ref_id": f"reading: {suggestion['reading'][:60]}",
                       "reason": f"possible match {suggestion['title']} not supported by the words read; not counted"})
    unread = books.get("unread_readings", [])
    if unread:
        review.append({"ref_id": "unread spines", "reason": f"{len(unread)} spine reading(s) matched no catalogue entry; "
                                                             "not counted as books (see unread_spine_readings)"})

    claim_items = []
    for item in list(items) + detected_items(detected or {}, items):
        art = item.get("category") in {"art", "framed_art"}
        said_print = any("print" in n.lower() for n in item.get("notes", []))
        status = "needs_appraisal" if art and not said_print else None      # items are not priced in this version
        claim_items.append({
            "id": item["id"], "category": item.get("category"), "description": item["description"],
            "brand_model": item.get("brand_model") or None, "frame_ref": item.get("frame_ref") or None,
            "dimensions_cm": {"w": None, "h": None, "d": None}, "status": status,
            "replacement_cost": {"low": None, "high": None, "source": None, "url": None, "retrieved_at": None},
            "confidence": item.get("detector_score"), "where": item.get("where") or None, "notes": item.get("notes", []),
            "count": item.get("count", 1), "source": item.get("source", "logged by the voice agent"),
        })
        if item.get("source"):
            review.append({"ref_id": item["id"], "reason": f"detected, not confirmed: {item['count']} seen at most in one frame; "
                                                           "ask the policyholder to confirm what and how many"})
        review.append({"ref_id": item["id"], "reason": "art: needs a human appraiser unless the policyholder says it is a print"
                       if status == "needs_appraisal" else "not priced: item pricing is not implemented yet"})
        if not item.get("frame_ref"):
            review.append({"ref_id": item["id"], "reason": "no camera frame was saved when it was logged"})

    room_status = books.get("room") or {}
    measured = room_status.get("room") if room_status.get("state") == "done" else None
    empty_room = {"length_m": None, "width_m": None, "height_m": None, "floor_area_m2": None, "wall_area_m2": None,
                  "shelved_wall_area_m2": None, "scale_method": None, "confidence": None}
    problems = room_problems(measured) if measured else []
    if measured and not problems:
        room = {"length_m": measured.get("length_m"), "width_m": measured.get("width_m"), "height_m": measured.get("height_m"),
                "floor_area_m2": measured.get("floor_area_m2"), "wall_area_m2": measured.get("wall_area_m2"),
                "shelved_wall_area_m2": measured.get("shelved_wall_area_m2"), "scale_method": measured.get("scale_method"),
                "confidence": measured.get("confidence"),
                "floor_area_ft2": measured.get("floor_area_ft2"), "wall_area_ft2": measured.get("wall_area_ft2"),
                "shape": measured.get("shape"), "shape_method": measured.get("shape_method"),
                "wall_area_method": measured.get("wall_area_method"),
                "frames_used": [f"frames/{r}.jpg" if r.startswith("f") else f"room_frames/{r}.jpg" for r in measured.get("frames_used", [])],
                "floorplan": "floorplan.png"}
        if measured.get("ceiling_plane_found") is False:
            # Without a ceiling the height is only the top of what the camera saw; the wall area depends on it.
            room["height_m"] = room["wall_area_m2"] = room["wall_area_ft2"] = None
            review.append({"ref_id": "room", "reason": "ceiling not found: height and wall area left empty; "
                                                       "film a strip of ceiling in the room turn"})
    else:
        room = empty_room
        if measured:
            review.append({"ref_id": "room", "reason": "room measurement not reliable, left empty: " + "; ".join(problems)})
        else:
            error = room_status.get("error") or ""
            review.append({"ref_id": "room", "reason": {
                "running": "room measurement still running; reload the report",
                "queued": "room measurement still running; reload the report",
                "failed": ("room not measured: the room turn was skipped or too short" if "room frames" in error
                           else f"room measurement failed: {error}")}.get(room_status.get("state"), "room not measured")})

    # Books the policyholder counted at a place beyond those identified there: counted, not identified.
    place_counts = []
    for place in books.get("place_counts", []):
        unidentified = max(0, place["stated"] - place["identified"]) if place.get("stated") is not None else 0
        place_counts.append({**place, "unidentified": unidentified})
        if unidentified:
            review.append({"ref_id": place["place"], "reason": f"policyholder counted {place['stated']} books here; "
                                                              f"{place['identified']} identified, {unidentified} counted but not identified"})
    unidentified_total = sum(p["unidentified"] for p in place_counts)

    # Totals, from the lines above only.
    def copies(b: dict) -> int:
        return b["copies"]

    priced = [b for b in claim_books if b["status"] == "identified" and b["replacement_cost"]["amount"] is not None]
    used = [b for b in claim_books if b["status"] == "identified" and b["used_value"]["amount"] is not None]
    item_low = [i["replacement_cost"]["low"] for i in claim_items if i["replacement_cost"]["low"] is not None]
    item_high = [i["replacement_cost"]["high"] for i in claim_items if i["replacement_cost"]["high"] is not None]
    upright = [b for b in claim_books if b["spine_thickness_cm"]]
    totals = {
        "book_count": sum(copies(b) for b in claim_books) + unidentified_total,
        "books_identified": sum(copies(b) for b in claim_books),
        "books_unidentified": unidentified_total,     # from the policyholder's count per place; unread spines alone are not counted
        "shelf_run_m": measurement.get("shelf_run_m") if upright else None,
        "books_replacement_cost": round(sum(b["replacement_cost"]["amount"] * copies(b) for b in priced), 2) if priced else None,
        "books_used_value": round(sum(b["used_value"]["amount"] * copies(b) for b in used), 2) if used else None,
        "items_replacement_cost_low": round(sum(item_low), 2) if item_low else None,
        "items_replacement_cost_high": round(sum(item_high), 2) if item_high else None,
        "excluded_from_totals": (sum(copies(b) for b in claim_books) - sum(copies(b) for b in priced))
                                + sum(i["count"] for i in claim_items if i["replacement_cost"]["low"] is None),
        "books_left_out_by_policyholder": len(excluded) + len(books.get("pending_exclusions", [])),
        "currency": books.get("currency") or meta.get("currency") or None,
    }

    pending = []
    if books.get("still_reading"):
        pending.append("reading book spines")
    if (books.get("measurement") or {}).get("state") == "running":
        pending.append("measuring spines")
    if (books.get("room") or {}).get("state") in {"queued", "running"}:
        pending.append("measuring the room")

    return {
        "packet_version": PACKET_VERSION,
        "pending": pending,          # still being worked on in the background; empty when the packet is final
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sweep": {"id": meta.get("sweep_id"), "captured_at": meta.get("captured_at"), "device": meta.get("device") or None,
                  "duration_s": meta.get("duration_s"), "country": meta.get("country") or None,
                  "currency": books.get("currency") or meta.get("currency") or None,
                  "purpose": meta.get("purpose") or None, "purpose_details": meta.get("purpose_details") or None},
        "room": room,
        "books": claim_books,
        "items": claim_items,
        "totals": totals,
        "review_queue": review,
        "place_counts": place_counts,
        "excluded_books": excluded + [{"title": p["title"], "excluded": p, "identified": False}
                                      for p in books.get("pending_exclusions", [])],
        "unread_spine_readings": [{"text": u["text"], "frame_refs": [_frame(r) for r in u["frame_refs"]],
                                   "ocr_confidence": u.get("ocr_confidence")} for u in unread],
        "measurement": {"method": measurement.get("method"), "card_frames": [_frame(c["frame_ref"]) for c in measurement.get("card_frames", [])],
                        "state": (books.get("measurement") or {}).get("state")},
        "book_worker_reached": bool(books),
    }
