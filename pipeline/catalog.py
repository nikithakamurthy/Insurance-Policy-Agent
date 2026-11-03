"""Book agent, step 2: resolve spine text to a real work.

Search goes to Google Books, which corrects misspelt words ('workng backwrds', 'AFTERTHE'), and
to Open Library when Google Books fails (an outage, quota, or no key). Google Books returns
editions; they are grouped into one result per title and author. Each candidate text is
searched and every returned work is scored on how much
of its title and author was actually read on the spine. A match is accepted only if most of
the title's significant words were read and either an author surname was read too or the
title is distinctive (three or more significant words). Everything else stays unidentified;
a blank is better than a confidently wrong title.

id_confidence is that match score, not a model's opinion. The edition is left empty: a spine
rarely shows which edition it is.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata

from rapidfuzz import fuzz

from .common import get_cached

logger = logging.getLogger(__name__)

SEARCH_URL = "https://openlibrary.org/search.json"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
FIELDS = "key,title,subtitle,author_name,first_publish_year,publisher,isbn,language,edition_count"
STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "your", "how", "why", "what", "who", "are", "our", "you",
    "der", "das", "und", "ein", "eine", "les", "une", "von", "zum", "zur", "edition", "press",
    "books", "book", "publishing", "verlag", "media", "second", "third",
}
PUBLISHER_NOISE = {"oreilly", "penguin", "portfolio", "random", "house", "harper", "collins", "pan", "macmillan",
                   "vintage", "picador", "manning", "wiley", "pearson", "rupa", "jaico"}
MATCH = 85      # rapidfuzz ratio for two words to count as the same word (allows OCR slips)


def tokens(text: str) -> list[str]:
    """Lower-case ASCII words. Apostrophes are removed first: O'REILLY -> oreilly, Engineer's -> engineers."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", re.sub(r"['`]", "", text))


def significant(words: list[str]) -> list[str]:
    return [w for w in words if (len(w) >= 3 or w.isdigit()) and w not in STOPWORDS]


def _found(word: str, pool: list[str]) -> bool:
    return any(fuzz.ratio(word, p) >= MATCH for p in pool)


def _with_joins(words: list[str]) -> list[str]:
    """Words plus each neighbouring pair run together, for OCR that drops a space ('DEADHAND')."""
    return words + [a + b for a, b in zip(words, words[1:])]


def _author_read(authors: list[str], words: list[str], *, full_title: bool) -> bool:
    """Was the author read? The surname always counts. A first name ("Peter", "JULIA C...") counts
    only when the whole title was read too: spines often show the full name and OCR cuts the
    surname, but a first name alone is too common to support a partial title."""
    for author in authors:
        parts = [t for t in tokens(author) if len(t) >= 3]
        if not parts:
            continue
        if _found(parts[-1], words):
            return True
        if full_title and any(_found(t, words) or any(w.startswith(t) for w in words) for t in parts[:-1] if len(t) >= 4):
            return True
    return False


def score_match(candidate_text: str, doc: dict) -> dict:
    cand_all = tokens(candidate_text)
    cand = significant(cand_all)
    title_all = tokens(doc.get("title", ""))
    title_words = significant(title_all)
    if not cand or not title_words:
        return {"score": 0.0, "title_coverage": 0.0, "author_hit": False, "accepted": False}
    pool = _with_joins(cand_all)
    joined_title = [a + b for a, b in zip(title_all, title_all[1:])]
    title_cov = sum(_found(w, pool) or any(w in j and _found(j, pool) for j in joined_title) for w in title_words) / len(title_words)
    author_hit = _author_read(doc.get("author_name", []), cand, full_title=title_cov >= 0.999)
    explained_pool = _with_joins(title_all) + significant(tokens(" ".join(doc.get("author_name", [])))) + \
        significant(tokens(doc.get("subtitle", ""))) + list(PUBLISHER_NOISE)
    explained = sum(_found(w, explained_pool) for w in cand) / len(cand)
    score = 0.6 * title_cov + 0.25 * author_hit + 0.15 * explained
    # Title and author both read: extra words are most likely a neighbouring spine. Without the
    # author, the reading must be mostly this title and the title must be distinctive.
    strong = title_cov >= 0.8 and author_hit and explained >= 0.3
    distinctive = title_cov >= 0.8 and len(title_words) >= 3 and explained >= 0.5
    accepted = (strong or distinctive) and score >= 0.7
    return {"score": round(score, 3), "title_coverage": round(title_cov, 3), "author_hit": author_hit,
            "explained_fraction": round(explained, 3), "accepted": accepted}


QUERY_FILLER = {"a", "an", "the", "of", "to", "and", "in", "on"}


def query_text(text: str, max_words: int = 8) -> str:
    """Search words: every real word read, minus filler, single letters and publisher names.

    Words such as "how", "not" or "die" are kept: "how to not die alone" reduced to "not alone"
    finds the wrong book. Scoring still ignores them (see significant)."""
    words = [w for w in tokens(text) if len(w) >= 2 and w not in QUERY_FILLER and w not in PUBLISHER_NOISE]
    return " ".join(words[:max_words])


MAX_FALLBACK_QUERIES = 3


def _open_library(query: str, limit: int) -> tuple[list[dict], str]:
    body, retrieved_at = get_cached(SEARCH_URL, {"q": query, "fields": FIELDS, "limit": limit}, cache=False)
    docs = body.get("docs", [])
    for d in docs:
        d.update(source="Open Library", url=f"https://openlibrary.org{d['key']}")
    return docs, retrieved_at


def _google_books(query: str, limit: int, key: str) -> tuple[list[dict], str]:
    """Google Books search, editions grouped into one result per main title and first author.

    first_publish_year stays empty: an edition's date is not when the work first appeared (a 2003
    reprint of Dickens). published_year_seen is the earliest edition date in the results."""
    body, retrieved_at = get_cached(GOOGLE_BOOKS_URL, {"q": query, "maxResults": min(2 * limit, 40), "printType": "books",
                                                       "key": key}, cache=False, attempts=2)
    works: dict[str, dict] = {}
    for volume in body.get("items", []):
        info = volume.get("volumeInfo", {})
        title, authors = info.get("title", ""), info.get("authors", [])
        if not title:
            continue
        surname = (tokens(authors[0]) or [""])[-1] if authors else ""
        work_key = "gb:" + "-".join(title_words(title)) + ("|" + surname if surname else "")
        year = int(info["publishedDate"][:4]) if re.match(r"\d{4}", info.get("publishedDate", "")) else None
        isbns = [i["identifier"] for i in info.get("industryIdentifiers", []) if i["type"].startswith("ISBN")]
        work = works.get(work_key)
        if work is None:
            if len(works) >= limit:
                continue
            works[work_key] = {"key": work_key, "title": title, "subtitle": info.get("subtitle", ""), "author_name": authors,
                               "first_publish_year": None, "published_year_seen": year, "isbn": isbns,
                               "language": [info["language"]] if info.get("language") else [], "edition_count": 1,
                               "source": "Google Books", "url": info.get("canonicalVolumeLink") or info.get("infoLink")}
        else:
            work["edition_count"] += 1
            work["isbn"] += isbns
            if year and (work["published_year_seen"] is None or year < work["published_year_seen"]):
                work["published_year_seen"] = year
    return list(works.values()), retrieved_at


def _search(query: str, limit: int, memo: dict | None = None, *, open_library: bool = False) -> tuple[list[dict], str]:
    # No disk cache: results are fetched in each sweep. `memo` is one sweep's in-memory record,
    # so a spine read again later in the same sweep is not searched twice.
    memo_key = (query, limit, open_library)
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    import urllib.error

    from .pricing import google_books_paused, load_books_key, note_google_books_error   # pricing imports this module

    key = None if open_library or google_books_paused() else load_books_key()
    result = None
    if key:
        try:
            result = _google_books(query, limit, key)
        except Exception as exc:   # 503s and quota errors happen; Open Library answers instead
            if isinstance(exc, urllib.error.HTTPError):
                note_google_books_error(exc, exc.read().decode("utf-8", "replace"))
            logger.info("Google Books search failed (%s); using Open Library for %r", type(exc).__name__, query)
    if result is None:
        result = _open_library(query, limit)
    if memo is not None:
        memo[memo_key] = result
    return result


def first_publish_year(title: str, author: str) -> int | None:
    """When the work first appeared, from Open Library (Google Books only knows edition dates).
    Used by the antiquarian check; None if Open Library has no such work or cannot be reached."""
    surname = (tokens(author.split(",")[0]) or [""])[-1] if author else ""
    params = {"title": title, "fields": "title,author_name,first_publish_year", "limit": 5}
    if surname:
        params["author"] = surname
    try:
        body, _ = get_cached(SEARCH_URL, params, cache=False, attempts=1)
    except Exception:
        return None
    want = significant(tokens(title))
    years = [d["first_publish_year"] for d in body.get("docs", []) if d.get("first_publish_year")
             and want and all(_found(w, tokens(d.get("title", ""))) for w in want)]
    return min(years) if years else None


ARTICLES = {"the", "a", "an"}


EDITION_WORDS = {"edition", "revised", "updated", "expanded", "new", "anniversary", "first", "second", "third", "fourth",
                 "fifth", "sixth", "1st", "2nd", "3rd", "4th", "5th", "6th"}


def title_words(title: str) -> list[str]:
    """Title tokens without a leading article or edition wording: "The Lean Startup" and "Lean
    Startup", or "The Intelligent Investor" and "The Intelligent Investor Third Edition", are one
    book (both pairs came back as separate catalogue entries in live sweeps and were counted twice)."""
    words = tokens(title)
    while len(words) > 1 and words[0] in ARTICLES:
        words = words[1:]
    if "edition" in words:
        words = [w for w in words if w not in EDITION_WORDS] or words
    return words


def work_identity(title: str, author: str) -> str:
    """Same book across catalogues, whose keys differ: main title and first author's surname.
    A leading series name ("Relate - After The Affair", a Google Books edition), a leading article
    and a subtitle after ':' are left out."""
    main = re.sub(r"^[^-:]{1,30}\s[-\u2013]\s+", "", title).split(":")[0]
    surname = (tokens(author.split(",")[0]) or [""])[-1] if author else ""
    return " ".join(title_words(main)) + "|" + surname


TITLE_NOISE = EDITION_WORDS | {"rev", "ed", "edn", "by", "illustrated", "paperback", "hardcover", "and"}


def _surnames(author: str) -> set[str]:
    names = re.split(r",|\band\b|&", author or "")
    return {tokens(n)[-1] for n in names if tokens(n) and len(tokens(n)[-1]) >= 3}


def same_work(title_a: str, author_a: str, title_b: str, author_b: str) -> bool:
    """Two catalogue entries for one book. Same author (a shared surname), and one title is the other
    plus only noise: edition words ("Rev. Ed", "Third Edition") or "by" and the author's name. Seen in
    one live sweep: "Spark: The Definitive Guide" / "Spark The Definitive Guide", "The Intelligent
    Investor" / "..., Rev. Ed", "How to Win Friends and Influence People" / "... by Dale Carnegie".
    A sequel adds real words ("Dune" / "Dune Messiah") and stays separate."""
    if work_identity(title_a, author_a) == work_identity(title_b, author_b):
        return True
    shared = _surnames(author_a) & _surnames(author_b)
    if not shared:
        return False
    author_words = set(tokens(author_a)) | set(tokens(author_b))

    def words(title: str) -> list[str]:
        out = tokens(re.sub(r"^[^-:]{1,30}\s[-\u2013]\s+", "", title))
        while len(out) > 1 and out[0] in ARTICLES:
            out = out[1:]
        return out

    a, b = words(title_a), words(title_b)
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if not short or long_[:len(short)] != short:
        return False
    return all(w in TITLE_NOISE or w in author_words for w in long_[len(short):])


_speller = None
_speller_lock = threading.Lock()


def correct_spelling(query: str) -> str:
    """OCR slips corrected with SymSpell's English word list: 'workng backwrds' -> 'working
    backwards', run-together words split ('afterthe' -> 'after the', 'lusecond' -> 'a second').
    Offline, about 1 ms. It also 'corrects' names it does not know (Huyen -> human), so the
    corrected text is only tried first as a query; scoring always uses the reading as read."""
    global _speller
    with _speller_lock:   # search runs in several threads; load the dictionary once
        if _speller is None:
            from importlib.resources import files

            from symspellpy import SymSpell
            speller = SymSpell(max_dictionary_edit_distance=2)
            speller.load_dictionary(str(files("symspellpy") / "frequency_dictionary_en_82_765.txt"), 0, 1)
            speller.load_bigram_dictionary(str(files("symspellpy") / "frequency_bigramdictionary_en_243_342.txt"), 0, 2)
            _speller = speller
    suggestions = _speller.lookup_compound(query, max_edit_distance=2, ignore_non_words=True)
    return suggestions[0].term if suggestions else query


def search(text: str, limit: int = 5, fallback: bool = True, memo: dict | None = None) -> tuple[list[dict], str, str]:
    """Search the catalogue. The query is spell-corrected first (correct_spelling), then searched
    as read if the corrected text finds nothing. Both catalogues require every query word to
    match (Google Books also corrects misspellings), so one junk word (such as 'OREILL', a publisher name cut off
    at the edge of the spine) returns nothing. When a
    query finds nothing that matches the reading, retry without words that look like a publisher, then (for a long reading,
    likely two spines run together) only its last three words, then leaving out one word at a
    time, starting from the end, where a word cut off at the spine edge is likeliest.
    At most MAX_FALLBACK_QUERIES retries. When no Google Books result clearly matches the reading,
    Open Library's results are added. Scoring still uses the full reading, so a looser query
    cannot loosen the match.
    """
    def matched(found: list[dict]) -> bool:
        return any(score_match(text, d)["accepted"] for d in found)

    query = used = query_text(text)
    docs = []
    fixed = query_text(correct_spelling(query))
    if fixed != query and len(fixed.split()) >= 2:
        docs, retrieved_at = _search(fixed, limit, memo)
        used = fixed
    if not matched(docs):   # no correction, or it changed a name ("huyen" -> "human")
        raw_docs, raw_retrieved_at = _search(query, limit, memo)
        if raw_docs or not docs:
            docs, retrieved_at, used = raw_docs, raw_retrieved_at, query
    if not matched(docs) and fallback:
        words = query.split()
        kept = [w for w in words if not _found(w, list(PUBLISHER_NOISE))]
        variants = [kept] if kept != words else []
        if len(kept) >= 5:
            variants.append(kept[-3:])   # neighbouring spines read as one line: the last spine's words
        if 3 <= len(kept) <= 7:
            variants += [kept[:i] + kept[i + 1:] for i in reversed(range(len(kept)))]
        for variant in [v for v in variants if len(v) >= 2][:MAX_FALLBACK_QUERIES]:
            found, found_at = _search(" ".join(variant), limit, memo)
            if matched(found) or (found and not docs):   # keep the first results, unless a retry matches
                docs, retrieved_at, used = found, found_at, " ".join(variant)
            if matched(found):
                break
    if all(d.get("source") == "Google Books" for d in docs) and not matched(docs):
        # Google Books lists editions, whose titles carry series prefixes ("Relate - After The
        # Affair"); Open Library's work titles are cleaner. One query there, its results added after.
        # Open Library does not correct spelling, so it gets the corrected text when there is one.
        extra, ol_retrieved_at = _search(fixed if len(fixed.split()) >= 2 else query, limit, memo, open_library=True)
        seen = {(d["title"].lower(), tuple(d.get("author_name", [])[:1])) for d in docs}
        docs = docs + [d for d in extra if (d["title"].lower(), tuple(d.get("author_name", [])[:1])) not in seen]
        retrieved_at = retrieved_at if not extra else ol_retrieved_at
    return docs, used, retrieved_at


def identify(candidate: dict) -> dict | None:
    """Best accepted Open Library work for one candidate text, or None."""
    if len(significant(tokens(candidate["text"]))) < 2:
        return None
    docs, query, retrieved_at = search(candidate["text"])
    scored = [(score_match(candidate["text"], d), d) for d in docs]
    accepted = [(s, d) for s, d in scored if s["accepted"]]
    if not accepted:
        return None
    s, d = max(accepted, key=lambda pair: pair[0]["score"])
    return {"work_key": d["key"], "title": d.get("title", ""), "subtitle": d.get("subtitle", ""),
            "author": ", ".join(d.get("author_name", [])[:3]), "first_publish_year": d.get("first_publish_year"),
            "edition_count": d.get("edition_count"), "match": s,
            "catalogue": {"source": d.get("source", "Open Library"), "url": d.get("url"),
                          "query": query, "retrieved_at": retrieved_at}}


VOWELS = set("aeiouy")


def plausible(text: str) -> bool:
    """Cheap filter for mirrored or garbled readings before spending a catalogue query.

    Needs at least two significant words that look like words: letters only and a normal share
    of vowels (mirrored capitals such as 'WVHVAS' or 'NIWVIN38' fail).
    """
    good = 0
    for word in significant(tokens(text)):
        if word.isalpha() and 0.2 <= sum(c in VOWELS for c in word) / len(word) <= 0.75:
            good += 1
    return good >= 2


def _cluster_readings(candidates: list[dict], similarity: int = 90) -> dict[str, str]:
    """Map each reading to a representative reading of (nearly) the same text.

    The same spine is read in many consecutive frames with small OCR differences; one
    catalogue query per cluster is enough. The highest-confidence reading represents it.
    """
    reps: list[str] = []
    mapping: dict[str, str] = {}
    for cand in sorted(candidates, key=lambda c: -c["ocr_confidence"]):
        key = " ".join(tokens(cand["text"]))
        if key in mapping:
            continue
        rep = next((r for r in reps if fuzz.token_set_ratio(key, r) >= similarity and abs(len(key) - len(r)) <= 0.3 * len(r)), None)
        if rep is None:
            reps.append(key)
            rep = key
        mapping[key] = rep
    return mapping


def identify_books(spine_text: dict, workers: int = 4) -> dict:
    """Identify every candidate, then merge sightings of the same work across frames."""
    from concurrent.futures import ThreadPoolExecutor

    from .spine_ocr import spine_candidates

    books: dict[str, dict] = {}
    unidentified: dict[str, dict] = {}
    frames = spine_candidates(spine_text)
    all_candidates = [c for cands in frames.values() for c in cands]
    usable = [c for c in all_candidates if plausible(c["text"])]
    mapping = _cluster_readings(usable)
    reps = sorted(set(mapping.values()))
    failed: list[str] = []

    def lookup(rep: str):
        try:
            return rep, identify({"text": rep})
        except OSError as exc:   # network failure after retries: record it, do not crash
            failed.append(f"{rep}: {type(exc).__name__}")
            return rep, None

    with ThreadPoolExecutor(workers) as pool:
        seen_queries = dict(pool.map(lookup, reps))
    for frame_ref, candidates in frames.items():
        for cand in candidates:
            key = " ".join(tokens(cand["text"]))
            if key not in mapping:
                continue
            match = seen_queries[mapping[key]]
            if match is None:
                # Up to two neighbouring strips: a tilted cover splits title and author into
                # separate strips. These readings only go to review as text; they are never
                # counted or priced, so an occasional two-spine merge costs little.
                if cand.get("strips", 1) <= 2 and cand["ocr_confidence"] >= 0.8 and len(significant(tokens(cand["text"]))) >= 2:
                    unidentified.setdefault(key, {"text": cand["text"], "frame_refs": set(), "ocr_confidence": cand["ocr_confidence"]})["frame_refs"].add(frame_ref)
                continue
            confidence = round(match["match"]["score"] * min(1.0, cand["ocr_confidence"] + 0.1), 3)
            book = books.setdefault(match["work_key"], {**match, "frame_refs": set(), "evidence": [], "id_confidence": 0.0})
            book["frame_refs"].add(frame_ref)
            if confidence > book["id_confidence"]:
                book.update(id_confidence=confidence, frame_ref=frame_ref, box_norm=cand["box_norm"],
                            ocr_text=cand["text"], orientation=cand["orientation"], match=match["match"])
            if len(book["evidence"]) < 5 and cand["text"] not in book["evidence"]:
                book["evidence"].append(cand["text"])
    # A readable fragment of a book identified elsewhere ('and Bill Carr') is not a new book.
    known = [_with_joins(tokens(f"{b['title']} {b['author']}")) for b in books.values()]

    def fragment_of_known(text: str) -> bool:
        words = [w for w in significant(tokens(text)) if not _found(w, list(PUBLISHER_NOISE))]
        return not words or any(sum(_found(w, k) for w in words) >= 0.6 * len(words) for k in known)

    unread = [{**u, "frame_refs": sorted(u["frame_refs"])} for u in unidentified.values() if not fragment_of_known(u["text"])]
    out = sorted(books.values(), key=lambda b: -b["id_confidence"])
    for i, b in enumerate(out, 1):
        b["id"] = f"book-{i:03d}"
        b["frame_refs"] = sorted(b["frame_refs"])
    return {"identified": out, "unidentified_readings": unread, "queries": len(seen_queries),
            "readings": len(all_candidates), "plausible_readings": len(usable), "lookup_failures": failed,
            "method": "docTR spine text -> Open Library search -> title/author coverage scoring"}
