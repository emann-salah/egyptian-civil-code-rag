"""Validation tests for the extracted corpus.

Run after every extraction: pytest tests/test_corpus.py -v
"""

import json
from collections import Counter
from pathlib import Path

import pytest

CORPUS_PATH = Path("data/processed/articles.json")

REPEALED_RANGES = [range(54, 81), range(389, 418)]
ANNOUNCEMENT_ARTICLES = {54}

# Known extraction gaps - documented as acceptable (<1% of corpus)
KNOWN_MISSING = {798, 799, 800, 802, 1017, 1018, 1019, 1020, 1022, 1023, 1024, 1025, 1026}

MAX_SANE_ARTICLE_CHARS = 6000


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS_PATH.exists():
        pytest.skip(f"{CORPUS_PATH} not found - run the extraction script first")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_corpus_not_empty(corpus):
    assert len(corpus) > 0, "corpus is empty - extraction produced nothing"


def test_every_record_has_nonempty_arabic_text(corpus):
    failures = [
        r["article_number"] for r in corpus
        if (not r.get("text_ar", "").strip() or r["text_ar"] == "[EXTRACTION_FAILED]")
        and r["article_number"] not in KNOWN_MISSING
    ]
    assert not failures, f"NEW unexplained empty/failed articles: {failures}"


def test_no_record_exceeds_sane_length(corpus):
    too_long = [
        r["article_number"] for r in corpus
        if len(r.get("text_ar", "")) > MAX_SANE_ARTICLE_CHARS
    ]
    assert not too_long, f"articles suspiciously long (likely merged articles): {too_long}"


def test_no_duplicate_article_numbers(corpus):
    numbers = [r["article_number"] for r in corpus]
    counts = Counter(numbers)
    duplicates = {n: c for n, c in counts.items() if c > 1}
    assert not duplicates, f"duplicate article numbers found: {duplicates}"


def test_article_numbers_are_positive_integers(corpus):
    for r in corpus:
        assert isinstance(r["article_number"], int) and r["article_number"] > 0


def test_repealed_ranges_are_present_and_flagged(corpus):
    """Articles 55-80 and 390-417 must exist as individual records, each
    flagged is_repealed=True. The announcement articles themselves (54, 389)
    must exist too, but stay is_repealed=False since they're the source
    article explaining the repeal, not repealed articles themselves."""
    by_number = {r["article_number"]: r for r in corpus}
    for rng in REPEALED_RANGES:
        missing = [n for n in rng if n not in by_number]
        assert not missing, f"repealed-range articles missing entirely: {missing}"

        to_check_flagged = [n for n in rng if n not in ANNOUNCEMENT_ARTICLES]
        not_flagged = [n for n in to_check_flagged if not by_number[n]["is_repealed"]]
        assert not not_flagged, f"articles in repealed range but is_repealed=False: {not_flagged}"

        for n in ANNOUNCEMENT_ARTICLES & set(rng):
            assert not by_number[n]["is_repealed"], (
                f"announcement article {n} should stay is_repealed=False"
            )


def test_known_gaps_are_the_only_gaps(corpus):
    """Any number missing from the corpus that ISN'T in a known repealed
    range or the documented extraction-failure list is a NEW, unexplained
    gap that needs investigation."""
    present = {r["article_number"] for r in corpus}
    max_number = max(present)
    all_expected = set(range(1, max_number + 1))

    missing = all_expected - present
    unexplained = missing - KNOWN_MISSING
    assert not unexplained, f"new unexplained gaps found: {sorted(unexplained)}"