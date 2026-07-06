"""
Name and school normalization helpers shared by every stage of the pipeline.

Nothing here talks to disk -- it is pure string/DataFrame transformation so
it can be unit-tested and reused from ingest.py, crosswalk.py and
features.py alike.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from pipeline.school_aliases import (
    CFBD_SCHOOL_ALIASES,
    COMBINE_SCHOOL_ALIASES,
    PFF_SCHOOL_ALIASES,
)

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# First-name nickname/diminutive unification, applied only to the first
# token of a normalized name. Without this, "Mitch Trubisky" (CFBD) vs
# "Mitchell Trubisky" (combine), or "Nate Stanley" (combine) vs "Nathan
# Stanley" (CFBD), fail to match even though schools/seasons line up
# perfectly. Deliberately conservative: only maps well-known diminutives to
# one canonical spelling, applied identically regardless of which spelling
# a given source happens to use.
_NICKNAMES = {
    "mitch": "mitchell",
    "nate": "nathan",
    "nathaniel": "nathan",
    "cam": "cameron",
    "alex": "alexander",
    "nick": "nicholas",
    "zach": "zachary",
    "zack": "zachary",
    "matt": "matthew",
    "chris": "christopher",
    "mike": "michael",
    "mikey": "michael",
    "dan": "daniel",
    "danny": "daniel",
    "will": "william",
    "billy": "william",
    "bill": "william",
    "joe": "joseph",
    "joey": "joseph",
    "tom": "thomas",
    "tommy": "thomas",
    "jake": "jacob",
    "sam": "samuel",
    "sammy": "samuel",
    "ben": "benjamin",
    "benny": "benjamin",
    "tony": "anthony",
    "steve": "steven",
    "stephen": "steven",
    "dave": "david",
    "rob": "robert",
    "bobby": "robert",
    "bob": "robert",
    "ed": "edward",
    "eddie": "edward",
    "greg": "gregory",
    "andy": "andrew",
    "drew": "andrew",
    "jim": "james",
    "jimmy": "james",
    "ken": "kenneth",
    "kenny": "kenneth",
}

_SCHOOL_ALIAS_TABLES = {
    "pff": PFF_SCHOOL_ALIASES,
    "combine": COMBINE_SCHOOL_ALIASES,
    "cfbd": CFBD_SCHOOL_ALIASES,
}

# Populated at import time by get_canonical_school() whenever it has to fall
# back to a best-effort guess instead of a table hit. build.py reads this to
# surface "please add an alias" warnings in the match report.
UNMAPPED_SCHOOLS: set[tuple[str, str]] = set()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_name(name) -> str | None:
    """Lowercase, punctuation-free, suffix-free name for entity matching.

    "C.J. Beathard" / "CJ Beathard" -> "cj beathard"
    "Jared Goff Jr." -> "jared goff"
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = strip_accents(str(name)).lower()
    text = text.replace(".", "").replace("'", "").replace("`", "")
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    while tokens and tokens[-1] in _SUFFIXES:
        tokens.pop()
    if tokens:
        tokens[0] = _NICKNAMES.get(tokens[0], tokens[0])
    return " ".join(tokens)


def slugify(*parts) -> str:
    """Build a URL/id-safe slug out of arbitrary string parts."""
    cleaned = []
    for part in parts:
        if part is None or (isinstance(part, float) and pd.isna(part)):
            continue
        text = strip_accents(str(part)).lower()
        text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
        if text:
            cleaned.append(text)
    return "-".join(cleaned)


def _fallback_school_normalize(raw: str) -> str:
    """Best-effort canonicalization for a school we have no alias entry for.

    Title-cases the raw string and expands a couple of extremely common
    abbreviations. This is deliberately conservative -- ambiguous short
    codes (single-letter directionals like "S"/"N"/"E"/"W") are NOT expanded
    here because they are ambiguous (e.g. PFF's "S" means "San" in
    "S JOSE ST" but "South" in "S CAROLINA"); those must be added to
    school_aliases.py explicitly instead of guessed.
    """
    text = raw.strip()
    text = re.sub(r"\bSt\.?$", "State", text)
    return " ".join(w.capitalize() if not w.isupper() or len(w) > 4 else w for w in text.split())


def get_canonical_school(raw, source: str) -> str | None:
    """Map a raw school string from `source` ("pff"|"combine"|"cfbd") to the
    canonical school string used to join across sources.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw = str(raw).strip()
    table = _SCHOOL_ALIAS_TABLES.get(source, {})
    if raw in table:
        return table[raw]
    if source == "cfbd":
        # CFBD's own team names ARE the canonical basis (see module
        # docstring) -- pass through unchanged rather than running the
        # PFF/combine-style title-case fallback, which would mangle
        # already-correct acronyms like "UCLA" -> "Ucla". Only the
        # small handful of genuine spelling divergences (accents, etc.)
        # need CFBD_SCHOOL_ALIASES at all, so no warning here.
        return strip_accents(raw)
    canonical = _fallback_school_normalize(raw)
    UNMAPPED_SCHOOLS.add((source, raw))
    return canonical


def add_normalized_columns(
    df: pd.DataFrame,
    name_col: str,
    school_col: str,
    source: str,
    name_out: str = "normalized_name",
    school_out: str = "canonical_school",
) -> pd.DataFrame:
    """Return a copy of df with normalized-name and canonical-school columns."""
    out = df.copy()
    out[name_out] = out[name_col].map(normalize_name)
    out[school_out] = out[school_col].map(lambda v: get_canonical_school(v, source))
    return out
