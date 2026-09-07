"""The Printer — a diagnostic, not a scheduled job.

Approval testing separates capturing output from **printing** it. The Printer
turns a response into something comparable, and its job is to scrub what is
non-deterministic so that what remains can be compared exactly.

**It outlived the tier that called it.** `drift/tier3.py` ran this on a schedule
and was cut: it sampled a stochastic process once and could not distinguish a
changed model from a different output, reporting Gemini's truncation shape as
drift when a confirming call showed it unchanged. The Printer itself was never
the problem — the sampling was.

It is now reached for during a fixture refresh, where the question is "what
changed in the shape" and both sides are already in hand. See REFRESH.md.

That inversion is the whole design. An earlier draft specified which fields to
COMPARE; this compares everything and names only what to DROP. The asymmetry
matters: **a gap in a scrub list produces noise, a gap in a compare list produces
a miss**, and noise is the failure you notice.

Structure comparison is GenSON — it infers a JSON Schema by observing instances
and merges across many, which gives three things for free: a removed field drops
out of `required`, a changed type shows as a changed `type`, and a node seeded
with `enum: []` records every value it ever saw. Verified against real fixtures
before this was written rather than trusted from the documentation.

What GenSON does not give, and what the rest of this module adds: **count
categories**. A schema records that a number is an integer, not that it went from
many to zero. Three searches becoming four is not drift; three becoming zero is,
and only the category boundary carries signal.
"""

from __future__ import annotations

import re
from typing import Any

try:
    from genson import SchemaBuilder
except ImportError:  # pragma: no cover
    raise SystemExit("drift needs GenSON:  pip install genson")

__all__ = ["print_response", "shape_of", "compare", "SCRUB_KEYS", "SCRUB_PATTERNS"]


# ═══════════════════════════════════════════════════════════════════════════════
# What gets scrubbed
#
# Everything NOT listed here is compared exactly. Adding a key here is a decision
# that its variation carries no signal — make it deliberately, because the cost
# of scrubbing something meaningful is a silent miss.
# ═══════════════════════════════════════════════════════════════════════════════

SCRUB_KEYS = {
    # ── model output. Different every call by construction. ──
    "text", "thinking", "summary", "summary_text", "content_text",
    "cited_text", "quote", "answer_text", "thought_text",

    # ── retrieval. Changes with the index, not with the API. ──
    "url", "title", "query", "queries", "encrypted_content", "page_age",
    "search_suggestions", "redirect_host", "domain",

    # ── identity and opaque state. ──
    "id", "request_id", "tool_use_id", "call_id", "signature",
    "encrypted_index", "container",

    # ── time and cost. Real but not structural. ──
    "created", "created_at", "started_at", "collected_at", "latency",
}

# Offsets are scrubbed as VALUES but their presence is not — a citation gaining
# or losing `start_index` is exactly the change a refresh diff should catch, while
# number itself changes with every answer.
SCRUB_KEYS |= {"start_index", "end_index", "start", "end"}

SCRUB_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),          # ISO dates
    re.compile(r"^(msg|resp|srvtoolu|rs|ws|req|v1)_"),   # provider id prefixes
]

# Numeric fields where the VALUE is noise but the CATEGORY is signal. Compared as
# zero / one / few / many rather than as numbers.
COUNT_KEYS = {
    "input_tokens", "output_tokens", "total_tokens", "total_input_tokens",
    "total_output_tokens", "total_thought_tokens", "thinking_tokens",
    "reasoning_tokens", "raw_prompt_token", "cached_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens",
    "web_search_requests", "web_fetch_requests", "num_requests", "count",
    "total_tool_use_tokens", "total_cached_tokens", "cache_write_tokens",
}

# Values that ARE the structure. Never scrubbed, compared exactly, and collected
# as enums so a new one shows up rather than being averaged away.
STRUCTURAL_KEYS = {
    "type", "status", "stop_reason", "role", "name", "offset_unit",
    "modality", "service_tier", "caller", "search_type", "code",
}

# Seeded with `enum: []` so GenSON collects values instead of inferring a type.
# `caller` is deliberately absent: it is a scalar on two providers and a DICT on
# Anthropic — `{"type": "direct"}` — and seeding a dict-valued node as an enum
# raises. It is still compared as structure, just not collected as an enum.
#
# The exclusion is a symptom rather than the fix. A key can be scalar on every
# provider today and a dict tomorrow, so `shape_of` also falls back to an
# unseeded builder rather than failing.
# Keys where the VALUE carries signal and is therefore collected and compared,
# rather than merely typed. Structural keys plus count categories.
#
# Named for what it does now. It was `SEEDED` while values were collected by
# seeding GenSON with `enum: []`, and kept that name for a while after the seed
# was removed — a constant describing a mechanism that no longer existed.
#
# `caller` is a scalar on two providers and a DICT on Anthropic; the walk skips
# non-scalars rather than excluding the key, so it needs no special case.
COLLECTED = STRUCTURAL_KEYS | COUNT_KEYS


def _category(n: Any) -> str:
    """A count as a category. The boundary is what carries signal.

    Three searches becoming four is not drift. Three becoming zero is — and
    comparing the numbers would report the first as loudly as the second, which
    is how a detector becomes noise and stops being read.
    """
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return "none"
    if n == 0:
        return "zero"
    if n == 1:
        return "one"
    if n <= 10:
        return "few"
    if n <= 1000:
        return "many"
    return "very_many"


def print_response(body: Any, _key: str | None = None) -> Any:
    """Scrub a response into its comparable form. THE PRINTER.

    Recurses the whole structure. Scrubbed values become a marker rather than
    disappearing, so the KEY's presence is still compared even though its
    content is not — losing a field and having a field change are different
    events and must not look alike.
    """
    if isinstance(body, dict):
        return {k: print_response(v, k) for k, v in body.items()}

    if isinstance(body, list):
        # Lists are compared by the shape of their members, not their length —
        # length is a count and belongs to the category comparison. GenSON
        # merges members into one item schema, which is the behaviour wanted.
        return [print_response(v, _key) for v in body]

    if _key in STRUCTURAL_KEYS and not isinstance(body, (dict, list)):
        return body
    if _key in COUNT_KEYS:
        return _category(body)
    if _key in SCRUB_KEYS:
        return f"<{_key}>"
    if isinstance(body, str):
        for pattern in SCRUB_PATTERNS:
            if pattern.match(body):
                return f"<{_key or 'value'}>"
        # Long free text is model output under a key nobody listed. Scrubbed on
        # length rather than name, because the scrub list will always be
        # incomplete and an unlisted 6,000-character answer is pure noise.
        if len(body) > 200:
            return f"<long:{_key or 'value'}>"
    return body


def shape_of(bodies: list) -> dict:
    """The comparable shape of a set of responses. Two parts, deliberately.

    **Structure** comes from GenSON: field paths, types, and `required` relaxed
    to fields present in ALL instances — which is the right semantics for a
    provider that omits zero-valued fields.

    **Values** come from our own walk, collecting every value seen at each path
    for structural keys and count categories.

    They are separate because seeding GenSON to collect values requires naming
    every path in advance, and a seed deep enough to cover three providers'
    nesting is combinatorial — the first attempt exhausted memory at depth five.
    Walking once and collecting as we go is O(n) and has no depth limit.
    """
    printed = [print_response(b) for b in bodies]

    builder = SchemaBuilder()
    for body in printed:
        builder.add_object(body)

    values: dict[str, set] = {}
    for body in printed:
        _collect(body, "", values)

    return {"structure": builder.to_schema(),
            "values": {k: sorted(v, key=str) for k, v in values.items()}}


def _collect(body: Any, path: str, out: dict[str, set]) -> None:
    """Every value seen at each path, for the keys where the VALUE is signal.

    Array indices are collapsed to `[]` so the tenth block and the first are the
    same path — otherwise a response with more blocks would look like a response
    with new paths, which is the noise this whole module exists to avoid.
    """
    if isinstance(body, dict):
        for key, value in body.items():
            here = f"{path}.{key}" if path else key
            if key in COLLECTED and not isinstance(value, (dict, list)):
                out.setdefault(here, set()).add(value)
            else:
                _collect(value, here, out)
    elif isinstance(body, list):
        for item in body:
            _collect(item, f"{path}[]", out)


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare(before: dict, after: dict) -> list[str]:
    """Differences between two shapes, as readable lines.

    A DIFF, not a verdict. This project has produced four functions that computed
    a conclusion and reported it while contradicting evidence sat in the same
    output; a drift detector that says "looks fine" would be the fifth.

    Empty list means no structural difference — which the caller reports as such
    rather than as approval.
    """
    diffs = _compare_schema(before.get("structure", {}), after.get("structure", {}))

    b_vals, a_vals = before.get("values", {}), after.get("values", {})
    for path in sorted(set(b_vals) | set(a_vals)):
        gone = sorted(set(b_vals.get(path, [])) - set(a_vals.get(path, [])), key=str)
        new_ = sorted(set(a_vals.get(path, [])) - set(b_vals.get(path, [])), key=str)
        if gone:
            diffs.append(f"  VALUES-   {path}: no longer seen: {gone}")
        if new_:
            diffs.append(f"  VALUES+   {path}: newly seen: {new_}")
    return diffs


def _compare_schema(before: dict, after: dict, path: str = "") -> list[str]:
    """Paths, types and presence. Values are compared separately by `compare`."""
    diffs: list[str] = []

    b_props = before.get("properties", {})
    a_props = after.get("properties", {})

    for key in sorted(set(b_props) | set(a_props)):
        here = f"{path}.{key}" if path else key
        if key not in a_props:
            diffs.append(f"  REMOVED   {here}")
            continue
        if key not in b_props:
            diffs.append(f"  ADDED     {here}")
            continue
        diffs += _compare_node(b_props[key], a_props[key], here)

    b_req = set(before.get("required", []))
    a_req = set(after.get("required", []))
    for key in sorted(b_req - a_req):
        if key in a_props:
            diffs.append(f"  OPTIONAL  {path}.{key} was always present, now sometimes absent"
                         .replace("  .", "  "))
    for key in sorted(a_req - b_req):
        if key in b_props:
            diffs.append(f"  REQUIRED  {path}.{key} was sometimes absent, now always present"
                         .replace("  .", "  "))

    if "items" in before or "items" in after:
        diffs += _compare_node(before.get("items", {}), after.get("items", {}),
                               f"{path}[]")
    return diffs


def _compare_node(before: dict, after: dict, path: str) -> list[str]:
    diffs: list[str] = []

    b_type, a_type = before.get("type"), after.get("type")
    if b_type != a_type and (b_type or a_type):
        diffs.append(f"  TYPE      {path}: {b_type} -> {a_type}")

    b_enum, a_enum = before.get("enum"), after.get("enum")
    if b_enum is not None or a_enum is not None:
        gone = sorted(set(b_enum or []) - set(a_enum or []))
        new = sorted(set(a_enum or []) - set(b_enum or []))
        if gone:
            diffs.append(f"  VALUES-   {path}: no longer seen: {gone}")
        if new:
            diffs.append(f"  VALUES+   {path}: newly seen: {new}")

    if "properties" in before or "properties" in after:
        diffs += _compare_schema(before, after, path)
    elif "items" in before or "items" in after:
        diffs += _compare_node(before.get("items", {}), after.get("items", {}),
                              f"{path}[]")
    return diffs
