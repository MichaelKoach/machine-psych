"""Reading a corpus.

Six functions, and one of them — `normalize` — is the concrete test of the whole
design premise. If a Tier 3 finding cannot be reached from a normalised corpus
without re-collecting, then normalising at analysis has failed and this is a
gateway library with extra steps.

The other five exist because of a discipline learned the hard way: **extract
before you read.** Every qualitative claim this project has made from a single
read has died at n=5 or n=7. `mentions` and `verdict` pull patterns across every
record mechanically; `read` is for understanding what the numbers mean, after.

`index` and `read` also carry a caveat the numbers cannot: on one provider the
answer is INFERRED from block position rather than stated by the API, so
`answer_text` is a different kind of claim depending on where it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .capabilities import caps_for
from .corpus import capability_note

__all__ = ["index", "read", "normalize", "mentions", "verdict",
           "format_stability", "NormalizeReport"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tiers
# ═══════════════════════════════════════════════════════════════════════════════

# Same meaning on every provider.
TIER1 = [
    "record_id", "investigation", "provider", "model", "served_model",
    "study", "probe", "probe_hash", "path", "rep", "turn", "conversation",
    "n_turns", "condition", "status", "conversation_status", "error",
    "attempts", "latency", "prompt", "prompt_hash", "answer_text",
    "answer_chars", "answer_extraction", "config_passed", "config_resolved",
    "intent_unmet",
]

# Analogous, with the mapping asserted rather than measured. Each carries a
# caveat that a caller has to know before comparing across providers.
TIER2 = {
    "in_tok_billed": "billing policy differs — one provider does not bill "
                     "retrieved content as input, so this is not context volume",
    "in_tok_processed": "comparable, but read from a different field per "
                        "provider; on one it differs from billed by 100x",
    # The text for this one is DERIVED from the capability table rather than
    # written here — see `_out_tok_caveat`. Hardcoding "two providers count
    # thinking inside it" would make this a second source of truth for a fact the
    # table already holds, and the two would diverge the first time a provider
    # changed its accounting.
    "out_tok_reported": None,
    "thinking_tok": "comparable in magnitude, not in meaning: effort levels are "
                    "an asserted mapping between provider vocabularies",
    "n_queries": "comparable — from each provider's own usage figure, never "
                 "from counting blocks or items",
    "n_citations": "comparable in count; the citation MECHANISM differs three "
                   "ways, so compare quotes rather than spans",
}

# Present only where the provider can. Each names the capability that decides.
TIER3 = {
    "thought_text": "readable_reasoning",
    "n_thought_steps": "readable_reasoning",
    "n_sources_retrieved": "retrieval_set",
    "grounded": "search_conditional",
    "n_answer_blocks": None,
    "n_process_blocks": None,
}


@dataclass
class NormalizeReport:
    """What `normalize` set aside, and how to get it back.

    Returned rather than printed. A data function with a side effect is invisible
    when called inside another function and unusable when the caller wants the
    list programmatically.
    """
    dropped: dict[str, str] = field(default_factory=dict)
    caveats: dict[str, str] = field(default_factory=dict)
    by_provider: dict[str, list[str]] = field(default_factory=dict)
    still_reachable: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [f"NormalizeReport: {len(self.dropped)} columns set aside"]
        for col, why in sorted(self.dropped.items()):
            who = ", ".join(self.by_provider.get(col, [])) or "no provider here"
            lines.append(f"  {col:<22} {why}")
            lines.append(f"  {'':<22} available on: {who}")
        if self.caveats:
            lines.append("")
            lines.append("  kept, with caveats:")
            for col, why in sorted(self.caveats.items()):
                lines.append(f"    {col:<20} {why}")
        if self.still_reachable:
            lines.append("")
            lines.append("  nothing was lost — still reachable from the corpus:")
            for fn in self.still_reachable:
                lines.append(f"    {fn}")
        return "\n".join(lines)


def normalize(corpus: pd.DataFrame) -> tuple[pd.DataFrame, NormalizeReport]:
    """Tier 1 + Tier 2 as a view, plus a report of what was set aside.

    **This is the test of the premise, not a convenience.** The design commitment
    is normalise at ANALYSIS, never at collection — raw responses go to disk
    unchanged and a narrow view is a function over them rather than a constraint
    on them. Gateway libraries do the opposite and the structure is permanently
    gone; here it must still be reachable after normalising, or the commitment is
    empty.

    So the report lists not just what was dropped but how to get it back, and a
    test asserts that the side tables still return rows after this is called.
    """
    keep = [c for c in TIER1 if c in corpus.columns]
    keep += [c for c in TIER2 if c in corpus.columns]
    view = corpus[keep].copy()

    report = NormalizeReport(caveats={
        c: (why if why is not None else _out_tok_caveat(corpus))
        for c, why in TIER2.items() if c in corpus.columns})

    for col, capability in TIER3.items():
        if col not in corpus.columns:
            continue
        if capability:
            can = capability_note(corpus, capability)
            report.by_provider[col] = sorted(
                m for m, yes in can.items() if yes)
            report.dropped[col] = (
                f"Tier 3 — present only where `{capability}` is true")
        else:
            report.by_provider[col] = sorted(corpus.model.dropna().unique())
            report.dropped[col] = "Tier 3 — provider-shaped response structure"

    report.still_reachable = [
        "citations(corpus)", "sources(corpus)", "queries(corpus)",
        "thoughts(corpus)", "units(corpus)",
        "capability_note(corpus, ...)",
    ]
    return view, report


def _out_tok_caveat(corpus: pd.DataFrame) -> str:
    """Why `out_tok_reported` cannot be compared, derived from the corpus.

    Reads `thinking_in_output_tokens` per model rather than restating it. The
    split is currently two-to-one, and writing that down would be correct today
    and a lie the first time a provider changed its accounting — which one of
    them did to a different parameter during this build.
    """
    models = sorted(corpus.model.dropna().unique()) if len(corpus) else []
    inside = [m for m in models if caps_for(m).thinking_in_output_tokens]
    outside = [m for m in models if not caps_for(m).thinking_in_output_tokens]
    if not inside or not outside:
        return ("comparable here — every model in this corpus uses the same "
                "thinking-token convention. Check again before adding a provider")
    return (f"NOT comparable — {len(inside)} model(s) count thinking INSIDE this "
            f"figure ({', '.join(inside)}) and {len(outside)} keep it separate "
            f"({', '.join(outside)}). Use answer_chars")


# ═══════════════════════════════════════════════════════════════════════════════
# Looking
# ═══════════════════════════════════════════════════════════════════════════════

def index(corpus: pd.DataFrame, **filters) -> pd.DataFrame:
    """A compact table of what is in the corpus.

    Deliberately shows `answer_chars` rather than `out_tok`: the token count is
    not comparable across providers, and putting a non-comparable number in the
    default view is how it ends up in a comparison.
    """
    sel = _filter(corpus, filters)
    cols = ["record_id", "provider", "probe", "condition", "rep", "turn",
            "status", "answer_chars", "n_queries", "n_citations", "attempts"]
    out = sel[[c for c in cols if c in sel.columns]].copy()
    return out.reset_index(drop=True)


def read(corpus: pd.DataFrame, record_id: int | None = None, chars: int | None = None,
         full: bool = False, sources_shown: int = 8, thoughts_shown: int = 0,
         return_text: bool = False, **filters):
    """Print records in full, for understanding rather than counting.

    Use AFTER extracting, never instead of it. Every qualitative claim this
    project has made from a single read has died at a larger n — the failure mode
    is not that reading is wrong but that **what you read is a sample whose
    selection you did not control**, and a function that prints the first two
    records is a sampling decision disguised as a display choice.
    """
    sel = corpus if record_id is None else corpus[corpus.record_id == record_id]
    sel = _filter(sel, filters)
    if not len(sel):
        text = "no records match"
        return text if return_text else print(text)

    out: list[str] = []
    for conversation, group in sel.groupby("conversation", sort=False):
        group = group.sort_values("turn")
        first = group.iloc[0]
        out.append("=" * 84)
        header = f"{first.provider} · {first.study} / {conversation} · {first.condition}"
        out.append(header)
        if first.conversation_status != "ok":
            out.append(f"      {first.conversation_status.upper()}")

        for r in group.itertuples(index=False):
            meta = []
            # NaN is a float and is truthy, so `or ""` does not catch it and
            # slicing raises. Type-check before formatting.
            meta.append(f"{int(r.answer_chars)} chars"
                        if isinstance(r.answer_chars, (int, float))
                        and not pd.isna(r.answer_chars) else "no answer")
            if _has(r, "thinking_tok"):
                meta.append(f"{int(r.thinking_tok)} think")
            if _has(r, "n_queries"):
                meta.append(f"{int(r.n_queries)} queries")
            if _has(r, "n_citations"):
                meta.append(f"{int(r.n_citations)} cites")
            if getattr(r, "attempts", 1) > 1:
                meta.append(f"{r.attempts} attempts")
            if r.status != "ok":
                meta.append(str(r.status).upper()
                            + (f": {r.error}" if r.error else ""))

            out.append("")
            turn_label = f"TURN {r.turn}   " if r.n_turns > 1 else ""
            out.append(f"▸ {turn_label}[{r.record_id}] · " + " · ".join(meta))
            if r.answer_extraction == "heuristic":
                out.append("  (answer INFERRED from block position, not stated "
                           "by the API)")
            out.append("")
            out.append(_indent(f"> {r.prompt}"))
            out.append("")

            text = r.answer_text if isinstance(r.answer_text, str) else None
            if text is not None and chars is not None:
                text = text[:chars]
            out.append(_indent(text) if text else "  (no answer)")

            if thoughts_shown:
                # Public accessor, not `_side` — an earlier version bypassed it
                # and silently missed the enrichment those functions add. A
                # private helper reached for convenience is how two views of the
                # same table drift apart.
                from .corpus import thoughts as _thoughts
                th = _thoughts(corpus, record_id=r.record_id)
                if len(th):
                    out.append("")
                    out.append("  REASONING")
                    for t in th.itertuples(index=False):
                        out.append(_indent(str(t.text)[:thoughts_shown], "    "))

            if sources_shown:
                from .corpus import sources as _sources
                src = _sources(corpus, record_id=r.record_id)
                if len(src):
                    n_cited = int(src.cited.fillna(False).sum())
                    # `retrieved` is None where a provider does not report what
                    # it saw — dropna, never fillna, or the line claims it saw
                    # nothing.
                    known = src.retrieved.dropna()
                    extra = (f", {int(known.sum())} retrieved" if len(known)
                             else "")
                    out.append("")
                    out.append(f"  SOURCES ({n_cited} cited{extra})")
                    for s in src.head(sources_shown).itertuples(index=False):
                        mark = "*" if s.cited else " "
                        label = getattr(s, "domain", None) or s.url
                        out.append(f"    {mark} {label}")
                    if len(src) > sources_shown:
                        out.append(f"      … {len(src) - sources_shown} more")
        out.append("")

    text = "\n".join(out)
    return text if return_text else print(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Extracting
# ═══════════════════════════════════════════════════════════════════════════════

def mentions(corpus: pd.DataFrame, vocabulary: list[str],
             whole_word: bool = True, column: str = "answer_text",
             **filters) -> pd.DataFrame:
    """Count vocabulary terms across every record. Extract before you read.

    `whole_word` defaults True because of a real failure: searching for a brand
    name matched every occurrence of the same string inside longer words, and the
    inflated count looked plausible enough to survive review.
    """
    sel = _filter(corpus, filters)
    if not vocabulary:
        return pd.DataFrame()

    rows = []
    for r in sel.itertuples(index=False):
        text = getattr(r, column, None)
        if not isinstance(text, str):
            continue
        for term in vocabulary:
            pattern = (rf"\b{re.escape(term)}\b" if whole_word
                       else re.escape(term))
            n = len(re.findall(pattern, text, re.I))
            if n:
                rows.append({"record_id": r.record_id, "provider": r.provider,
                             "probe": r.probe, "condition": r.condition,
                             "rep": r.rep, "turn": r.turn,
                             "term": term, "count": n})
    return pd.DataFrame(rows)


def verdict(corpus: pd.DataFrame, pattern: str, column: str = "answer_text",
            chars: int = 160, flags=re.I, **filters) -> pd.DataFrame:
    """Where a pattern appears in each answer, and how far in.

    `position` is the fraction of the answer at which the match occurs. Verdicts
    have measured early on one provider — 4.4% to 9% in — and that is a property
    of that provider's answers, not a general one: another produces a format
    stability near zero, where a verdict may not be locatable at all.

    The pattern is supplied by the caller because verdict phrasing is
    probe-specific, and a built-in one would fit the probe it was written for and
    silently miss every other.
    """
    sel = _filter(corpus, filters)
    rows = []
    for r in sel.itertuples(index=False):
        text = getattr(r, column, None)
        if not isinstance(text, str) or not text:
            continue
        m = re.search(pattern, text, flags)
        rows.append({
            "record_id": r.record_id, "provider": r.provider, "probe": r.probe,
            "condition": r.condition, "rep": r.rep,
            "found": bool(m),
            "position": round(m.start() / len(text), 3) if m else None,
            # False, not None, on a non-match — `~df.in_heading` raises on None,
            # and a filter that raises is worse than one that is merely wrong.
            "in_heading": bool(m and _in_heading(text, m.start())),
            "snippet": text[max(0, m.start() - 20):m.start() + chars] if m else None,
        })
    return pd.DataFrame(rows)


def format_stability(corpus: pd.DataFrame, **filters) -> pd.DataFrame:
    """Header-set overlap across repetitions of the same probe.

    **A PROXY, not a measurement.** It is used to ask "will regex extraction work
    on this probe", and that relationship has never been validated — a probe
    could have stable prose and unstable headers, or the reverse.

    It has been useful: 0.9 on recognition probes against 0.0 on a projective
    one, and 0.00–0.18 across every probe on one provider, which correctly
    predicted that deductive extraction would not work there. But the docstring
    says proxy because a number that behaves well is the easiest kind to start
    trusting for the wrong reason.
    """
    sel = _filter(corpus, filters)
    rows = []
    for (provider, probe, condition), group in sel.groupby(
            ["provider", "probe", "condition"], sort=False):
        sets = []
        for text in group[group.turn == 0].answer_text:
            if isinstance(text, str):
                sets.append(frozenset(re.findall(r"^#{1,4}\s*(.+)$", text, re.M)))
        if len(sets) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        # Two answers with NO headers are not perfectly stable — they are
        # unmeasurable by this proxy, and 1.0 would invert the signal it exists
        # to give. This measures whether headers are consistent as a stand-in for
        # whether regex extraction will work, and "no structure at all" is the
        # case where extraction definitely will not.
        scores = [len(a & b) / len(a | b) for a, b in pairs if (a | b)]
        rows.append({"provider": provider, "probe": probe,
                     "condition": condition, "n": len(sets),
                     "mean_headers": round(sum(len(s) for s in sets) / len(sets), 1),
                     "jaccard": (round(sum(scores) / len(scores), 2)
                                 if scores else float("nan"))})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _filter(corpus: pd.DataFrame, filters: dict) -> pd.DataFrame:
    sel = corpus
    for key, value in filters.items():
        if key not in sel.columns:
            raise KeyError(
                f"{key!r} is not a column. Available: "
                f"{sorted(c for c in sel.columns if not c.startswith('_'))}")
        sel = sel[sel[key].isin(value) if isinstance(value, (list, tuple, set))
                  else sel[key] == value]
    return sel


def _has(row, column: str) -> bool:
    value = getattr(row, column, None)
    return value is not None and not pd.isna(value) and value != 0


def _in_heading(text: str, position: int) -> bool:
    line_start = text.rfind("\n", 0, position) + 1
    return text[line_start:line_start + 6].lstrip().startswith("#")


def _indent(text: str | None, prefix: str = "  ", width: int = 84) -> str:
    if not isinstance(text, str):
        return f"{prefix}(none)"
    out = []
    for line in text.split("\n"):
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            out.append(prefix + line[:cut])
            line = line[cut:].lstrip()
        out.append(prefix + line)
    return "\n".join(out)
