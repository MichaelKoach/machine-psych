"""Exporting a corpus for analysis in a conversation.

**Derived tables, not raw responses.** The format this replaces was near-raw and
would run 600–900K tokens for a three-provider corpus, which fits in no context
window — and that constraint is what forced the question of what an analysing
conversation actually needs.

It needs prompts, answers, citations with quotes and domains, queries, thought
text, and the identity and status columns. It does not need the raw block and step
arrays, which are largely redundant with the extracted side tables and are where
the bulk sits. Raw stays on disk for questions that need it, and
`include_raw_for` covers the case where a conversation needs one specific record
and cannot reach the disk.

**`capabilities` is in the export**, and that is the part most likely to be
dropped as unnecessary. It is not. An analysing conversation reading a `None` in
`thought_text` cannot otherwise tell whether the provider **cannot** expose
reasoning or simply **did not** on that record — one is a fact about the API and
the other a finding about the model, and they lead to opposite conclusions.
"""

from __future__ import annotations

import json
import pathlib
import pandas as pd

from .capabilities import DELIBERATELY_UNREAD, caps_for
# `units` is deliberately NOT exported. It is the provider-native block/item/step
# structure, it is the largest side table by a wide margin, and every question it
# answers is either already in `citations`/`queries` or needs the raw response
# anyway — which `include_raw_for` provides. Excluding it was an implicit choice
# until an unused-import sweep made it visible; it is a choice now.
from .corpus import (citations, load_corpus, queries, sources, thoughts)

__all__ = ["export_corpus", "HOW_TO_READ"]


HOW_TO_READ = [
    "One investigation's corpus, exported for analysis in a conversation.",
    "",
    "STRUCTURE. `spec` is the investigation that produced these records, "
    "including the rationale fields explaining what each probe was for. "
    "`records` is one entry per API call. The side tables — citations, sources, "
    "queries, thoughts — carry the identity columns, so they group and join "
    "without a merge.",
    "",
    "A record's identity is provider / study / probe / path / rep / turn. "
    "`conversation` groups the turns of one multi-turn exchange. `prompt_hash` "
    "is a content hash, so the SAME PROMPT sent to three providers joins across "
    "them with no shared identifier and no coordination — which is the mechanism "
    "the whole cross-provider design rests on.",
    "",
    "── WHAT IS NOT COMPARABLE ACROSS PROVIDERS ──────────────────────────────",
    "",
    "`out_tok_reported` is NOT comparable across providers. Two count thinking "
    "tokens INSIDE this figure and one keeps it separate and adds it to the "
    "total, so comparing it directly measures accounting convention rather than "
    "output. **Use `answer_chars`**, which is the same measurement everywhere.",
    "",
    "`in_tok_billed` is NOT comparable either. Billing policy differs: one "
    "provider does not bill retrieved search content as input at all, so a "
    "grounded call there reports a fraction of what it processed. "
    "`in_tok_processed` is the comparable figure, and on that provider it has "
    "measured 100x the billed number — a gap that looked architectural until it "
    "turned out to be an invoice.",
    "",
    "`condition` — PROVIDER-SCOPED. Conditions are written per provider because "
    "they are not cross-comparable: a reasoning level on one provider is not the "
    "same manipulation as the same word on another. **Never group on `condition` "
    "across providers.** Group on `config_resolved`, which records what was "
    "actually sent.",
    "",
    "── CITATIONS: COMPARE QUOTES, NOT OFFSETS ───────────────────────────────",
    "",
    "The mechanisms present in THIS corpus are listed under `citation_mechanisms` "
    "below, read from the records rather than asserted here. Known kinds:",
    "  'quoted' — the citation carries the quoted text directly, no offsets",
    "  'char'   — offsets index characters",
    "  'byte'   — offsets index UTF-8 BYTES",
    "  null     — neither: a bare URL list, with nothing to locate in the answer",
    "",
    "Where `quote` is populated it is the comparable object and the offsets are "
    "provenance. Do not re-slice an answer using offsets without checking "
    "`offset_unit`: applying character indexing to byte offsets produces spans "
    "that are silently wrong and drift further into the answer — measured wrong "
    "on 70 of 107 citations on one record, with no error raised.",
    "",
    "**Where `quote` is null there is nothing to compare but the URL.** A provider "
    "that returns a numbered source list attributes the whole answer to the set, "
    "not a span to a source, and a comparison that assumes otherwise silently "
    "drops it.",
    "",
    "── ABSENCE IS NOT ZERO ──────────────────────────────────────────────────",
    "",
    "A `None` in a Tier 3 column can mean the provider CANNOT do the thing or "
    "that it DID NOT on this record. `capabilities` in this file distinguishes "
    "them, and the difference matters: one is a fact about an API, the other a "
    "finding about a model.",
    "",
    "`n_sources_retrieved` is None wherever a provider reports only what it "
    "CITED. `capabilities` below says which models in this corpus expose the "
    "whole retrieval set — every URL seen, cited or not — and that is the only "
    "place 'what distinguishes a cited source from an uncited one' is answerable.",
    "",
    "`grounded` is meaningful only where `search_conditional` is true in "
    "`capabilities` below: there, offering the search tool is a PERMISSION rather "
    "than a condition, and a prompt that does not need retrieval comes back "
    "ungrounded with the tool attached. Elsewhere the tool means search happened, "
    "so the column is None rather than True — and on a search-ALWAYS provider "
    "there is no ungrounded arm to compare against at all.",
    "",
    "`thought_text` is present on the providers that expose readable reasoning. "
    "On one of them the summary must be REQUESTED per call, so a None there can "
    "also mean the request did not ask — `config_resolved` on the record says "
    "which.",
    "",
    "── STATUS ───────────────────────────────────────────────────────────────",
    "",
    "  ok          finished normally, answer present",
    "  truncated   hit a limit, PARTIAL answer present and usable",
    "  incomplete  hit a limit, NO answer",
    "  unavailable transient failure that exhausted its retries",
    "  error       rejected",
    "",
    "`truncated` fires on some providers and not others, and that is "
    "architectural rather than incidental: one that streams text into blocks as "
    "it generates gets caught mid-answer, while one that emits its message "
    "complete or not at all does not. Check `statuses_seen` below for which "
    "appear here. A truncated record's answer is REAL and should not be "
    "discarded.",
    "",
    "`conversation_status` is separate: `failed` means the conversation died on "
    "turn 0, `incomplete` means it died later and earlier turns are usable.",
    "",
    "── OMITTED ──────────────────────────────────────────────────────────────",
    "",
    "Raw response bodies. They are on disk and are several times the size of "
    "everything here — mostly encrypted reasoning signatures and provider widget "
    "markup that no analysis reads. `include_raw_for=[record_ids]` attaches "
    "specific ones.",
    "",
    "The `units` table — the provider-native block/item/step structure. It is the "
    "largest side table and every question it answers is either already in "
    "`citations` and `queries` or needs the raw response anyway. Reachable from "
    "the corpus on disk as `units(corpus)`.",
    "",
    "── READING IT ───────────────────────────────────────────────────────────",
    "",
    "**Extract before you read.** Pull pairings, entities and citations "
    "mechanically across ALL records first, then read to understand what the "
    "numbers mean. Reading a few records and inferring the pattern is how this "
    "project produced three findings that died at n=7 — and in each case the "
    "records that were read were simply the ones a notebook happened to print "
    "first, which is a sampling decision disguised as a display choice.",
    "",
    "**One record is a story; five is a finding.** Run-to-run variance is high "
    "on at least one provider — six of six distinct answers on a one-sentence "
    "prompt, with product-level disagreement — so a difference between two arms "
    "must clear that floor before it means anything.",
]


def export_corpus(investigation: str, run: str | None = None,
                  path=None, include_raw_for: list[int] | None = None,
                  max_answer_chars: int | None = None,
                  quiet: bool = False) -> pathlib.Path:
    """Write one run as derived tables. Returns the path.

    `include_raw_for` attaches full raw responses for named records — the escape
    hatch for when a conversation needs to see exactly what came back and cannot
    reach the disk.

    **`max_answer_chars` exists because dropping the raw bodies is not enough.**
    Measured: this format runs roughly 1,700 tokens per record, of which the
    answer text is most of it. That is a 3x improvement on the near-raw format
    and it does NOT solve the problem — a 200-record corpus is still ~340K
    tokens, because 200 real answers ARE 300K tokens and no format change
    removes them.

    So a large corpus needs a deliberate decision rather than a smaller
    serialisation: truncate the answers here and read specific ones in full from
    disk, or export a filtered corpus. Truncation is recorded per record in
    `answer_truncated_for_export`, so an analysis cannot mistake a cut answer for
    a short one — which would turn an export setting into a finding about
    verbosity.
    """
    corpus = _load_quietly(investigation, run)
    run_dir = pathlib.Path(corpus.attrs["run_dir"])

    spec = _read_json(run_dir / "_spec.json")
    meta = _read_json(run_dir / "_run.json")

    models = sorted(corpus.model.dropna().unique())
    payload = {
        "how_to_read_this": HOW_TO_READ,
        "provenance": {
            # Three mechanisms, three questions: what code ran, whether the
            # instrument changed, and what each intent resolved to. Three corpora
            # collected earlier in this project were produced by code that has
            # since had bugs fixed, and there is no way to reconstruct what the
            # collecting version did.
            "harness": meta.get("harness"),
            "probe_hashes": meta.get("probe_hashes"),
            "spec_sha256": meta.get("spec_sha256"),
            "collected_at": meta.get("started_at"),
            "investigation": investigation,
            "run": run_dir.name,
        },
        "spec": spec,
        "capabilities": {m: _caps_summary(m) for m in models},
        # Read from the records, because the guide above now POINTS at these
        # instead of asserting counts. Static prose saying "three mechanisms" and
        # "quote is populated for all three" became false the moment a fourth
        # provider appeared whose citations are a bare URL list — and that
        # document travels WITH the corpus to an analysing conversation.
        "citation_mechanisms": _mechanisms(corpus),
        "statuses_seen": (sorted(corpus.status.dropna().unique())
                          if len(corpus) else []),
        "deliberately_unread": DELIBERATELY_UNREAD,
        "records": _records(corpus, include_raw_for, run_dir, max_answer_chars),
        "citations": _rows(citations(corpus)),
        "sources": _rows(sources(corpus)),
        "queries": _rows(queries(corpus)),
        "thoughts": _rows(thoughts(corpus)),
    }

    dest = pathlib.Path(path) if path else run_dir / "_corpus.json"
    dest.write_text(json.dumps(payload, indent=1, ensure_ascii=False,
                               default=str))

    if not quiet:
        size_kb = dest.stat().st_size / 1024
        raw_kb = sum(p.stat().st_size for p in run_dir.glob("[0-9]*.json")) / 1024
        n = len(payload["records"])
        # ~4 chars per token. Rough, and the point is the order of magnitude.
        tokens = dest.stat().st_size // 4
        print(f"  exported {n} records · {size_kb:,.0f} KB · ~{tokens:,} tokens")
        if raw_kb:
            print(f"    {raw_kb:,.0f} KB raw on disk → "
                  f"{raw_kb / max(size_kb, 1):.1f}x smaller")
        if n:
            per = tokens / n
            print(f"    ~{per:,.0f} tokens/record → a 200-record corpus would be "
                  f"~{per * 200 / 1000:,.0f}K tokens")
            # MEASURED, not assumed. An earlier version of this advice said "the
            # answers are the bulk" and offered truncation — which saved 5% on
            # the first corpus tested, because the bulk was the side tables. The
            # breakdown is printed so the decision is made against this corpus
            # rather than against a guess about corpora in general.
            for section, share in _breakdown(payload):
                if share >= 0.08:
                    print(f"      {section:<14} {share:>5.0%}")
            if per * 200 > 200_000:
                print(f"    a corpus that size does not fit a context window. "
                      f"Reduce the LARGEST section above —")
                print(f"    max_answer_chars only helps if `records` dominates.")
        print(f"    {dest}")
    return dest


def _breakdown(payload: dict) -> list[tuple[str, float]]:
    """Which sections actually account for the size.

    Exists because an assumption about this was wrong: the obvious answer is that
    the answer text dominates, and on the first corpus measured the side tables
    did. Guessing here produces advice that sounds right and does nothing.
    """
    sizes = {k: len(json.dumps(v, default=str)) for k, v in payload.items()}
    total = sum(sizes.values()) or 1
    return sorted(((k, v / total) for k, v in sizes.items()),
                  key=lambda kv: -kv[1])


def _mechanisms(corpus) -> dict:
    """Which citation mechanisms are present, and whether each carries a quote.

    Derived rather than declared. A provider returning a numbered source list has
    no offsets AND no quote — it attributes the whole answer to the set rather
    than a span to a source — and an analysis assuming a quote exists would
    silently drop it.
    """
    from .corpus import citations
    try:
        cites = citations(corpus)
    except (KeyError, AttributeError):
        return {}
    if not len(cites):
        return {}
    out = {}
    for unit, group in cites.fillna({"offset_unit": "null"}).groupby("offset_unit"):
        quoted = int(group["quote"].notna().sum())
        out[str(unit)] = {
            "citations": len(group),
            "with_quote": quoted,
            "providers": sorted(group.provider.dropna().unique()),
            "note": ("quotes present — compare these" if quoted == len(group)
                     else "NO quote: only the URL is comparable" if quoted == 0
                     else "quote present on some rows only"),
        }
    return out


def _caps_summary(model: str) -> dict:
    """The capability fields an analysing conversation needs.

    Not all of them — `tokens_per_query` and `measured_on` are for planning a run
    rather than reading one. These are the fields that decide how to interpret a
    None.
    """
    caps = caps_for(model)
    return {
        "readable_reasoning": caps.readable_reasoning,
        "retrieval_set": caps.retrieval_set,
        "search_conditional": caps.search_conditional,
        "citation_offsets": caps.citation_offsets,
        "answer_extraction": caps.answer_extraction,
        "thinking_in_output_tokens": caps.thinking_in_output_tokens,
        "reasoning_off": caps.reasoning_off,
        "page_age": caps.page_age,
        "measured_on": caps.measured_on,
        "notes": caps.notes or None,
    }


# Columns that would bloat the export without informing an analysis. `file` is a
# local path meaningless elsewhere; `path_on_disk` the same.
_SKIP_COLUMNS = {"file", "path_on_disk"}


def _records(corpus: pd.DataFrame, include_raw_for, run_dir,
             max_answer_chars=None) -> list[dict]:
    wanted = set(include_raw_for or [])
    raw_by_id = {}
    if wanted:
        for path in sorted(run_dir.glob("[0-9]*.json")):
            d = json.loads(path.read_text())
            rec_id = int(path.name.split("_")[0])
            if rec_id in wanted:
                raw_by_id[rec_id] = d.get("response")

    out = []
    for row in corpus.to_dict("records"):
        record = {k: _clean(v) for k, v in row.items()
                  if k not in _SKIP_COLUMNS}
        text = record.get("answer_text")
        if max_answer_chars and isinstance(text, str) and len(text) > max_answer_chars:
            record["answer_text"] = text[:max_answer_chars]
            # Recorded, not silent. An analysis that mistook a truncated answer
            # for a short one would read an export setting as a finding about
            # verbosity — and `answer_chars` is deliberately left at the TRUE
            # length so length comparisons stay valid.
            record["answer_truncated_for_export"] = len(text)
        if row.get("record_id") in raw_by_id:
            record["_raw_response"] = raw_by_id[row["record_id"]]
        out.append(record)
    return out


def _rows(df: pd.DataFrame) -> list[dict]:
    if not len(df):
        return []
    return [{k: _clean(v) for k, v in row.items()}
            for row in df.to_dict("records")]


def _clean(value):
    """NaN to None, so the JSON says 'absent' rather than 'NaN'.

    This is the one place a fill IS correct: JSON has no NaN, and `null` is the
    honest rendering of a value that is not present. It is emphatically NOT
    filling a Tier 3 column with False — the distinction is that None preserves
    'unknown or inapplicable' while False would claim the provider declined.
    """
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is pd.NaT:
        return None
    return value


def _load_quietly(investigation: str, run: str | None):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return load_corpus(investigation, run)


def _read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}
