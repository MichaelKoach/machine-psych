"""Structural checks on every record, at the moment it is parsed.

This is what replaced a scheduled shape-diffing tier. That tier sampled a
stochastic process once and could not distinguish a changed model from a
different output — it produced a false positive on its first real run, reporting
Gemini's truncation shape as drift when a confirming call showed it unchanged.

The lesson generalises. **A weekly check samples once; a check at parse time sees
every call.** `served_model` is on every record already, and `n_queries` coming
back null on a grounded call is knowable the moment it happens rather than a week
later.

### Warn, never raise

Standard practice across data-pipeline tooling is three violation policies —
warn, drop, fail — with **warn as the default**, and the reason given is precise:
failing loses the metric. A run that dies on the first null `n_queries` cannot
tell you it happened on 3 records or on 300, and that count is the finding.

So nothing here raises. Issues are recorded on the record, counted in the corpus
summary, and printed prominently. *Bad data is silent; good validation makes it
loud* — recording alone is not enough, which is why `describe_integrity` exists
and why the corpus summary calls it.

### Severity

Borrowed from the same source, because a binary is too coarse:

- **critical** — the record is unusable or a claim in it is false
- **warning** — something is missing that should be present, but the record is
  still usable
- **info** — worth counting, not worth acting on
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Issue", "check_record", "describe_integrity"]


@dataclass(frozen=True)
class Issue:
    severity: str        # "critical" | "warning" | "info"
    code: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.detail}"


def check_record(parsed, status: str, intent: dict, caps) -> list[Issue]:
    """Structural problems with one parsed response.

    `status` is passed separately because it is NOT on `ParsedResponse` — the
    provider computes it from the raw body, and parsing does not carry it. That
    separation is deliberate elsewhere in the package and is respected here
    rather than worked around.

    Takes the parsed record, its status, the resolved intent that produced it,
    and the model's capabilities — all four, because most of these checks are
    "the capability table says X and the response says Y".
    """
    issues: list[Issue] = []
    requested = (intent.get("model") or "").split("/", 1)[-1]
    served = parsed.served_model

    # ── the model that answered is not the one requested ─────────────────────
    #
    # All three providers currently echo the requested name exactly, so a
    # divergence is signal rather than noise. This is the one deterministic
    # indicator that a model changed underneath a stable name — an alias
    # resolving to a different build changes the SUBJECT of every record while
    # leaving the roster, the enums and the response shape identical.
    if served and requested and requested not in served and served not in requested:
        issues.append(Issue(
            "critical", "served_model_mismatch",
            f"requested {requested!r}, served {served!r} — the subject of this "
            f"record is not the model the spec named"))

    # ── search actually RAN, and the telemetry is missing ────────────────────
    #
    # THE tool_usage CASE. OpenAI removed `usage.tool_usage` and `n_queries`
    # returned None on every grounded record for a full session. Nothing failed,
    # nothing raised, and the column was simply null.
    #
    # **Gated on `grounded`, not on `intent["search"]`.** These checks asked
    # whether search was PERMITTED, which is the distinction this project spent a
    # session establishing — and then failed to apply here. On a real battery
    # where 5 of 80 records searched, that produced 75 warnings about missing
    # query counts on records that never searched, plus 76 "grounded_but_uncited"
    # on records that were not grounded.
    #
    # 150 warnings out of 80 records, almost all false. An analyst reading that
    # concludes the collection failed. A checker that cries wolf is worse than no
    # checker, which is the failure this module exists to avoid.
    if parsed.grounded and status in ("ok", "truncated"):
        if parsed.n_queries is None:
            issues.append(Issue(
                "warning", "n_queries_absent",
                "search RAN on this record and the provider reported no query "
                "count — the usage field it comes from may have been removed"))
        if len(parsed.citations) == 0 and parsed.answer_chars:
            issues.append(Issue(
                "info", "searched_but_cited_nothing",
                "search ran, sources came back, and the answer cites none of "
                "them — a real behaviour worth looking at, not a defect"))

    # ── a capability the table claims, absent from the response ──────────────
    if caps.readable_reasoning and status == "ok":
        wanted = str(intent.get("reasoning") or "") not in ("", "off", "none")
        if wanted and not parsed.thought_text:
            issues.append(Issue(
                "warning", "reasoning_text_absent",
                "the table records readable_reasoning=True and reasoning was "
                "requested, but no thought text came back"))

    # Also gated on `grounded`: a record that declined to search has no retrieval
    # set BY CONSTRUCTION, and warning about it is noise.
    if (caps.retrieval_set and parsed.grounded and status == "ok"
            and parsed.n_sources_retrieved is None):
        issues.append(Issue(
            "warning", "retrieval_set_absent",
            "the table records retrieval_set=True but no retrieval set came "
            "back — the only provider answering 'what did it see but not "
            "cite' may have stopped reporting it"))

    # ── the answer was separated from process text by a heuristic ────────────
    #
    # One provider (`answer_extraction="heuristic"`) returns blocks that interleave
    # process with answer. The parser takes text AFTER the last tool block as the
    # answer and text before it as process. On an ungrounded record there is no
    # tool block, so all text is answer and nothing can be cut. On a grounded one,
    # text written before the last search lands in process.
    #
    # **Only text written BETWEEN searches is flagged.** A first draft flagged any
    # process text, and fired on a healthy fixture whose only process text was a
    # preamble ("I'll research this for you..."). Preambles are the common case,
    # and a flag that fires on the common case trains readers to ignore the
    # integrity report. Text before the first search is a preamble; text between
    # searches was written mid-research and may be part of the answer.
    interleaved = (parsed.extra or {}).get("interleaved_text_blocks") or 0
    if caps.answer_extraction == "heuristic" and interleaved > 0:
        issues.append(Issue(
            "info", "answer_split_heuristically",
            f"{interleaved} text block(s) were written between searches and filed "
            f"as process, not answer — they may be part of the answer. Read the "
            f"record"))

    # ── the token budget bound the response, even though it finished ─────────
    #
    # On a combined-budget model `max_tokens` caps thinking AND output together.
    # **A call can come back `ok` with its thinking cut short:** the model thinks
    # until the budget runs low, then writes a short answer in what is left. The
    # answer completes, so status is `ok`, and nothing else says so.
    #
    # Measured on gemini-3.8-flash at 16,384: six calls that came back `ok`
    # reported thinking of 15,623-15,729 — the cap minus ~660 for the answer. Run
    # again at 65,536 those six thought 16K-35K. Alongside 13 outright
    # truncations, 19 of 24 calls on that battery were bound by the budget, and
    # status flagged only 13 of them.
    #
    # Only meaningful where the budget is shared, and only when both counts exist:
    # a missing count is "not reported", not zero, and summing it as zero would
    # understate the total.
    budget = intent.get("max_tokens")
    if (caps.combined_token_budget and isinstance(budget, int) and budget > 0
            and parsed.out_tok_reported is not None):
        thinking = 0 if caps.thinking_in_output_tokens else parsed.thinking_tok
        if thinking is not None:
            used = parsed.out_tok_reported + thinking
            if used >= 0.97 * budget:
                issues.append(Issue(
                    "warning", "budget_exhausted",
                    f"used {used:,} of a {budget:,} token budget shared by thinking "
                    f"and output — the cap shaped this response"
                    + (" even though it finished" if status == "ok" else "")
                    + ". Raise max_tokens toward the model's maximum"))

    # ── status and content disagree ──────────────────────────────────────────
    #
    # These are contradictions rather than absences: a record cannot be `ok` and
    # empty, and an answer cannot be present on a record that reports no answer.
    if status == "ok" and not parsed.answer_chars:
        issues.append(Issue(
            "critical", "ok_but_empty",
            "status is ok and the answer is empty — one of the two is wrong"))
    if status == "incomplete" and parsed.answer_chars:
        issues.append(Issue(
            "critical", "incomplete_but_answered",
            f"status is incomplete and {parsed.answer_chars} answer characters "
            f"are present — this is `truncated`, and calling it incomplete "
            f"discards a real partial answer"))

    return issues


def describe_integrity(corpus) -> str:
    """A summary line per issue code. Called by the corpus summary.

    Loud by design. Recording an issue and never surfacing it is the silent
    failure this module exists to prevent, and a count is more informative than
    a first example — three occurrences and three hundred mean different things.
    """
    if "integrity" not in corpus.columns:
        return ""
    from collections import Counter

    counts: Counter = Counter()
    examples: dict[str, str] = {}
    for issues in corpus.integrity.dropna():
        for issue in issues or []:
            severity, code, detail = issue["severity"], issue["code"], issue["detail"]
            counts[(severity, code)] += 1
            examples.setdefault(code, detail)

    if not counts:
        return ""

    order = {"critical": 0, "warning": 1, "info": 2}
    lines = ["", "  INTEGRITY"]
    for (severity, code), n in sorted(counts.items(),
                                      key=lambda kv: (order[kv[0][0]], -kv[1])):
        lines.append(f"    {severity.upper():<9}{code:<26}{n} record(s)")
        lines.append(f"    {'':<9}{examples[code][:96]}")
    return "\n".join(lines)
