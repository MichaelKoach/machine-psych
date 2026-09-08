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

    # ── search was requested and no query count came back ────────────────────
    #
    # THE tool_usage CASE. OpenAI removed `usage.tool_usage` and `n_queries`
    # returned None on every grounded record for a full session. Nothing failed,
    # nothing raised, and the column was simply null. Caught here, it is visible
    # on the run that hits it.
    if intent.get("search") and status in ("ok", "truncated"):
        if parsed.n_queries is None:
            issues.append(Issue(
                "warning", "n_queries_absent",
                "search was enabled and the provider reported no query count — "
                "the usage field it comes from may have been removed"))
        if len(parsed.citations) == 0 and parsed.answer_chars:
            issues.append(Issue(
                "info", "grounded_but_uncited",
                "search was enabled and the answer cites nothing"))

    # ── a capability the table claims, absent from the response ──────────────
    if caps.readable_reasoning and status == "ok":
        wanted = str(intent.get("reasoning") or "") not in ("", "off", "none")
        if wanted and not parsed.thought_text:
            issues.append(Issue(
                "warning", "reasoning_text_absent",
                "the table records readable_reasoning=True and reasoning was "
                "requested, but no thought text came back"))

    if (caps.retrieval_set and intent.get("search") and status == "ok"
            and parsed.n_sources_retrieved is None):
        issues.append(Issue(
            "warning", "retrieval_set_absent",
            "the table records retrieval_set=True but no retrieval set came "
            "back — the only provider answering 'what did it see but not "
            "cite' may have stopped reporting it"))

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
