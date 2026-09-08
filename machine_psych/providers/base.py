"""The provider contract.

Everything provider-specific lives behind these methods, and everything above
them — the runner, the corpus loader, the analysis surface — sees only
`ParsedResponse`. That boundary is where the design premise gets tested: if a
provider's distinctive structure cannot survive `parse()`, then normalising at
analysis has failed and this is a gateway library with extra steps.

So `ParsedResponse` carries Tier 3 fields that only one provider can fill.
`thought_text` is None on two of three; `n_sources_retrieved` is None on two of
three. **None because a provider cannot is different from None because it did
not**, and the capability table is what tells them apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

import requests

from ..capabilities import (ENUMS, ModelCaps,
                            UnknownModelError, caps_for)

__all__ = ["ParsedResponse", "Provider", "PROVIDERS", "get_provider"]


@dataclass(frozen=True)
class ParsedResponse:
    """One response, extracted. The only shape the layers above ever see.

    Frozen because it crosses a boundary: a parse result mutated downstream is a
    silent corruption path, and the corpus is supposed to record what came back.
    """

    # ── Tier 1 — same meaning on every provider ──────────────────────────────
    answer_text: str | None
    answer_chars: int
    answer_extraction: str
    """'structural' or 'heuristic'. Not a constant of the codebase — Anthropic
    infers the answer from block position and can be wrong; the other two are
    told where it is. Analysis should know which it is holding."""

    # ── Tier 2 — analogous, mapping asserted ─────────────────────────────────
    in_tok_billed: int | None
    in_tok_processed: int | None
    """Different source field per provider. Gemini alone separates them, and by
    two orders of magnitude on grounded calls. Comparing the billed figure across
    providers measures billing policy rather than context volume."""

    out_tok_reported: int | None
    """NOT comparable across providers: two count thinking inside it and one does
    not. `answer_chars` is the comparable length measure."""

    thinking_tok: int | None
    n_queries: int | None
    """From the provider's own usage figure, never from counting blocks. Anthropic
    reported 16 requests against 3 blocks; OpenAI reported 5 against 6 items. The
    same trap in two places."""

    # ── Tier 3 — None where the provider cannot ──────────────────────────────
    thought_text: str | None = None
    n_thought_steps: int | None = None
    n_sources_retrieved: int | None = None
    """Anthropic only. The uncited candidates, which is what makes "why this
    source and not that one" answerable there and nowhere else."""

    n_answer_blocks: int | None = None
    n_process_blocks: int | None = None
    grounded: bool | None = None
    """Gemini only, because there alone the tool is a permission rather than a
    condition. Elsewhere providing search means search happened."""

    # ── side-table rows, provider-shaped but uniformly keyed ─────────────────
    citations: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    queries: list[dict] = field(default_factory=list)
    thoughts: list[dict] = field(default_factory=list)
    units: list[dict] = field(default_factory=list)
    """Blocks on Anthropic, items on OpenAI, steps on Gemini. Named neutrally
    because three names for one concept is the naming rule violated inside our
    own API."""

    # ── provenance ───────────────────────────────────────────────────────────
    served_model: str | None = None
    """What actually answered, which is not always what was asked for."""

    extra: dict = field(default_factory=dict)
    """Provider fields worth keeping that have no column. Not a dumping ground:
    anything numeric that lands here must be listed in DELIBERATELY_UNREAD with a
    reason, and a test enforces it."""


class Provider(ABC):
    """One provider's request-building, dispatch, status and parsing.

    Subclasses declare `name` and `CAPABILITIES`. Completeness is enforced ONCE
    over `CAPABILITIES` in `capabilities.py`, where the data lives — not by a
    class-creation hook.

    An earlier version used `__init_subclass__` for this, and for registration.
    Both were redundant. Every provider's `CAPABILITIES` is a filtered view of
    the same module dict — the objects are literally identical — so a per-class
    check validated a subset of data it did not own. And `ModelCaps` is a
    dataclass with twenty required fields, so presence was already guaranteed by
    the type; the hook's own comment admitted it. Thirty-five lines of
    class-creation machinery replaced by one loop where the data is defined.
    """

    name: str
    CAPABILITIES: dict[str, ModelCaps]

    # ── keys ─────────────────────────────────────────────────────────────────

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                f"No API key for {self.name}. Call "
                f"set_api_key({self.name!r}, ...) before dispatching.")
        return self.api_key

    def caps(self, model: str, fallback: str | None = None) -> ModelCaps:
        """Capabilities for a bare or prefixed model name.

        `fallback` is used when `model` is unknown, and exists because parsers
        read the model from the RESPONSE — which is what the provider actually
        served, and may be a build absent from the table.

        A hard failure there is wrong. `caps_for` raises on an unknown model
        because DISPATCHING to a guess produces an arm whose condition is a
        fiction, and that reasoning does not carry over to parsing: the call has
        already happened, the response exists, and refusing to read it discards
        real data over a name.

        The mismatch is not swallowed — `integrity.check_record` reports
        `served_model_mismatch` as critical on exactly this case, so the record
        is parsed AND the swap is visible.
        """
        key = model if "/" in model else f"{self.name}/{model}"
        try:
            return caps_for(key)
        except UnknownModelError:
            if fallback is None:
                raise
            fb = fallback if "/" in fallback else f"{self.name}/{fallback}"
            return caps_for(fb)

    # ── the contract ─────────────────────────────────────────────────────────

    @abstractmethod
    def build(self, prompt, resolved: dict) -> dict:
        """Resolved intent -> request body.

        No validation of anything the API will catch; unknown parameters are
        rejected by every provider with the offending field named, and those
        rejections stay current for free. Guards exist only for what the API will
        silently accept — a sampling parameter that is ignored, or a field
        belonging to the provider's other endpoint.
        """

    def dispatch(self, body: dict, timeout: int = 1800) -> dict:
        """POST the body, return the response unchanged.

        Concrete, not abstract. All three providers are the same `requests.post`;
        the differences are URL construction and auth headers, and two small
        hooks beat three copies of identical try/except — copies are what drifted
        last time, in seven of ten shared functions.

        NO RETRY HERE. Retry lives in the runner so that `latency` measures a
        single call and `attempts` is visible in the record.
        """
        return self.dispatch_with_status(body, timeout)[1]

    def dispatch_with_status(self, body: dict,
                             timeout: int = 1800) -> tuple[int, dict]:
        """The same POST, keeping the HTTP status code.

        `dispatch` discards the status because the runner does not need it —
        every provider reports its own outcome in the response body, and
        `provider.status(body)` reads that. But the drift probes DO need it: a
        rejection probe distinguishes 400 from 200, and that is the whole signal.

        This exists because they were otherwise rebuilding `url_for` and
        `headers` themselves, which is two more copies of auth and URL
        construction — and this file already carries the lesson that copies are
        what drifted last time, in seven of ten shared functions.
        """
        r = requests.post(self.url_for(body), headers=self.headers(),
                          json=body, timeout=timeout)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"error": {
                "message": f"non-JSON body: {r.text[:400]}",
                "code": r.status_code}}

    @abstractmethod
    def url_for(self, body: dict) -> str:
        """Endpoint for this request. Gemini interpolates the model into the
        path; the other two are constant."""

    @abstractmethod
    def headers(self) -> dict:
        """Auth. Three header names for the same idea."""

    @staticmethod
    def classify_exception(exc: BaseException) -> str:
        """A transport exception -> 'unavailable' or 'error'.

        Concrete and shared: the classification is about the network, not the
        provider. Without this the runner treats every exception as permanent,
        and a single connection reset on a forty-minute battery loses a record
        that a retry would have recovered — which is precisely the failure retry
        exists for. The spec's runner said `if exc: break`, which is wrong for a
        timeout and right for a malformed URL.

        Deliberately narrow. Only errors that are transient BY CONSTRUCTION
        retry; anything else is a real fault and retrying it wastes a minute per
        record and never succeeds.

        Raises on a non-Exception BaseException — KeyboardInterrupt, SystemExit.
        Returning 'error' for those would look safe (it means "do not retry") and
        is not: a caller that wrapped the retry loop in `except BaseException`
        would classify the interrupt, break the loop, record a failed call, and
        CONTINUE THE RUN. The user pressed stop and the battery kept going. So
        this re-raises rather than classifying, and the interrupt reaches the
        handler that returns partial results.
        """
        if not isinstance(exc, Exception):
            raise exc
        transient = (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        )
        return "unavailable" if isinstance(exc, transient) else "error"

    @abstractmethod
    def status(self, body: dict) -> tuple[str, str | None]:
        """-> (state, message). Five states, on every provider.

            ok           finished normally, answer present
            truncated    hit a limit, PARTIAL answer present
            incomplete   hit a limit, NO answer
            unavailable  transient; the runner retries
            error        rejected; the runner does not retry

        `truncated` exists because measurement found it. Anthropic returns 8,
        132, 944 and 4,338 characters of real text alongside
        `stop_reason: max_tokens` depending on the budget — calling that
        `incomplete` discards a usable answer, calling it `ok` hides that it
        stops mid-sentence.

        It may exist on the other two as well. Their boundary characterisations
        used a single small max_tokens value, too small to produce partial text
        by construction, so their predicates are provisional.
        """

    @abstractmethod
    def parse(self, body: dict) -> ParsedResponse:
        """Extract everything the corpus records.

        Usage is derived from the response, never from fields written earlier by
        the runner: if the two disagree, trusting the stored value propagates the
        stale one silently.
        """

    @abstractmethod
    def user_turn(self, text: str) -> dict:
        """A user message in this provider's conversation shape.

        Separate from `passback` because the shapes differ: Anthropic takes
        role/content, OpenAI takes an input list, Gemini takes a `user_input`
        step — and sending the wrong one is a 400 on every turn after the first.
        """

    @abstractmethod
    def passback(self, body: dict) -> list:
        """This response, as history items to append before the next turn.

        Returns a list because providers differ in how many items one response
        becomes: Anthropic is one message, Gemini is every step verbatim
        including thought signatures.

        NOTE: the spec gives two signatures — §2 has `passback(history, body)`
        returning the whole history, §6b calls `provider.passback(body)` and
        appends. This is the second, because a function that takes a list and
        returns a modified copy invites the caller to forget the assignment,
        and because 'what does this response contribute' is the smaller and more
        testable question. §2 should be corrected to match.
        """

    # ── diagnostics ──────────────────────────────────────────────────────────

    def enumerations(self) -> dict[str, tuple[str, ...]]:
        """Recorded valid values per parameter, recovered by bogus-value probing.

        Not used for validation. This is the baseline `compare_capabilities`
        diffs the live API against, which is what would have caught an
        enumeration drifting before a spec was written against the old values.
        """
        return ENUMS.get(self.name, {})




# Populated by `providers/__init__.py`, which owns the registry. Late-bound
# because base.py cannot import its own subclasses.
PROVIDERS: dict[str, type] = {}


def _bind_registry(registry: dict) -> None:
    PROVIDERS.update(registry)


def get_provider(model_or_name: str, api_key: str | None = None) -> Provider:
    """A provider instance from 'anthropic' or 'anthropic/claude-sonnet-5'."""
    name = model_or_name.split("/", 1)[0]
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise UnknownModelError(
            f"No provider registered for {name!r}. Registered: "
            f"{sorted(PROVIDERS) or '(none — import machine_psych.providers)'}"
        ) from None
    return cls(api_key=api_key)
