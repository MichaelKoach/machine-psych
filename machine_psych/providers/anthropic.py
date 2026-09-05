"""Anthropic.

Built first deliberately, because it is the awkward one. It has the only
heuristic answer extraction, the only retrieval set, the only citation form that
is neither byte nor character offsets, and blocks that interleave process with
answer. If `ParsedResponse` survives this provider it survives the other two; had
OpenAI gone first — one clean `message` item, structural extraction — the
contract would have looked adequate and then broken.

Everything here was written against recorded fixtures, never against the design
document. That is not caution for its own sake: on the day these fixtures were
collected, four things in the document turned out to be wrong, and each would
have passed a synthetic suite built from it.
"""

from __future__ import annotations

from ..capabilities import CAPABILITIES, model_of
from .base import ParsedResponse, Provider

__all__ = ["Anthropic"]

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

# Blocks that are neither answer nor tool. `thinking` is the reason this constant
# exists: it is not text, so a naive "last non-text block" search would treat it
# as a tool boundary and cut the answer at the model's own reasoning. Since
# thinking now fires on every call by default, that would truncate every record.
NON_TOOL_BLOCKS = {"text", "thinking", "redacted_thinking"}


class Anthropic(Provider):
    name = "anthropic"
    CAPABILITIES = {k: v for k, v in CAPABILITIES.items()
                    if k.startswith("anthropic/")}

    # ── transport ────────────────────────────────────────────────────────────

    def url_for(self, body: dict) -> str:
        return API_URL

    def headers(self) -> dict:
        return {"x-api-key": self.require_key(),
                "anthropic-version": API_VERSION,
                "content-type": "application/json"}

    # ── request ──────────────────────────────────────────────────────────────

    def build(self, prompt, resolved: dict) -> dict:
        """Resolved intent -> request body.

        The reasoning mapping was re-measured on 2026-08-27 and had moved.
        `thinking: {"type": "enabled", "budget_tokens": N}` — what every earlier
        note assumed — now returns 400 on claude-sonnet-5:

            "thinking.type.enabled" is not supported for this model.
            Use "thinking.type.adaptive" and "output_config.effort".

        So intensity is `output_config.effort`, whose enumeration is
        low|medium|high|xhigh|max — OpenAI's minus `none`. Thinking is also ON BY
        DEFAULT here now: a call sending no thinking key returned 823 thinking
        tokens.
        """
        model = resolved.get("model", "claude-sonnet-5")
        body: dict = {
            "model": model_of(model),
            "max_tokens": resolved.get("max_tokens", 4096),
            "messages": prompt if isinstance(prompt, list)
                        else [{"role": "user", "content": prompt}],
        }

        if resolved.get("system"):
            body["system"] = resolved["system"]

        reasoning = resolved.get("reasoning")
        if reasoning == "off":
            body["thinking"] = {"type": "disabled"}
        elif reasoning is not None:
            # Adaptive is the only enabled form now. Effort modulates it and
            # works with or without the thinking key; both are sent so the
            # recorded request states the intent explicitly rather than relying
            # on a default that has already changed once.
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": reasoning}

        search = resolved.get("search")
        if search:
            tool = dict(SEARCH_TOOL)
            if isinstance(search, dict):
                tool.update(search)
            body["tools"] = [tool]

        # Anything the resolver passed through that is not an intent this method
        # consumes — the escape hatch, and any parameter added since. Sent as-is,
        # because the API rejects unknown fields and names them, and duplicating
        # that here would be a second thing to keep in sync with a moving API.
        consumed = {"model", "max_tokens", "system", "reasoning", "search",
                    "repetitions", "exclude", "include", "note"}
        for key, value in resolved.items():
            if key not in consumed:
                body[key] = value
        return body

    def user_turn(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def passback(self, body: dict) -> list:
        """One assistant message carrying every block verbatim.

        Verbatim including thinking blocks and their signatures. A thinking block
        whose signature is stripped or altered invalidates the turn, and dropping
        them entirely changes what the model is continuing from — the corpus
        should record what the API returned, and the runner should not make
        silent editorial choices about it.
        """
        content = body.get("content") or []
        return [{"role": "assistant", "content": content}] if content else []

    # ── status ───────────────────────────────────────────────────────────────

    def status(self, body: dict) -> tuple[str, str | None]:
        """Five states. See `Provider.status`.

        Two measured facts shape this:

        **Truncation returns a PARTIAL answer.** `stop_reason: max_tokens` came
        back with 8, 132, 944 and 4,338 characters of real text across four
        budgets. `incomplete` would discard a usable answer; `ok` would hide that
        it stops mid-sentence.

        **The no-answer shape is "no text block", not "empty content".** An
        August measurement found `content: []` at max_tokens 5; the same call now
        returns a thinking block with empty thinking text, because thinking runs
        first and consumed the budget. A predicate written against the older
        shape would call this `ok` with a null answer — the precise bug the
        fifth state exists to prevent, reintroduced by a stale fixture.
        """
        if not isinstance(body, dict):
            return "error", f"non-dict body: {type(body).__name__}"

        err = body.get("error")
        if err:
            kind = err.get("type", "")
            message = err.get("message", "")
            # Errors are uniform in shape across 400 and 404 — `type: "error"`
            # with `error.type` and `error.message` — so one classifier covers
            # them. Only overloaded_error and rate_limit are worth retrying;
            # retrying an invalid_request_error five times wastes a minute per
            # record and never succeeds.
            transient = kind in ("overloaded_error", "rate_limit_error",
                                 "api_error")
            return ("unavailable" if transient else "error"), f"{kind}: {message}"

        content = body.get("content") or []
        # The SAME extraction the parser uses, not a looser "is there any text
        # anywhere" check. Those two definitions disagree on exactly the records
        # where the answer matters: a truncated grounded call can carry process
        # commentary BEFORE its tool calls — "I'll research current CRM
        # options..." — and no answer after them. The loose check calls that
        # `truncated` while `parse` reports zero answer characters, so a record
        # would claim a partial answer and contain none.
        answer, _, _ = self._split_answer(content)
        has_text = bool(answer)
        stop = body.get("stop_reason")

        if stop == "max_tokens":
            return ("truncated" if has_text else "incomplete",
                    "hit max_tokens" + ("" if has_text else ", no text block"))
        if stop == "refusal":
            return "incomplete", "refusal"
        if not has_text:
            # end_turn with no text is not a shape anyone has observed, and
            # guessing at it would be the kind of assumption these fixtures exist
            # to prevent. Reporting it as incomplete with the stop reason means
            # the record survives and the surprise is visible.
            return "incomplete", f"no text block (stop_reason={stop})"
        return "ok", None

    # ── parse ────────────────────────────────────────────────────────────────

    def parse(self, body: dict) -> ParsedResponse:
        content = body.get("content") or []
        usage = body.get("usage") or {}
        caps = self.caps(body.get("model") or "claude-sonnet-5")

        answer_text, n_answer, n_process = self._split_answer(content)

        return ParsedResponse(
            answer_text=answer_text or None,
            answer_chars=len(answer_text or ""),
            # Read from capabilities, not restated here. Hardcoding "heuristic"
            # would make the parser a second source of truth for a fact the table
            # already holds, and the two could then disagree without anything
            # noticing — which is how a documented constant becomes a lie.
            answer_extraction=caps.answer_extraction,

            in_tok_billed=usage.get("input_tokens"),
            in_tok_processed=usage.get(caps.in_tok_processed_field),
            out_tok_reported=usage.get("output_tokens"),
            thinking_tok=(usage.get("output_tokens_details") or {}
                          ).get("thinking_tokens"),
            # From the usage figure, NEVER from counting blocks. Measured 16
            # requests against 3 server_tool_use blocks — a block count would
            # have been wrong by a factor of five.
            n_queries=(usage.get("server_tool_use") or {}
                       ).get("web_search_requests"),

            # `readable_reasoning` is False here: thinking blocks carry a
            # signature and an empty `thinking` string. Gated on the capability
            # rather than assumed, so a provider that gains readable reasoning
            # needs one table edit and not a parser rewrite.
            thought_text=(self._thought_text(content)
                          if caps.readable_reasoning else None),
            # NOT `or None`. Zero thinking blocks is a measurement — the model
            # produced none — and None means the concept does not apply. An
            # earlier version wrote `sum(...) or None` and destroyed that
            # distinction on exactly the records where it matters most: a
            # reasoning-off arm would report None rather than 0, and no analysis
            # could tell it apart from a provider that cannot think.
            n_thought_steps=sum(1 for b in content if b.get("type") == "thinking"),
            n_sources_retrieved=(self._n_retrieved(content)
                                 if caps.retrieval_set else None),
            n_answer_blocks=n_answer,
            n_process_blocks=n_process,
            # `grounded` is only meaningful where the tool is a permission
            # rather than a condition. Here providing search means search ran, so
            # None says the concept does not apply — distinct from False, which
            # would claim it declined.
            grounded=(self._grounded(content) if caps.search_conditional else None),

            citations=self._citations(content, caps.citation_offsets),
            sources=self._sources(content, caps.page_age),
            queries=self._queries(content),
            thoughts=[],
            units=self._units(content),

            served_model=body.get("model"),
            extra={"stop_reason": body.get("stop_reason"),
                   "stop_details": body.get("stop_details"),
                   "container": body.get("container"),
                   "cached_tok": usage.get("cache_read_input_tokens"),
                   "cache_write_tok": usage.get("cache_creation_input_tokens")},
        )

    # ── extraction helpers ───────────────────────────────────────────────────

    @staticmethod
    def _split_answer(content: list) -> tuple[str, int, int]:
        """Text after the last tool block is the answer; text before is process.

        THE ONLY HEURISTIC IN ANY PROVIDER, and it can be wrong in ways that are
        hard to notice downstream — which is why `answer_extraction` records that
        this column was inferred rather than read.

        `thinking` is excluded from the tool boundary. It is not text, so a naive
        "last non-text block" search would treat the model's own reasoning as a
        tool boundary and cut the answer there. Since thinking now fires on every
        call by default, that would truncate every grounded record.

        On the three grounded fixtures the heuristic cut nothing — no text
        appeared before the last tool block in any of them, because the model
        searches first and then writes. That is three records, not a property.
        """
        tool_idx = [i for i, b in enumerate(content)
                    if b.get("type") not in NON_TOOL_BLOCKS]
        boundary = max(tool_idx) + 1 if tool_idx else 0
        after = content[boundary:]
        before = content[:boundary]
        answer = "".join(b.get("text", "") for b in after
                         if b.get("type") == "text")
        n_answer = sum(1 for b in after if b.get("type") == "text")
        n_process = sum(1 for b in before if b.get("type") == "text")
        # Counts, never `or None`. Zero process blocks means the model searched
        # first and then wrote — which is what all three grounded fixtures show,
        # and is a finding. None would say the split does not apply here, which
        # is false and would hide it.
        return answer, n_answer, n_process

    @staticmethod
    def _thought_text(content: list) -> str | None:
        """Readable reasoning, if this provider had any.

        It does not: `thinking` blocks carry a signature and an empty string.
        Present so the capability gate has something to call and so a provider
        that gains readable reasoning changes one table entry.
        """
        parts = [b.get("thinking") for b in content
                 if b.get("type") == "thinking" and b.get("thinking")]
        return "\n\n".join(parts) or None

    @staticmethod
    def _grounded(content: list) -> bool:
        return any(b.get("type") == "web_search_tool_result" for b in content)

    @staticmethod
    def _citations(content: list, offset_unit: str | None) -> list[dict]:
        """One row per citation, carrying the quote directly.

        No offsets. This provider's citations are `web_search_result_location`
        entries with `cited_text` — a third mechanism alongside byte offsets and
        character offsets, and the easiest to work with, since the analysis rule
        is compare quotes rather than offsets and the quote arrives without any
        arithmetic to get wrong.
        """
        rows = []
        for i, block in enumerate(content):
            if block.get("type") != "text":
                continue
            for cite in block.get("citations") or []:
                rows.append({
                    "unit": i,
                    "type": cite.get("type"),
                    "offset_unit": offset_unit,
                    "start": None,
                    "end": None,
                    "quote": cite.get("cited_text"),
                    "url": cite.get("url"),
                    "title": cite.get("title"),
                    "is_redirect": False,
                })
        return rows

    @staticmethod
    def _sources(content: list, page_age: bool) -> list[dict]:
        """Every URL retrieved, with `cited` marking which were used.

        THE RETRIEVAL SET, which exists on no other provider. OpenAI and Gemini
        report only what they cited, so "what distinguishes a cited source from
        an uncited one" is answerable here and nowhere else — and that question
        is what the scraping layer is being built for.

        `page_age` comes along free and is also unique: how fresh the sources
        behind a recommendation are is a direct question for this research.
        """
        cited_urls = {c.get("url") for block in content
                      if block.get("type") == "text"
                      for c in (block.get("citations") or [])}
        rows, seen = [], set()
        for block in content:
            if block.get("type") != "web_search_tool_result":
                continue
            for result in block.get("content") or []:
                url = result.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                rows.append({
                    "url": url,
                    "title": result.get("title"),
                    **({"page_age": result.get("page_age")} if page_age else {}),
                    "retrieved": True,
                    "cited": url in cited_urls,
                    # Declared even though it is always False here. The citations
                    # rows on this provider already set it; sources did not, so
                    # the column came back all-NaN and every downstream branch
                    # keyed on it took the wrong path — NaN is truthy.
                    "is_redirect": False,
                    "tool_use_id": block.get("tool_use_id"),
                    "caller": (block.get("caller") or {}).get("type"),
                })
        return rows

    @staticmethod
    def _queries(content: list) -> list[dict]:
        """One row per search issued.

        `caller` distinguishes a direct search from one wrapped in code
        execution — two tool modes with different cross-turn behaviour, which is
        why `ModelCaps.cross_turn` is a provider-level field describing something
        that varies per mode and is documentation-only for that reason.
        """
        return [{
            "call_id": b.get("id"),
            "search": (b.get("input") or {}).get("query"),
            "caller": (b.get("caller") or {}).get("type", "direct"),
        } for b in content
            if b.get("type") == "server_tool_use" and b.get("name") == "web_search"]

    @staticmethod
    def _units(content: list) -> list[dict]:
        """Provider-native blocks, one row each.

        Signatures are recorded as a length rather than kept: they run to
        thousands of characters, are opaque, and would dominate any export.
        """
        rows = []
        for i, b in enumerate(content):
            sig = b.get("signature")
            rows.append({
                "unit": i,
                "type": b.get("type"),
                "text": b.get("text"),
                "n_citations": len(b.get("citations") or []) or None,
                "sig_chars": len(sig) if isinstance(sig, str) else None,
                "tool_use_id": b.get("tool_use_id") or b.get("id"),
                "n_results": (len(b.get("content") or [])
                              if b.get("type") == "web_search_tool_result" else None),
            })
        return rows

    @staticmethod
    def _n_retrieved(content: list) -> int | None:
        """Unique URLs retrieved, or None when nothing searched.

        This one IS `None` on an ungrounded call, and correctly: a call with no
        search tool has no retrieval set, which is different from a search that
        returned nothing. The distinction is only expressible because the
        surrounding counts stopped conflating zero with absent.
        """
        if not any(b.get("type") == "web_search_tool_result" for b in content):
            return None
        return len({r.get("url") for b in content
                    if b.get("type") == "web_search_tool_result"
                    for r in (b.get("content") or []) if r.get("url")})
