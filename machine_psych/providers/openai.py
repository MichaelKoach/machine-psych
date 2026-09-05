"""OpenAI.

Second, and easier than the first by design. Anthropic went first because it is
the awkward one — heuristic answer extraction, a retrieval set, blocks that
interleave process with answer. This provider is the clean case, and building it
second means the contract was stressed before it was flattered.

Three measured differences shape what is here:

**No `truncated` state.** The budget was swept across five values because that is
the only way the state was found on Anthropic. Every incomplete arm returned zero
message items and zero text; 4096 completed cleanly with 4,891 characters. The
difference is architectural — Anthropic streams text into content blocks as it
generates, so truncation catches it mid-answer, while this provider emits the
message item complete or not at all.

**Readable reasoning, opt-in.** `reasoning.summary` returned empty arrays at
characterisation time and populates now. It arrives only when requested, unlike
Gemini's, which comes whenever thinking summaries are on.

**Character-offset citations**, against Anthropic's quoted text. This is the first
real test of whether one `citations()` shape can hold two mechanisms without
either being second-class.
"""

from __future__ import annotations

from ..capabilities import CAPABILITIES, model_of
from .base import ParsedResponse, Provider

__all__ = ["OpenAI"]

API_URL = "https://api.openai.com/v1/responses"

SEARCH_TOOL = {"type": "web_search"}

# NOTE: an earlier draft defined NON_ANSWER_ITEMS here, mirroring Anthropic's
# NON_TOOL_BLOCKS. It was never referenced — this provider's answer extraction is
# structural (`type == "message"`), so there is no boundary to compute and no set
# to exclude from it. Removed rather than kept: an unread constant looks
# load-bearing, and the mirror-image structure of the two modules made it read as
# though it were.


class OpenAI(Provider):
    name = "openai"
    CAPABILITIES = {k: v for k, v in CAPABILITIES.items()
                    if k.startswith("openai/")}

    # ── transport ────────────────────────────────────────────────────────────

    def url_for(self, body: dict) -> str:
        return API_URL

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.require_key()}",
                "Content-Type": "application/json"}

    # ── request ──────────────────────────────────────────────────────────────

    def build(self, prompt, resolved: dict) -> dict:
        """Resolved intent -> request body.

        `store: false` by default. The endpoint otherwise keeps the response
        server-side for later resumption, which means a corpus would depend on
        provider state rather than being self-contained — and a research artifact
        that cannot be reconstructed from its own files is not one.
        """
        model = resolved.get("model", "gpt-5.6-sol")
        body: dict = {
            "model": model_of(model),
            "max_output_tokens": resolved.get("max_tokens", 8192),
            "input": prompt,
            "store": False,
        }

        if resolved.get("system"):
            body["instructions"] = resolved["system"]

        reasoning = resolved.get("reasoning")
        if reasoning is not None:
            # `none` is a real level here — measured at 0 reasoning tokens — and
            # is the off arm that one provider cannot express at all. So unlike
            # Anthropic, "off" maps to a value rather than to a different key.
            body["reasoning"] = {"effort": "none" if reasoning == "off" else reasoning}
            # Readable reasoning is OPT-IN on this provider. Without this the
            # items carry `encrypted_content` only, and the capability would be
            # recorded as available while every record showed nothing.
            caps = self.caps(model)
            if caps.readable_reasoning and reasoning != "off":
                body["reasoning"]["summary"] = "detailed"

        if resolved.get("verbosity"):
            # The only output-length control of the three providers. Measured
            # swinging output 5.7x AND restructuring the answer — one markdown
            # header at low, thirteen at high — so it is a confound in any
            # cross-condition length comparison, not merely a length knob.
            body["text"] = {"verbosity": resolved["verbosity"]}

        if resolved.get("search"):
            tool = dict(SEARCH_TOOL)
            if isinstance(resolved["search"], dict):
                tool.update(resolved["search"])
            body["tools"] = [tool]

        if resolved.get("max_tool_calls") is not None:
            # The only search cap of the three. Measured 65-77% cheaper while
            # still producing a complete answer, which is what makes n_queries
            # the cost lever rather than max_tokens.
            body["max_tool_calls"] = resolved["max_tool_calls"]

        consumed = {"model", "max_tokens", "system", "reasoning", "search",
                    "verbosity", "max_tool_calls",
                    "repetitions", "exclude", "include", "note"}
        for key, value in resolved.items():
            if key not in consumed:
                body[key] = value
        return body

    def user_turn(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def passback(self, body: dict) -> list:
        """Every output item verbatim.

        With `store: false` there is no server-side conversation to resume, so
        history is the caller's responsibility. Reasoning items go back too —
        they carry `encrypted_content` the model uses to continue, and a turn
        assembled without them is a different turn.
        """
        return list(body.get("output") or [])

    # ── status ───────────────────────────────────────────────────────────────

    def status(self, body: dict) -> tuple[str, str | None]:
        """Four states in practice: `truncated` never fires here.

        HTTP 200 IS NOT SUCCESS. `status: "incomplete"` arrives with a 200 and no
        message item, which is the shape that made this provider the reason the
        predicate exists at all.

        And **not every 429 is transient**. An exhausted credit balance returns
        429 with `type: insufficient_quota`; retrying it means five attempts per
        record across a battery, hours wasted, never succeeding. The error type
        discriminates, never the status code — a distinction learned by writing
        the wrong retry loop by hand in a probe cell while this method's
        specification already said otherwise.
        """
        if not isinstance(body, dict):
            return "error", f"non-dict body: {type(body).__name__}"

        err = body.get("error")
        if err:
            kind = err.get("type", "")
            message = err.get("message", "")
            transient = kind in ("rate_limit_exceeded", "server_error",
                                 "service_unavailable", "api_error")
            return ("unavailable" if transient else "error"), f"{kind}: {message}"

        state = body.get("status")
        has_text = bool(self._answer(body.get("output") or []))

        if state == "incomplete":
            reason = ((body.get("incomplete_details") or {}).get("reason")
                      or "unspecified")
            # `truncated` is checked for rather than assumed absent. Five budgets
            # produced no partial text, but that is a measurement on one prompt
            # and not a guarantee — if a partial answer ever appears, it should
            # be classified correctly rather than silently discarded.
            return ("truncated" if has_text else "incomplete"), f"incomplete: {reason}"
        if state in ("failed", "cancelled"):
            return "error", f"status={state}"
        if state == "in_progress":
            return "incomplete", "still running"
        if not has_text:
            return "incomplete", f"no message item (status={state})"
        return "ok", None

    # ── parse ────────────────────────────────────────────────────────────────

    def parse(self, body: dict) -> ParsedResponse:
        output = body.get("output") or []
        usage = body.get("usage") or {}
        caps = self.caps(body.get("model") or "gpt-5.6-sol")

        answer_text = self._answer(output)
        n_answer = sum(1 for i in output if i.get("type") == "message")
        n_process = sum(1 for i in output if i.get("type") == "reasoning")

        return ParsedResponse(
            answer_text=answer_text or None,
            answer_chars=len(answer_text or ""),
            # Structural, not inferred: the message item IS the answer. Anthropic
            # has to guess from block position, and the column records which kind
            # of claim it is holding.
            answer_extraction=caps.answer_extraction,

            in_tok_billed=usage.get("input_tokens"),
            in_tok_processed=usage.get(caps.in_tok_processed_field),
            out_tok_reported=usage.get("output_tokens"),
            thinking_tok=(usage.get("output_tokens_details") or {}
                          ).get("reasoning_tokens"),
            n_queries=self._n_queries(usage, output),

            thought_text=(self._thought_text(output)
                          if caps.readable_reasoning else None),
            n_thought_steps=n_process,
            n_sources_retrieved=(self._n_retrieved(output)
                                 if caps.retrieval_set else None),
            n_answer_blocks=n_answer,
            n_process_blocks=n_process,
            grounded=(self._grounded(output) if caps.search_conditional else None),

            citations=self._citations(output, caps.citation_offsets),
            sources=self._sources(output, caps.page_age),
            queries=self._queries(output),
            thoughts=self._thoughts(output) if caps.readable_reasoning else [],
            units=self._units(output),

            served_model=body.get("model"),
            extra={"n_queries_source": (
                       "usage" if (usage.get("tool_usage") or {}).get("web_search")
                       else "item_count (usage.tool_usage no longer returned)"),
                   "status": body.get("status"),
                   "incomplete_details": body.get("incomplete_details"),
                   "cached_tok": (usage.get("input_tokens_details") or {}
                                  ).get("cached_tokens"),
                   "cache_write_tok": (usage.get("input_tokens_details") or {}
                                       ).get("cache_write_tokens")},
        )

    # ── extraction helpers ───────────────────────────────────────────────────

    @staticmethod
    def _n_queries(usage: dict, output: list) -> int | None:
        """Searches run, preferring the usage figure and falling back to items.

        `usage.tool_usage.web_search.num_requests` was the authoritative count and
        **it is no longer returned** — measured 2026-08-28 across four grounded
        fixtures, where usage carries only input/output/total. When it existed it
        disagreed with the item count (5 requests against 6 items), because items
        batch queries.

        So the fallback is genuinely worse and is used anyway, because a count
        that is approximately right beats None. `n_queries_source` in `extra`
        records which one produced the number, so an analysis comparing counts
        across providers can see that this one is estimated.
        """
        reported = ((usage.get("tool_usage") or {})
                    .get("web_search", {}).get("num_requests"))
        if reported is not None:
            return reported
        items = sum(1 for i in output if i.get("type") == "web_search_call")
        return items or None

    @staticmethod
    def _answer(output: list) -> str:
        """Text from message items. No heuristic — the API says which is which."""
        return "".join(
            c.get("text", "")
            for item in output if item.get("type") == "message"
            for c in (item.get("content") or [])
            if c.get("type") in ("output_text", "text"))

    @staticmethod
    def _thought_text(output: list) -> str | None:
        """Readable reasoning from `summary[].text`.

        Opt-in per call: without `reasoning.summary` in the request, these items
        carry `encrypted_content` and an empty summary array. So None here can
        mean either the provider cannot or the request did not ask — the request
        is recorded alongside every record, which is where that is resolved.
        """
        parts = [s.get("text", "")
                 for item in output if item.get("type") == "reasoning"
                 for s in (item.get("summary") or [])
                 if s.get("text")]
        return "\n\n".join(parts) or None

    @staticmethod
    def _thoughts(output: list) -> list[dict]:
        return [{"unit": i, "text": s.get("text"), "chars": len(s.get("text") or "")}
                for i, item in enumerate(output) if item.get("type") == "reasoning"
                for s in (item.get("summary") or []) if s.get("text")]

    @staticmethod
    def _grounded(output: list) -> bool:
        return any(i.get("type") == "web_search_call" for i in output)

    @staticmethod
    def _citations(output: list, offset_unit: str | None) -> list[dict]:
        """Annotations, with the quote EXTRACTED from the answer.

        The first real test of whether one side-table shape holds two mechanisms.
        Anthropic supplies `cited_text` and no offsets; this provider supplies
        offsets and no quote. Both must produce a populated `quote` column,
        because the analysis rule is compare quotes rather than offsets — so the
        quote is extracted here and read directly there, and neither is a special
        case downstream.

        Offsets index CHARACTERS, and that is recorded per row rather than
        assumed: one provider uses bytes, and a loader applying the wrong
        convention produces spans that are silently wrong and drift further into
        the answer.
        """
        rows = []
        for i, item in enumerate(output):
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                text = content.get("text") or ""
                for ann in content.get("annotations") or []:
                    start, end = ann.get("start_index"), ann.get("end_index")
                    quote = None
                    if isinstance(start, int) and isinstance(end, int):
                        quote = text[start:end]
                    rows.append({
                        "unit": i,
                        "type": ann.get("type"),
                        "offset_unit": offset_unit,
                        "start": start,
                        "end": end,
                        "quote": quote,
                        "url": ann.get("url"),
                        "title": ann.get("title"),
                        "is_redirect": False,
                    })
        return rows

    @staticmethod
    def _sources(output: list, page_age: bool) -> list[dict]:
        """Cited URLs only.

        There is no retrieval set here. `sources_retrieved` is None rather than
        an empty list, because "this provider does not report what it saw" is a
        different claim from "it saw nothing" — and conflating them would make
        this provider look more selective than Anthropic when it is merely
        quieter.
        """
        rows, seen = [], set()
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                for ann in content.get("annotations") or []:
                    url = ann.get("url")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    rows.append({
                        "url": url,
                        "title": ann.get("title"),
                        **({"page_age": None} if page_age else {}),
                        "retrieved": None,   # unknown, not False
                        "cited": True,
                        "tool_use_id": None,
                        "caller": "direct",
                    })
        return rows

    @staticmethod
    def _queries(output: list) -> list[dict]:
        """One row per search call.

        The `action.query` field carries the query text where present. Items
        batch multiple queries, which is why `n_queries` comes from usage and not
        from counting these rows — they are the call structure, not the count.
        """
        rows = []
        for item in output:
            if item.get("type") != "web_search_call":
                continue
            action = item.get("action") or {}
            queries = action.get("queries") or (
                [action["query"]] if action.get("query") else [])
            for q in queries or [None]:
                rows.append({"call_id": item.get("id"), "search": q,
                             "caller": "direct"})
        return rows

    @staticmethod
    def _units(output: list) -> list[dict]:
        rows = []
        for i, item in enumerate(output):
            content = item.get("content") or []
            text = "".join(c.get("text", "") for c in content
                           if isinstance(c, dict))
            enc = item.get("encrypted_content")
            rows.append({
                "unit": i,
                "type": item.get("type"),
                "text": text or None,
                "n_citations": sum(len(c.get("annotations") or [])
                                   for c in content if isinstance(c, dict)) or None,
                "sig_chars": len(enc) if isinstance(enc, str) else None,
                "tool_use_id": item.get("id"),
                "n_results": None,
            })
        return rows

    @staticmethod
    def _n_retrieved(output: list) -> int | None:
        return None
