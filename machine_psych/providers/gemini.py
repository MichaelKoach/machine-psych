"""Gemini.

Third and last, and it differs from the other two in more ways than they differ
from each other. Every one of these was measured, and several contradicted
documentation:

**Byte offsets.** Citation spans index UTF-8 BYTES, not characters — 34/35 byte
matches against 2/35 character matches on an answer containing en-dashes, and
70 of 107 citations wrong under character indexing on a second. The failure is
silent: quotes come back slightly wrong and drift further into the answer.

**Zero-valued fields are omitted, not zeroed.** `start_index` is absent when it is
0, so the FIRST citation of every grounded record has no start at all. Same for
thought and output token counts when nothing was produced. `x.get(k) or 0` is
required and is not defensive coding.

**Grounding is conditional.** Providing the search tool is a permission, not a
condition — a prompt that does not need retrieval comes back ungrounded with the
tool attached. This is the only provider where `grounded` is a meaningful column.

**Thinking cannot be disabled.** `thinking_budget: 0` is rejected. There is no
reasoning-off arm here, which is the one place the union vocabulary genuinely
cannot be honoured.

**Billed and processed input differ by two orders of magnitude.** A grounded call
reported 326 input tokens and processed 32,606. Only this provider separates them.

Endpoint is `/v1beta/interactions`, not `generateContent`. That decision took four
attempts to get right: `total_tool_use_tokens` reads 0 even when search runs, the
`google_search_result` step contains only widget HTML, and a field-name check
returned false positives from inside that HTML. Attribution lives on the
`model_output` step, which an exhaustive path walk found in one pass.
"""

from __future__ import annotations

from urllib.parse import urlparse

from ..capabilities import CAPABILITIES, model_of
from .base import ParsedResponse, Provider

__all__ = ["Gemini"]

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
API_URL = f"{API_BASE}/interactions"

SEARCH_TOOL = {"type": "google_search"}

# Fields belonging to the OTHER endpoint. Sending one is a 400 on every call,
# which reads as a provider outage rather than a config error — that mistake cost
# a full round of "all models unavailable" before anyone looked at the latency.
GENERATECONTENT_ONLY = {
    "include_thoughts": "thinking_summaries",
    "includeThoughts": "thinking_summaries",
    "thinking_budget": "thinking_level",
    "thinkingBudget": "thinking_level",
    "thinkingConfig": "generation_config.thinking_level",
    "maxOutputTokens": "max_output_tokens",
    "safetySettings": None,
    "candidateCount": None,
}


class Gemini(Provider):
    name = "gemini"
    CAPABILITIES = {k: v for k, v in CAPABILITIES.items()
                    if k.startswith("gemini/")}

    # ── transport ────────────────────────────────────────────────────────────

    def url_for(self, body: dict) -> str:
        return API_URL

    def headers(self) -> dict:
        return {"x-goog-api-key": self.require_key(),
                "content-type": "application/json"}

    # ── request ──────────────────────────────────────────────────────────────

    def build(self, prompt, resolved: dict) -> dict:
        """Resolved intent -> Interactions request body.

        `max_output_tokens` defaults to 16384 rather than 4096 because on this
        provider the budget covers thinking AND output, with thinking measured
        taking ~96% of it. A 4096 budget where thinking runs 4,000 leaves nothing
        for the answer, and the truncation reads as a model failure.
        """
        model = resolved.get("model", "gemini-3.7-flash")
        gen: dict = {"max_output_tokens": resolved.get("max_tokens", 16384)}

        reasoning = resolved.get("reasoning")
        if reasoning is not None and reasoning != "off":
            gen["thinking_level"] = reasoning
        # `reasoning: "off"` never reaches here — `caps.supports` rejects it and
        # the resolver raises. Silently omitting the key would produce an arm
        # labelled reasoning-off with thinking on, which is the failure the whole
        # unmet-intent design exists to prevent.

        caps = self.caps(model)
        if caps.readable_reasoning:
            # Free: measured 3,458 thought tokens with summaries visible against
            # 4,098 without. Unlike OpenAI's, which is opt-in per call and
            # returns nothing unless asked, this arrives whenever it is on — so
            # it is on by default and thought text is never silently missing.
            gen["thinking_summaries"] = "auto"

        body: dict = {
            "model": model_of(model),
            "input": prompt,
            # `store` defaults to TRUE here. A research corpus should be
            # self-contained and should not leave client material on a
            # provider's servers.
            "store": False,
            "generation_config": gen,
        }

        if resolved.get("system"):
            body["system_instruction"] = resolved["system"]

        if resolved.get("search"):
            tool = dict(SEARCH_TOOL)
            if isinstance(resolved["search"], dict):
                tool.update(resolved["search"])
            body["tools"] = [tool]

        consumed = {"model", "max_tokens", "system", "reasoning", "search",
                    "repetitions", "exclude", "include", "note"}
        for key, value in resolved.items():
            if key in consumed:
                continue
            if key in GENERATECONTENT_ONLY:
                alt = GENERATECONTENT_ONLY[key]
                raise ValueError(
                    f"{key!r} belongs to generateContent and is rejected by the "
                    f"Interactions endpoint. An unknown field is a 400 on EVERY "
                    f"call, which reads as a provider outage rather than a config "
                    f"error — sub-second failures are the tell. "
                    + (f"Use {alt!r} instead." if alt else "It has no equivalent here."))
            body[key] = value
        return body

    def user_turn(self, text: str) -> dict:
        """A `user_input` STEP, not a role/content turn.

        Sending `{"role": "user", "content": ...}` returns:
            "When using the steps-based API version, use step_list input format
             instead of turn_list."
        """
        return {"type": "user_input", "content": text}

    def passback(self, body: dict) -> list:
        """ALL steps verbatim, not just the output.

        The documentation is explicit: every model-generated step must be resent
        exactly as received, including thought and function-call steps, because
        they carry signatures required to continue. An earlier version appended
        only the model output — which is what the other endpoint tolerates and
        this one does not.
        """
        return list(body.get("steps") or [])

    # ── status ───────────────────────────────────────────────────────────────

    def status(self, body: dict) -> tuple[str, str | None]:
        """Five states.

        Capacity failures arrive as `code: "api_error"` with a prose message
        rather than an HTTP 503, so keying on the code alone means retry never
        fires on the failures it exists for — measured at roughly 1 in 6 on the
        newest model, with a 44-record battery needing 105 calls.
        """
        if not isinstance(body, dict):
            return "error", f"non-dict body: {type(body).__name__}"

        err = body.get("error")
        if err:
            code = err.get("code")
            message = str(err.get("message") or "")
            low = message.lower()
            transient = (
                code == 503
                or "UNAVAILABLE" in message.upper()
                or "high demand" in low
                or "try again later" in low
                or "overloaded" in low
                or "temporarily" in low
                or code in (429, 500, 502, 504)
            )
            return ("unavailable" if transient else "error"), f"{code}: {message}"

        steps = body.get("steps") or []
        answer = self._answer(steps)
        state = body.get("status")

        if state == "completed" and answer:
            return "ok", None
        if answer:
            # Partial text with a non-completed status. Never observed here, and
            # checked for rather than assumed absent — the state is real on one
            # provider and absent on another, so it is a measured property and
            # not a family trait.
            return "truncated", f"status={state}, partial answer"
        if not any(s.get("type") == "model_output" for s in steps):
            return "incomplete", f"no model_output step (status={state})"
        return "incomplete", f"status={state}, no answer text"

    # ── parse ────────────────────────────────────────────────────────────────

    def parse(self, body: dict) -> ParsedResponse:
        steps = body.get("steps") or []
        usage = body.get("usage") or {}
        # Fallback because the model here is what was SERVED, which may be a
        # build not in the table. Parsing must not die over a name; the
        # mismatch is reported by integrity.check_record instead.
        caps = self.caps(body.get("model") or "gemini-3.7-flash", "gemini-3.7-flash")

        answer = self._answer(steps)

        return ParsedResponse(
            answer_text=answer or None,
            answer_chars=len(answer or ""),
            answer_extraction=caps.answer_extraction,

            in_tok_billed=usage.get("total_input_tokens"),
            # `raw_prompt_token`, not the billed figure. This provider is the only
            # one that separates them, and by two orders of magnitude on grounded
            # calls — 326 billed against 32,606 processed. Comparing the billed
            # number across providers measures billing policy, not context volume.
            in_tok_processed=usage.get(caps.in_tok_processed_field),
            out_tok_reported=usage.get("total_output_tokens"),
            # Kept SEPARATE from output here and added to the total, where the
            # other two count it inside output. This is why answer_chars rather
            # than out_tok is the comparable length measure.
            thinking_tok=usage.get("total_thought_tokens"),
            n_queries=self._n_queries(usage, steps),

            thought_text=(self._thought_text(steps)
                          if caps.readable_reasoning else None),
            n_thought_steps=sum(1 for s in steps if s.get("type") == "thought"),
            n_sources_retrieved=(self._n_retrieved(steps)
                                 if caps.retrieval_set else None),
            n_answer_blocks=sum(1 for s in steps
                                if s.get("type") == "model_output"),
            n_process_blocks=0,   # no interleaved process text on this endpoint
            grounded=(self._grounded(steps) if caps.search_conditional else None),

            citations=self._citations(steps, caps.citation_offsets),
            sources=self._sources(steps, caps.page_age),
            queries=self._queries(steps),
            thoughts=self._thoughts(steps) if caps.readable_reasoning else [],
            units=self._units(steps),

            served_model=body.get("model"),
            extra={"status": body.get("status"),
                   "cached_tok": usage.get("total_cached_tokens"),
                   "tool_use_tok": usage.get("total_tool_use_tokens"),
                   "raw_prompt_token": usage.get("raw_prompt_token")},
        )

    # ── extraction helpers ───────────────────────────────────────────────────

    @staticmethod
    def _answer(steps: list) -> str:
        return "".join(
            c.get("text", "")
            for s in steps if s.get("type") == "model_output"
            for c in (s.get("content") or [])
            if isinstance(c, dict) and c.get("text"))

    @staticmethod
    def _thought_text(steps: list) -> str | None:
        """Readable reasoning from `summary`, NOT `content`.

        The key is absent entirely without `thinking_summaries: "auto"`, so a
        loader reading `content` reports zero thought text on every record — which
        looks like the API withholding reasoning and cost a full shakedown round.

        Shape is handled defensively: observed as a string, and this provider's
        step content elsewhere is a list of typed parts.
        """
        parts = []
        for s in steps:
            if s.get("type") != "thought":
                continue
            summary = s.get("summary")
            if isinstance(summary, str):
                parts.append(summary)
            elif isinstance(summary, dict):
                parts.append(summary.get("text") or "")
            elif isinstance(summary, list):
                parts.extend(x if isinstance(x, str) else (x.get("text") or "")
                             for x in summary)
        return "\n\n".join(p for p in parts if p) or None

    @staticmethod
    def _thoughts(steps: list) -> list[dict]:
        rows = []
        for i, s in enumerate(steps):
            if s.get("type") != "thought":
                continue
            text = Gemini._thought_text([s])
            if text:
                rows.append({"unit": i, "text": text, "chars": len(text)})
        return rows

    @staticmethod
    def _grounded(steps: list) -> bool:
        """Did retrieval actually happen.

        The ONLY provider where this is a real question. Providing the tool is a
        permission — a prompt that does not need search comes back ungrounded
        with it attached — so "search on" is not a study condition here and this
        column is how an analysis knows which arm a record landed in.
        """
        return any(s.get("type") == "google_search_call" for s in steps)

    @staticmethod
    def _n_queries(usage: dict, steps: list) -> int | None:
        """From `grounding_tool_count`, which is also the billing unit.

        Never from counting steps. `total_tool_use_tokens` is NOT this figure —
        it reads 0 on search calls and populates only for url_context, which is
        why it looked like an unused field and why reading it as a retrieval
        measure would report zero searches on every grounded record.
        """
        counts = usage.get("grounding_tool_count")
        if isinstance(counts, list) and counts:
            return sum(c.get("count", 0) for c in counts if isinstance(c, dict))
        calls = sum(1 for s in steps if s.get("type") == "google_search_call")
        return calls or None

    @staticmethod
    def _citations(steps: list, offset_unit: str | None) -> list[dict]:
        """Annotations on the model_output step, with BYTE-indexed spans.

        Two traps, both silent:

        **Offsets are UTF-8 bytes.** Slicing with character indices produces
        quotes that are wrong by a character or two and drift further into the
        answer — measured wrong on 70 of 107 citations on one record, with no
        error and no exception.

        **`start_index` is ABSENT when it is 0**, which is the first citation of
        every grounded record. `ann.get("start_index") or 0` is required, and
        works here only because absent and 0 mean the same thing.
        """
        rows = []
        for i, step in enumerate(steps):
            if step.get("type") != "model_output":
                continue
            offset = 0
            for content in step.get("content") or []:
                if not isinstance(content, dict):
                    continue
                part = content.get("text") or ""
                part_bytes = part.encode("utf-8")
                for ann in content.get("annotations") or []:
                    start = ann.get("start_index") or 0
                    end = ann.get("end_index")
                    quote = None
                    if isinstance(end, int):
                        quote = part_bytes[start:end].decode("utf-8", errors="replace")
                    url = ann.get("url")
                    rows.append({
                        "unit": i,
                        "type": ann.get("type"),
                        "offset_unit": offset_unit,
                        # rebased onto the concatenated answer, so spans from
                        # different content parts do not collide
                        "start": start + offset,
                        "end": (end + offset) if isinstance(end, int) else None,
                        "quote": quote,
                        "url": url,
                        # `title` carries the real domain; the URL is a redirect
                        # whose netloc is always Google's.
                        "title": ann.get("title"),
                        "is_redirect": bool(url and "vertexaisearch" in url),
                    })
                offset += len(part_bytes)
        return rows

    @staticmethod
    def _sources(steps: list, page_age: bool) -> list[dict]:
        """Cited URLs only, and every one is a redirect.

        The URLs resolve to real pages, and the redirect yields the address even
        when the fetch is blocked — so resolution and retrieval are separate
        steps for the scraping layer. Roughly half of vendor marketing pages
        block bots, which is exactly the population this research targets.
        """
        rows, seen = [], set()
        for step in steps:
            if step.get("type") != "model_output":
                continue
            for content in step.get("content") or []:
                if not isinstance(content, dict):
                    continue
                for ann in content.get("annotations") or []:
                    url = ann.get("url")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    rows.append({
                        "url": url,
                        "title": ann.get("title"),
                        **({"page_age": None} if page_age else {}),
                        "retrieved": None,
                        "cited": True,
                        "tool_use_id": None,
                        "caller": "direct",
                        "is_redirect": "vertexaisearch" in url,
                        "redirect_host": urlparse(url).netloc or None,
                    })
        return rows

    @staticmethod
    def _queries(steps: list) -> list[dict]:
        """One row per query, linked to its call.

        Search is EXPLICIT here — `google_search_call` steps carrying the query
        list, matched to `google_search_result` by `call_id` — where the other
        endpoint buries it in a metadata sidecar.
        """
        rows = []
        for step in steps:
            if step.get("type") != "google_search_call":
                continue
            for q in (step.get("arguments") or {}).get("queries") or [None]:
                rows.append({"call_id": step.get("id"), "search": q,
                             "caller": step.get("search_type", "direct")})
        return rows

    @staticmethod
    def _units(steps: list) -> list[dict]:
        """Provider-native steps.

        `search_suggestions` is stripped: ~4,700 characters of Google's widget
        markup per grounded record that no analysis reads, and its presence also
        triggers a display obligation in Google's terms — worth a deliberate
        answer for a research corpus rather than an oversight.
        """
        rows = []
        for i, s in enumerate(steps):
            content = s.get("content") or []
            text = "".join(c.get("text", "") for c in content
                           if isinstance(c, dict))
            sig = s.get("signature")
            rows.append({
                "unit": i,
                "type": s.get("type"),
                "text": text or None,
                "n_citations": sum(len(c.get("annotations") or [])
                                   for c in content if isinstance(c, dict)) or None,
                "sig_chars": len(sig) if isinstance(sig, str) else None,
                "tool_use_id": s.get("call_id") or s.get("id"),
                "n_results": (len(s.get("result") or [])
                              if s.get("type") == "google_search_result" else None),
            })
        return rows

    @staticmethod
    def _n_retrieved(steps: list) -> int | None:
        return None
