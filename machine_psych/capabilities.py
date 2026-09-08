"""Per-model capabilities, measured rather than declared.

Every field here was established by sending requests and reading responses. None
of it comes from documentation, which has been wrong on `thinking_level`,
`reasoning.context`, `prompt_cache_retention` and the sampling parameters.

Two design choices, each borrowed from a library that got it right:

**Keyed per model, not per provider** (from LiteLLM). Fable 5 cannot disable
thinking while Sonnet 5 can; that difference is invisible at provider
granularity, and a capability table that cannot see it will silently produce an
arm whose condition is a fiction.

**Completeness enforced at class definition** (from any-llm). A provider cannot
be added without answering every capability question. Silence is not an option
the type system permits.

There is no maintenance process behind this data. LiteLLM keeps ~3,000 entries
and a team watching changelogs; this keeps a dozen and a person who notices when
something breaks. Hence `measured_on`, hence `UnknownModelError` rather than a
family default, and hence `compare_capabilities()` — which sends bogus values and
diffs the API's own enumerations against the recorded ones, the same technique
that recovered them in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

__all__ = [
    "CAPABILITIES",
    "CONSUMED_BY",
    "DELIBERATELY_UNREAD",
    "ENUMS",
    "REQUIRED_FIELDS",
    "ModelCaps",
    "UnknownModelError",
    "caps_for",
    "known_models",
    "known_providers",
    "model_of",
    "provider_of",
]


class UnknownModelError(LookupError):
    """A model absent from CAPABILITIES.

    Deliberately fatal. A defaulted capability produces an arm whose condition is
    a guess, and a guessed condition is worse than a failed run because it looks
    like data. The message names what is known and what a new entry needs.

    LookupError rather than KeyError: KeyError's __str__ is repr() of the
    argument, which turns a carefully formatted multi-line message into one
    escaped line. The message is the point of this exception.
    """


@dataclass(frozen=True)
class ModelCaps:
    """What a model can actually be asked to do.

    Every field is required. `Provider.__init_subclass__` asserts that each entry
    in a provider's CAPABILITIES is a fully populated ModelCaps, so adding a
    provider forces answering all of these rather than inheriting a default.
    """

    # ── reasoning ────────────────────────────────────────────────────────────
    reasoning_off: bool
    """Can thinking be disabled at all.

    False on Gemini — `thinking_budget: 0` is rejected and there is no off
    switch, so a thinking on/off manipulation is impossible there. Also False on
    Fable 5, which is always-on adaptive.
    """

    reasoning_levels: tuple[str, ...]
    """Valid intensity values, in the provider's own vocabulary."""

    readable_reasoning: bool
    """Does the provider return reasoning text a human can read.

    **True on Gemini AND OpenAI as of 2026-08-27.** It was Gemini-only when the
    characterisations were written, and that claim survived into the design
    document, the project notes and a Tier 3 column justified by one provider
    having something the others could not do.

    OpenAI's `reasoning.summary: "detailed"` was validated as a parameter and
    returned EMPTY arrays in blocking mode at the time. It populates now:

        "summary": [{"type": "summary_text",
                     "text": "**Resolving ambiguity in calculations**\n\n
                              I'm working through some financial..."}]

    Same register as Gemini's — headed, narrated, a summary rather than a raw
    trace. Both arrive under a `summary` key; the surrounding structure differs.

    Anthropic remains the exception: thinking blocks carry a signature and an
    empty `thinking` string.

    The parser gates on this field rather than hardcoding it, which is why this
    became a one-line edit instead of a rewrite.
    """

    # ── output shaping ───────────────────────────────────────────────────────
    verbosity: bool
    """Is there an output-length control beyond max_tokens.

    True only on OpenAI. Measured swinging output 5.7x low-to-high and
    restructuring the answer with it — one markdown header at low, thirteen at
    high — so it is a stronger lever than reasoning effort and a confound in any
    cross-condition length comparison.
    """

    combined_token_budget: bool
    """Does max_tokens cover thinking AND output together.

    True on Gemini, where thinking has been measured taking ~96% of the budget. A
    4096 budget on a prompt where thinking runs 4,000 tokens leaves nothing for
    the answer, and the truncation reads as a model failure rather than a config
    one.
    """

    # ── search ───────────────────────────────────────────────────────────────
    search: bool
    search_conditional: bool
    """Is providing the tool a permission rather than a condition.

    True on Gemini: a prompt that does not need retrieval comes back ungrounded
    with the tool attached. So "search on" is not a study condition there, and
    `grounded` must be recorded per record to know which arm a call landed in.
    """

    domain_filter: str | None
    """How a corpus can be constrained: 'param', 'prompt', or None.

    'param' on Anthropic and OpenAI, but they behave oppositely — Anthropic's
    constrains the *query* and thrashes (66-143K input against ~20K open web),
    OpenAI's terminates early and costs 61% less. Gemini has no parameter; the
    constraint goes in the prompt text, which makes it part of the manipulation
    rather than beside it.
    """

    max_tool_calls: bool
    """Can the number of searches be capped. True only on OpenAI, where it is a
    real budget rather than a truncation — capping at 1 measured 65-77% cheaper
    while still producing a complete answer."""

    page_age: bool
    """Does a retrieved result report how old the page is.

    True only on Anthropic, which returns `page_age` on every
    `web_search_result` — 'May 25, 2026' and similar. How fresh the sources
    behind a recommendation are is a direct question for this research and no
    other provider answers it.
    """

    retrieval_set: bool
    """Does the response expose URLs that were retrieved but NOT cited.

    True only on Anthropic, which returns the whole retrieval set inline — 134
    URLs and 414KB of content in one measured response. This is the only provider
    on which "what distinguishes cited from uncited sources" is answerable, and
    that question is what the scraping layer exists for.
    """

    # ── attribution ──────────────────────────────────────────────────────────
    citation_offsets: str | None
    """How a citation locates itself: 'char', 'byte', 'quoted', or None.

    Three genuinely different mechanisms, and the difference decides how the
    quote is recovered:

    - **'byte'** (Gemini) — offsets index UTF-8 BYTES. Measured 34/35 byte
      matches against 2/35 character matches on an answer containing en-dashes.
      Character indexing produces spans that are silently wrong and drift further
      into the answer, with no error.
    - **'char'** (OpenAI) — offsets index characters.
    - **'quoted'** (Anthropic) — no offsets at all. The citation carries
      `cited_text` directly, plus `url` and `title`.

    'quoted' is the easiest to work with and was mis-recorded as None until
    fixture collection on 2026-08-27 showed citations present on every grounded
    text block. It also happens to be what the analysis layer wants: the rule is
    compare quotes, not offsets, and this provider supplies the quote without any
    arithmetic to get wrong.
    """

    # ── extraction ───────────────────────────────────────────────────────────
    answer_extraction: str
    """'structural' or 'heuristic'.

    'heuristic' only on Anthropic, where the answer is "text blocks after the
    last tool-use block" — an inference that can be wrong in ways that are hard
    to notice downstream. OpenAI's `message` item and Gemini's `model_output`
    step are told to us.
    """

    cross_turn: str
    """'researches' | 'front_loads' | 'reuses_context'.

    Recorded, never acted on. This once gated multi-turn in the Anthropic
    validator, which was a provider mechanism promoted to a format constraint.
    All three behave differently and none of them is wrong.
    """

    # ── sampling ─────────────────────────────────────────────────────────────
    sampling_meaningful: bool
    """Does temperature/top_p actually do anything. False everywhere so far."""

    sampling_silently_ignored: bool
    """Does the API ACCEPT a dead sampling parameter instead of rejecting it.

    The dangerous case, and true only on Gemini. Anthropic and OpenAI return 400,
    so the API catches the mistake for free. Gemini takes the parameter and
    discards it — measured, `temperature: 0.0` produced six distinct answers
    across six repetitions naming three different products, indistinguishable
    from 2.0.

    This is why the client-side guard exists at all: it is the one place where
    "never duplicate the API's validation" does not apply, because there is no
    API validation to duplicate. Recorded separately from `sampling_meaningful`
    because the two failure modes need different error messages, and conflating
    them produced a guard that told OpenAI users their rejected parameter was
    being silently ignored.
    """

    # ── token accounting ─────────────────────────────────────────────────────
    in_tok_processed_field: str
    """Which usage field carries the tokens the model actually processed.

    `raw_prompt_token` on Gemini, `input_tokens` elsewhere. Gemini is the only
    provider that separates billed from processed: a grounded call reported 326
    input tokens and 32,606 processed. Comparing the billed figure across
    providers measures billing policy, not context volume.
    """

    thinking_in_output_tokens: bool
    """Are thinking tokens counted INSIDE the reported output count.

    True on Anthropic and OpenAI, False on Gemini, which keeps
    `thoughtsTokenCount` separate and adds it to the total. This is why
    `answer_chars` rather than `out_tok` is the comparable length measure.
    """

    tokens_per_query: int
    """Measured retrieved tokens per search. The cost estimator's constant.

    ~11,600 Anthropic, ~11,100 OpenAI, ~4,100 Gemini. Anthropic and OpenAI land
    within 5% of each other, so the large difference in their total input is how
    many searches each chose to run, not how much each retrieves. n=1 per
    provider on one prompt — direction, not precision.
    """

    # ── provenance ───────────────────────────────────────────────────────────
    measured_on: str
    """ISO date. Not a correctness flag, a staleness one. A year-old entry is not
    wrong, but it is worth seeing before a battery relies on it."""

    notes: str = ""
    """Anything a study designer should know that has no field of its own."""

    def supports(self, intent: str, value: Any = None) -> bool:
        """Can this model honour an intent AT THIS VALUE.

        The value matters and the key alone does not: `reasoning` is supported on
        Gemini, `reasoning: "off"` is not. Checking only the key is how an unmet
        intent becomes a silently mislabelled arm.
        """
        if intent == "reasoning":
            if value in (None, "off") and not self.reasoning_off:
                return value is None
            if value is None:
                return True
            # CHECK THE LEVEL AGAINST THIS MODEL, not just the off case. An
            # earlier version returned True for every non-off value, so
            # `reasoning_levels` was read only by `why_unsupported` — a field
            # that shaped an error message and never a decision.
            #
            # The cost was measured: the shakedown sent `reasoning: minimal` to
            # a model that rejects it, and the spec layer passed it through to
            # be refused by the API one call and thirty seconds later. The
            # levels are PER-MODEL and the schema does not know it.
            usable = set(self.reasoning_levels)
            if "off" in usable or self.reasoning_off:
                usable.add("off")
                usable.add("none")
            return value in usable
        if intent == "verbosity":
            return self.verbosity
        if intent == "search":
            return self.search or not value
        if intent == "max_tool_calls":
            return self.max_tool_calls or value is None
        if intent in ("temperature", "top_p", "top_k"):
            return self.sampling_meaningful
        # max_tokens, system, repetitions and anything unlisted are universal
        return True

    def why_unsupported(self, intent: str, value: Any = None) -> str:
        """A sentence explaining a False from supports(), for the error message.

        Worth its own method because the reason is what tells a study designer
        what to do instead — and 'unsupported' alone has, in practice, led to
        someone assuming the harness was broken.
        """
        if intent == "reasoning" and value not in (None, "off"):
            return (f"`reasoning: {value!r}` is not accepted by this model. "
                    f"It takes {list(self.reasoning_levels)}. Levels are "
                    f"per-model — the API schema lists values that individual "
                    f"models reject.")
        if intent == "reasoning" and value in (None, "off"):
            usable = [lv for lv in self.reasoning_levels if lv != "off"]
            return (f"thinking cannot be disabled on this model, so there is no "
                    f"reasoning-off arm to run. Levels that do work: "
                    f"{', '.join(usable)}")
        if intent == "verbosity":
            return "no output-length control beyond max_tokens on this provider"
        if intent == "max_tool_calls":
            return "search count cannot be capped on this provider"
        if intent in ("temperature", "top_p", "top_k"):
            if self.sampling_silently_ignored:
                return ("accepted by this API and silently ignored — a sweep "
                        "would produce arms that look different and are "
                        "identical, and nothing downstream would object")
            return ("rejected by this API on current models; sampling controls "
                    "were removed rather than deprecated")
        if intent == "search":
            return "this model has no search tool"
        return f"{intent!r} is not supported at {value!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# The measured data.
#
# `provider/model` keys throughout. Model aliases are NOT keys: `gpt-5.6`
# resolves to `-sol` and `gemini-3.7-flash` has three variants that name
# different companies on the same prompt, so an alias would make the served model
# an uncontrolled variable.
# ═══════════════════════════════════════════════════════════════════════════════

_ANTHROPIC_COMMON = {
    # Re-measured 2026-08-27 during fixture collection, and the surface had moved.
    # `thinking: {"type": "enabled", "budget_tokens": N}` is now REJECTED on
    # claude-sonnet-5 — "not supported for this model. Use thinking.type.adaptive
    # and output_config.effort" — so the budget-token mapping every earlier note
    # assumed no longer exists. Intensity is `output_config.effort`, whose
    # enumeration was recovered by bogus value and is low|medium|high|xhigh|max.
    #
    # Note this is the SAME enumeration OpenAI uses minus `none`. Two providers
    # that were previously incomparable — budget tokens against effort levels —
    # now share a vocabulary. Whether the levels MEAN the same thing is a
    # separate question and remains an asserted mapping, not a measured one.
    "reasoning_levels": ("off", "low", "medium", "high", "xhigh", "max"),
    "readable_reasoning": False,
    "verbosity": False,
    "combined_token_budget": False,
    "search": True,
    "search_conditional": False,
    "domain_filter": "param",
    "max_tool_calls": False,
    "page_age": True,
    "retrieval_set": True,
    "citation_offsets": "quoted",
    "answer_extraction": "heuristic",
    "cross_turn": "researches",
    "sampling_meaningful": False,
    "sampling_silently_ignored": False,
    "in_tok_processed_field": "input_tokens",
    "thinking_in_output_tokens": True,
    "tokens_per_query": 11_600,
    "measured_on": "2026-08-27",   # family default; per-model overrides below
}

_OPENAI_COMMON = {
    "reasoning_levels": ("none", "low", "medium", "high", "xhigh", "max"),
    # Requires `reasoning.summary` to be requested; without it the reasoning
    # items carry `encrypted_content` only. Unlike Gemini's, where the summary
    # arrives whenever thinking_summaries is set, this one is opt-in per call.
    "readable_reasoning": True,
    "verbosity": True,
    "combined_token_budget": False,
    "search": True,
    "search_conditional": False,
    "domain_filter": "param",
    "max_tool_calls": True,
    "page_age": False,
    "retrieval_set": False,
    "citation_offsets": "char",
    "answer_extraction": "structural",
    "cross_turn": "front_loads",
    "sampling_meaningful": False,
    "sampling_silently_ignored": False,
    "in_tok_processed_field": "input_tokens",
    "thinking_in_output_tokens": True,
    "tokens_per_query": 11_100,
    "measured_on": "2026-08-27",   # family default; per-model overrides below
}

_GEMINI_COMMON = {
    "reasoning_off": False,
    # SCHEMA default. Per-model overrides below — `minimal` is rejected by
    # 3.7-flash and 3.8-flash and accepted by 3.5-flash, measured 2026-09-05.
    "reasoning_levels": ("minimal", "low", "medium", "high"),
    "readable_reasoning": True,
    "verbosity": False,
    "combined_token_budget": True,
    "search": True,
    "search_conditional": True,
    "domain_filter": "prompt",
    "max_tool_calls": False,
    "page_age": False,
    "retrieval_set": False,
    "citation_offsets": "byte",
    "answer_extraction": "structural",
    "cross_turn": "reuses_context",
    "sampling_meaningful": False,
    "sampling_silently_ignored": True,
    "in_tok_processed_field": "raw_prompt_token",
    "thinking_in_output_tokens": False,
    "tokens_per_query": 4_100,
    "measured_on": "2026-08-24",   # family default; per-model overrides below
}

CAPABILITIES: dict[str, ModelCaps] = {
    # ── Anthropic ────────────────────────────────────────────────────────────
    # The three models below carry their own `measured_on`. Everything else
    # inherits its family's date, which is older — and that asymmetry is the
    # point. An earlier edit set every model to the fixture-collection date,
    # which claimed currency for seven models that were never re-measured and
    # inverted the exact signal this field exists to give.
    "anthropic/claude-sonnet-5": ModelCaps(
        reasoning_off=True, **{**_ANTHROPIC_COMMON, "measured_on": "2026-09-05"},
        notes="Adaptive thinking is ON BY DEFAULT — a call sending no thinking key "
              "returned three thinking blocks and 823 thinking tokens. "
              "`thinking: {type: disabled}` still works, so an off arm exists; "
              "`thinking: {type: enabled, budget_tokens: N}` is rejected."),
    "anthropic/claude-opus-5": ModelCaps(reasoning_off=True, **_ANTHROPIC_COMMON),
    "anthropic/claude-fable-5": ModelCaps(
        reasoning_off=False, **_ANTHROPIC_COMMON,
        notes="Adaptive thinking always on and `thinking: disabled` returns 400 — "
              "which is what distinguishes it from Sonnet 5, where thinking is "
              "also on by default but CAN be disabled. The only model likely to "
              "produce a populated `stop_details`, via its refusal classifier — "
              "still never observed."),

    # ── OpenAI ───────────────────────────────────────────────────────────────
    # The three 3.7-era variants are NOT interchangeable. Asked which companies
    # resemble one vendor, they named substantially different sets and only one
    # company appeared in all of them. Pinning is required, not preferred.
    "openai/gpt-5.6-sol": ModelCaps(
        reasoning_off=True, **{**_OPENAI_COMMON, "measured_on": "2026-09-05"},
        notes="2026-09-05, FIRST LIVE RUN: this provider DOES truncate with a "
              "PARTIAL answer — 1,071 and 1,169 characters on a two-turn "
              "ladder at max_output_tokens 200. That contradicts a five-budget "
              "fixture sweep which found zero message items on every incomplete "
              "arm and was written up as 'emits the message item complete or "
              "not at all'. The difference is probably single-turn versus a "
              "ladder; either way the earlier claim does not survive as stated. "
              "`gpt-5.6` resolves here — confirmed 2026-08-27. `effort: none` "
              "gives 0 reasoning tokens, so the off arm is real. Whether the "
              "levels separate is UNMEASURED: on one arithmetic prompt low gave "
              "59 reasoning tokens and high gave 49, which is backwards at n=1 "
              "and probably says the problem was too easy rather than that the "
              "parameter does nothing."),
    "openai/gpt-5.6-terra": ModelCaps(
        reasoning_off=True, **_OPENAI_COMMON,
        notes="Spends about a third of sol's reasoning for 55% the output."),
    "openai/gpt-5.6-luna": ModelCaps(reasoning_off=True, **_OPENAI_COMMON),
    "openai/gpt-5.5": ModelCaps(
        reasoning_off=True,
        # MEASURED 2026-09-05: "Supported values are: 'none', 'low', 'medium',
        # 'high', and 'xhigh'" — no `max`, which the shared block claims.
        **{**_OPENAI_COMMON, "measured_on": "2026-09-05",
           "reasoning_levels": ("none", "low", "medium", "high", "xhigh")},
        notes="Itself an alias for gpt-5.5-2026-04-23; kept in alias form because "
              "it stands in for what a user would actually hit. `served_model` "
              "records the resolution."),

    # ── Gemini ───────────────────────────────────────────────────────────────
    "gemini/gemini-3.7-flash": ModelCaps(
        **{**_GEMINI_COMMON, "measured_on": "2026-09-05",
           # MEASURED, and the schema disagrees. "'minimal' is not a supported
           # thinking level for this model. Allowed values are: medium, low,
           # high." The shakedown's study D sent `reasoning: minimal` here and
           # would have been rejected — it failed on a different bug first, so
           # this was never discovered by running it.
           "reasoning_levels": ("low", "medium", "high")},
        notes="Google's current default. Measured a genuine transient failure "
              "rate around 1 in 6 — retry is mandatory, and a 44-record battery "
              "needed 105 calls."),
    "gemini/gemini-3.6-flash": ModelCaps(**_GEMINI_COMMON),
    "gemini/gemini-3.1-pro-preview": ModelCaps(
        **_GEMINI_COMMON,
        notes="The only Pro. There is no Pro at the current tier — the lineup "
              "splits by release cadence rather than capability."),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Enumerations recovered by sending a bogus value and reading the rejection.
#
# Cheaper and more current than documentation, which was wrong on several. These
# are not used for validation — the API catches bad values and names the field —
# they are the baseline `compare_capabilities()` diffs against, which is what
# would have caught `reasoning.context` drifting before a spec was written
# against the wrong values.
# ═══════════════════════════════════════════════════════════════════════════════

# TODO (drift/README.md): `compare_capabilities(model)` should send a
# deliberately bogus value for each parameter below, read the rejection, and diff
# the returned valid set against what is recorded here. That is the technique that
# recovered these in the first place, and it costs one rejected call per parameter.
#
# It would have caught `output_config.effort` replacing `thinking.budget_tokens`
# before a spec was written against the old one, and `reasoning.context` drifting
# before its wrong values reached a document. Four provider surfaces moved in
# roughly a week; none broke loudly.
# ═══════════════════════════════════════════════════════════════════════════════
# WHAT THE API SCHEMA VALIDATES — not what any given model accepts.
#
# Measured 2026-09-05, and the distinction was expensive to learn. A BOGUS value
# returns the schema-wide list, identical across every model. A PLAUSIBLE value —
# one real elsewhere but maybe not here — returns the per-model set, and those
# differ:
#
#     bogus on any OpenAI model:  none, minimal, low, medium, high, xhigh, max
#     `minimal` on gpt-5.6-sol:   none, low, medium, high, xhigh, max
#     `minimal` on gpt-5.5:       none, low, medium, high, xhigh
#     `minimal` on gpt-6-astra:   low, medium, high, xhigh, max   ← no off arm
#
# Every enum in this table was recovered by the bogus-value technique, so every
# one describes the SCHEMA. Three of seven parameters probed turned out to have
# per-model restrictions the technique could not see.
#
# Per-model truth lives in `ModelCaps.reasoning_levels`. This block is kept
# because it is still what the API validates against, and a value absent HERE is
# rejected everywhere — but a value present here may be rejected by any
# particular model.
ENUMS: dict[str, dict[str, tuple[str, ...]]] = {
    "openai": {
        # NOT ALL 429s ARE TRANSIENT. Measured 2026-08-27: an exhausted credit
        # balance returns HTTP 429 with `type: insufficient_quota` and
        # `code: credit_balance_exhausted`. A classifier keyed on the status code
        # alone would retry it — five attempts per record across a battery, hours
        # wasted, never succeeding. The error TYPE is the discriminator, not the
        # code, and this was learned by writing exactly that retry loop by hand.
        "error.type": ("insufficient_quota", "rate_limit_exceeded",
                       "invalid_request_error", "server_error"),
        # SCHEMA-WIDE, NOT PER-MODEL. Measured 2026-09-05: `minimal` appears in
        # this list and is rejected by every model tested — "not supported with
        # the 'gpt-5.6-sol' model", and likewise for gpt-5.5 and gpt-6-astra.
        # The per-model sets are in `ModelCaps.reasoning_levels` and they differ.
        "reasoning.effort": ("none", "minimal", "low", "medium", "high",
                             "xhigh", "max"),
        "reasoning.mode": ("standard", "pro"),
        "reasoning.context": ("auto", "current_turn", "all_turns"),
        "reasoning.summary": ("concise", "detailed", "auto"),
        "text.verbosity": ("low", "medium", "high"),
    },
    "gemini": {
        # Interactions only; `generateContent` rejects thinking_level entirely.
        # SCHEMA-WIDE. `minimal` is accepted by gemini-3.5-flash and rejected by
        # 3.7-flash and 3.8-flash — "not a supported thinking level for this
        # model. Allowed values are: medium, low, high". Note that message lists
        # values in a DIFFERENT ORDER on each call, so any extraction must sort.
        "generation_config.thinking_level": ("minimal", "low", "medium", "high"),
        "generation_config.thinking_summaries": ("auto", "none"),
        "role": ("user", "model"),
    },
    "anthropic": {
        # Recovered 2026-08-27 by bogus value: "Input should be 'low', 'medium',
        # 'high', 'xhigh' or 'max'". This parameter did not exist in any earlier
        # note — the previous control was thinking.budget_tokens, now rejected.
        "output_config.effort": ("low", "medium", "high", "xhigh", "max"),

        # `enabled` is valid and REQUIRES budget_tokens — measured 2026-09-05 on
        # sonnet-5, opus-5 and opus-4-8, all returning
        # "thinking.enabled.budget_tokens: Field required".
        #
        # So the OLD control still exists. Appendix F recorded `thinking.type.
        # enabled` as rejected on 2026-08-27, and that finding drove the whole
        # migration to `output_config.effort` — but the rejection was for sending
        # `enabled` WITHOUT its required companion field, not for the tag being
        # withdrawn. Both controls are live.
        #
        # `output_config.effort` remains the one in use: it needs no companion
        # field and its enum matches OpenAI's minus `none`.
        "thinking.type": ("adaptive", "disabled", "enabled"),
    },
}


# Fields the API reports that are knowingly not mapped to a column. Every numeric
# usage field must be either mapped or listed here with a reason —
# `test_available_measures_read` enforces it, because an unread measure is
# invisible to review: there is nothing on the page to look at.
DELIBERATELY_UNREAD: dict[str, str] = {
    "anthropic/inference_geo": "routing metadata; no research use identified",
    "anthropic/service_tier": "billing tier; constant across observed calls",
    "anthropic/cache_creation.ephemeral_1h_input_tokens":
        "cache granularity; the 5m/1h split has no bearing on any question so far",
    "gemini/input_tokens_by_modality":
        "text-only corpora, so the modality breakdown is always a single text row; "
        "revisit if images or audio ever enter a battery",
    "gemini/tool_use_tokens_by_modality":
        "same as input_tokens_by_modality — one text row on every observed call. "
        "Note this field DOES populate for url_context (51 tokens measured) and not "
        "for google_search, so it is tool-specific rather than a general retrieval "
        "measure, which is why it looked empty",
    "openai/input_tokens_details.cache_write_tokens":
        "recorded via cached_tok; the write/read split is provider-side caching, "
        "which contaminates sequential cost comparison but is not a manipulation",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Lookup
# ═══════════════════════════════════════════════════════════════════════════════

def provider_of(model: str) -> str:
    """'anthropic/claude-sonnet-5' -> 'anthropic'."""
    if "/" not in model:
        raise UnknownModelError(
            f"{model!r} has no provider prefix. Models are named "
            f"'provider/model' so the provider is explicit rather than inferred "
            f"from a string pattern. Known: {sorted(known_models())}")
    return model.split("/", 1)[0]


def model_of(model: str) -> str:
    """'anthropic/claude-sonnet-5' -> 'claude-sonnet-5'."""
    return model.split("/", 1)[1] if "/" in model else model


def known_providers() -> set[str]:
    """Provider names, derived from the model keys rather than listed.

    Exists because `spec.resolve` hardcoded `("anthropic", "openai", "gemini")`
    to recognise escape-hatch keys, so a fourth provider's hatch would have been
    rejected as an unrecognised intent — silently, in the one place that would
    have needed editing and gave no sign of it.
    """
    return {k.split("/", 1)[0] for k in CAPABILITIES}


def known_models(provider: str | None = None) -> list[str]:
    """Every model with measured capabilities, optionally for one provider.

    Deliberately not "every model the provider offers" — a model absent here
    cannot be dispatched to, because a guessed capability produces an arm whose
    condition is a fiction.
    """
    return sorted(k for k in CAPABILITIES
                  if provider is None or k.startswith(f"{provider}/"))


def caps_for(model: str) -> ModelCaps:
    """Capabilities for a 'provider/model' key. Raises on anything unknown.

    No family defaults, deliberately. Guessing a capability produces an arm whose
    condition is a fiction, and that failure is silent — it looks like data.
    """
    try:
        return CAPABILITIES[model]
    except KeyError:
        prov = model.split("/", 1)[0] if "/" in model else None
        siblings = known_models(prov)
        # Two branches, because one message for both cases lied. With an UNKNOWN
        # provider, `siblings` is empty and the fallback printed every model from
        # every provider under the heading "Known for perplexity" — a misleading
        # message on the exact path a new provider takes.
        if siblings:
            known = f"  Known for {prov}: {siblings}\n"
        else:
            known = (f"  No provider {prov!r} — known providers: "
                     f"{sorted(known_providers())}\n"
                     f"  A new provider needs a Provider subclass AND "
                     f"CAPABILITIES entries; see drift/README.md.\n")
        raise UnknownModelError(
            f"{model!r} is not in CAPABILITIES.\n"
            + known
            + "  Adding one is deliberate, not automatic. Four probes establish "
              "an entry: reasoning off, a sampling parameter, a grounded call, "
              "and a small max_tokens on a long prompt — plus a bogus value per "
              "enumerated parameter to record the enums. See drift/README.md; "
              "the script that automates this is planned, not built.\n"
              "  A defaulted capability would produce an arm whose condition is "
              "a guess, which is worse than this error because it looks like "
              "data."
        ) from None


REQUIRED_FIELDS = tuple(f.name for f in fields(ModelCaps) if f.name != "notes")


def _validate_capabilities() -> None:
    """Every entry complete, checked ONCE at import, where the data lives.

    This was a `__init_subclass__` hook on the Provider base class. It ran per
    subclass, over a filtered VIEW of this same dict — the objects are literally
    identical — so it validated data it did not own, three times, in a file that
    does not define it.

    `ModelCaps` is a dataclass with twenty required fields, so presence is
    guaranteed by the type. What actually needs checking is that nothing was left
    at a placeholder, and that keys carry their provider prefix.

    `None` is LEGITIMATE for `domain_filter` and `citation_offsets` — Anthropic
    has no citation offsets at all, and "this provider cannot" is exactly what
    None means. An earlier version treated None as missing and rejected every
    valid Anthropic entry.
    """
    nullable = {"domain_filter", "citation_offsets"}
    for key, caps in CAPABILITIES.items():
        if "/" not in key:
            raise TypeError(
                f"CAPABILITIES key {key!r} is not 'provider/model' — the "
                f"provider is explicit rather than inferred.")
        unset = [f for f in REQUIRED_FIELDS
                 if f not in nullable and getattr(caps, f, None) is None]
        if unset:
            raise TypeError(
                f"CAPABILITIES[{key!r}] leaves {unset} unset. Every field is "
                f"required; there are no defaults, because an unanswered "
                f"capability question becomes a silent assumption.")


_validate_capabilities()


# ═══════════════════════════════════════════════════════════════════════════════
# Who consumes each capability.
#
# Most of these have no reader yet — the modules that will use them are not built.
# That is fine and expected; what is NOT fine is a field quietly staying unread
# forever, which is how `EXPLICIT_DEFAULTS` and the recovered enumerations became
# decorative constants in the modules this replaces.
#
# So every field names the module that owes it a consumer, and
# `test_capability_fields_consumed` asserts that once a module exists, it reads
# what it was assigned. Two fields are documentation by design and say so.
# ═══════════════════════════════════════════════════════════════════════════════

CONSUMED_BY: dict[str, str] = {
    # NOTE: several of these are read by capabilities.supports(), not by the
    # module that CALLS supports(). An earlier version named spec.resolve as the
    # owner of every field the unmet check consults, and the consumer test caught
    # it the moment spec.py existed — the owner is whoever reads the attribute,
    # not whoever benefits from it being read.
    "reasoning_off":             "capabilities.supports — the unmet check",
    "reasoning_levels":          "capabilities.why_unsupported — naming the levels that DO work",
    "readable_reasoning":        "providers.*.parse — whether to extract thought_text",
    "verbosity":                 "capabilities.supports — the unmet check",
    "combined_token_budget":     "spec._guard_token_budget",
    "search":                    "capabilities.supports — the unmet check",
    "search_conditional":        "providers.*.parse — whether `grounded` is meaningful",
    "domain_filter":             "PENDING — providers.*.build, once a domain-constraint "
                                 "intent exists. Recorded now because the three providers "
                                 "differ in kind (param, param, prompt-only) and that "
                                 "shapes whether it can be an intent at all.",
    "max_tool_calls":            "capabilities.supports — the unmet check",
    "page_age":                  "providers.*.parse — whether the sources table carries page_age",
    "retrieval_set":             "providers.*.parse — whether n_sources_retrieved exists",
    "citation_offsets":          "providers.*.parse — byte vs character span extraction",
    "answer_extraction":         "providers.*.parse — sets the ParsedResponse field",
    "sampling_meaningful":       "spec._guard_inert",
    "sampling_silently_ignored": "capabilities.why_unsupported — which failure mode to describe",
    "in_tok_processed_field":    "providers.*.parse — which usage field to read",
    "thinking_in_output_tokens": "analysis.normalize — why out_tok is not comparable",
    "tokens_per_query":          "spec._estimate — cost is n_queries x this",
    "measured_on":               "compare_capabilities — the staleness warning",
    # NOTE: compare_capabilities is NOT BUILT. See drift/README.md. It is
    # deferred until after the first live battery, and this comment exists so the
    # assignment above is not mistaken for a working consumer — an unread field
    # with a plausible owner is worse than one with none, because the map says it
    # is covered.

    # Documentation by design, with no consumer and no plans for one.
    "cross_turn":                "DOCUMENTATION — recorded, never acted on. This once "
                                 "gated multi-turn in the Anthropic validator, which was "
                                 "a provider mechanism promoted to a format constraint. "
                                 "Study designers need it; the code must not branch on it.",
    "notes":                     "DOCUMENTATION — free text for study designers.",
}
