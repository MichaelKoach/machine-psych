"""OpenAI provider tests.

Two jobs. The first is the same as any provider suite: does this parser read this
provider correctly. The second is new and matters more — **does the contract hold
across two providers**, which is the first point in the build where that can be
asked at all.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from machine_psych.providers import Anthropic, OpenAI

HERE = pathlib.Path(__file__).parent
OA = HERE / "fixtures" / "openai"
AN = HERE / "fixtures" / "anthropic"


def load(name: str, where=OA) -> dict:
    return json.loads((where / f"{name}.json").read_text())["response"]


@pytest.fixture
def provider():
    return OpenAI(api_key="test-key")


# ═══════════════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fixture,expected", [
    ("ungrounded_ok", "ok"),
    ("grounded_ok", "ok"),
    ("multiturn_t1", "ok"),
    ("effort_none", "ok"),
    ("reasoning_summary", "ok"),
    ("truncate_4096", "ok"),
    ("truncate_16", "incomplete"),
    ("truncate_1024", "incomplete"),
    ("truncate_grounded_512", "incomplete"),
    ("rejects_temperature", "error"),
    ("rejects_unknown_param", "error"),
    ("rejects_bad_model", "error"),
    ("rejects_no_credit", "error"),
    ("rate_limited", "unavailable"),
])
def test_status(provider, fixture, expected):
    assert provider.status(load(fixture))[0] == expected


def test_http_200_is_not_success(provider):
    """`status: incomplete` arrives with a 200 and no message item.

    This provider is the reason the status predicate exists at all — a truthiness
    check on the response would call every truncated arm a success with a null
    answer.
    """
    body = load("truncate_1024")
    assert body["status"] == "incomplete"
    assert provider.status(body)[0] == "incomplete"


def test_no_truncated_state_across_the_whole_sweep(provider):
    """Open question 6, answered by sweeping the budget across five values.

    Every incomplete arm has ZERO message items and ZERO text. The state is real
    on Anthropic, which streams text into content blocks as it generates; this
    provider emits the message item complete or not at all.

    Testing one budget would have proved nothing — a small budget cannot produce
    partial text by construction, which is exactly why the state went unnoticed
    on the other provider until the value was varied.
    """
    for budget in (16, 64, 256, 1024):
        body = load(f"truncate_{budget}")
        assert provider.status(body)[0] == "incomplete", budget
        assert provider.parse(body).answer_chars == 0, budget
    assert provider.parse(load("truncate_4096")).answer_chars > 0


def test_not_every_429_is_transient(provider):
    """An exhausted credit balance wears a rate-limit status code.

    `insufficient_quota` at HTTP 429 is a billing error. A classifier keyed on
    the status code retries it — five attempts per record across a battery, hours
    wasted, never succeeding. Learned by writing exactly that loop by hand in a
    probe cell while this method's specification already said to use the type.
    """
    quota = load("rejects_no_credit")
    limit = load("rate_limited")
    assert quota["error"]["type"] == "insufficient_quota"
    assert limit["error"]["type"] == "rate_limit_exceeded"
    assert provider.status(quota)[0] == "error"
    assert provider.status(limit)[0] == "unavailable"


# ═══════════════════════════════════════════════════════════════════════════════
# Parse
# ═══════════════════════════════════════════════════════════════════════════════

def test_answer_extraction_is_structural(provider):
    """The message item IS the answer — no inference.

    The contrast is the point: Anthropic must guess the answer from block
    position, and the column records which kind of claim each provider's
    answer_text is.
    """
    assert provider.parse(load("grounded_ok")).answer_extraction == "structural"
    an = Anthropic(api_key="k").parse(load("grounded_ok", AN))
    assert an.answer_extraction == "heuristic"


def test_n_queries_falls_back_when_usage_stops_reporting_it(provider):
    """The authoritative field was REMOVED, and the fallback is worse.

    `usage.tool_usage.web_search.num_requests` was the count that mattered —
    items batch queries, so counting them understates it, measured at 5 requests
    against 6 items. As of 2026-08-28 the field is absent from every grounded
    fixture and usage carries only input/output/total.

    So `n_queries` now comes from the item count, which is approximate. That is
    better than None and worse than the truth, and `n_queries_source` records
    which one produced the number so a cross-provider comparison can see that
    this one is estimated rather than reported.
    """
    body = load("grounded_ok")
    items = sum(1 for i in body["output"] if i.get("type") == "web_search_call")
    parsed = provider.parse(body)

    assert "tool_usage" not in body["usage"], (
        "the field is back — restore the disagreement assertion, which is "
        "stronger than this one")
    assert parsed.n_queries == items
    assert "item_count" in parsed.extra["n_queries_source"]


def test_n_queries_prefers_usage_when_it_exists(provider):
    """The fallback must not shadow the authoritative figure if it returns."""
    body = json.loads(json.dumps(load("grounded_ok")))
    body["usage"]["tool_usage"] = {"web_search": {"num_requests": 99}}
    parsed = provider.parse(body)
    assert parsed.n_queries == 99
    assert parsed.extra["n_queries_source"] == "usage"


def test_readable_reasoning_populates(provider):
    """No longer a Gemini-only capability.

    `reasoning.summary` returned EMPTY arrays at characterisation time and
    populates now. The claim that only one provider could do this had reached the
    capability table, the design document and the project notes.
    """
    # Asserts that reasoning ARRIVES, not what it says — an earlier version
    # asserted a phrase from a synthetic fixture and was therefore a test of my
    # own invented content.
    parsed = provider.parse(load("reasoning_summary"))
    assert parsed.thought_text and len(parsed.thought_text) > 50
    assert parsed.thoughts


def test_readable_reasoning_is_opt_in(provider):
    """Without `reasoning.summary` in the request, the items carry
    `encrypted_content` only — so None here can mean the request did not ask
    rather than the provider cannot. The request is recorded alongside every
    record, which is where that is resolved."""
    assert provider.parse(load("ungrounded_ok")).thought_text is None


def test_no_retrieval_set(provider):
    """None, not zero and not an empty list.

    "This provider does not report what it saw" is a different claim from "it saw
    nothing", and conflating them would make it look more selective than
    Anthropic when it is merely quieter.
    """
    parsed = provider.parse(load("grounded_ok"))
    assert parsed.n_sources_retrieved is None
    assert all(s["retrieved"] is None for s in parsed.sources)
    assert all(s["cited"] for s in parsed.sources)


def test_grounded_reports_whether_search_actually_ran(provider):
    """Search is a PERMISSION on every provider, not a condition — measured
    2026-09-09. A grounded fixture searched, so this is True; a call with the
    tool attached on a prompt needing nothing would be False."""
    assert provider.parse(load("grounded_ok")).grounded is True


def test_zero_is_not_absent(provider):
    parsed = provider.parse(load("truncate_16"))
    assert parsed.n_answer_blocks == 0
    assert parsed.n_process_blocks == 1


# ═══════════════════════════════════════════════════════════════════════════════
# THE CROSS-PROVIDER CONTRACT — the first point this can be asked
# ═══════════════════════════════════════════════════════════════════════════════

def test_citations_have_identical_columns_across_two_mechanisms():
    """Character offsets and quoted text must produce the same side table.

    Anthropic supplies `cited_text` and no offsets; OpenAI supplies offsets and
    no quote. If either needed a special case downstream, the Tier 2 design would
    have failed — and Gemini adds a third mechanism (byte offsets), so a shape
    that only holds two is not enough.
    """
    oa = OpenAI(api_key="k").parse(load("grounded_ok")).citations
    an = Anthropic(api_key="k").parse(load("grounded_ok", AN)).citations
    assert oa and an
    assert sorted(oa[0]) == sorted(an[0])


def test_quote_is_populated_by_both_mechanisms():
    """`quote` is the comparable column; offsets are provenance.

    The analysis rule is compare quotes, not offsets — so the quote is EXTRACTED
    where only offsets exist and READ where it is supplied, and neither provider
    is a special case for anything downstream.
    """
    oa = OpenAI(api_key="k").parse(load("grounded_ok")).citations
    an = Anthropic(api_key="k").parse(load("grounded_ok", AN)).citations
    assert all(c["quote"] for c in oa), "offsets were not resolved to text"
    assert all(c["quote"] for c in an), "supplied quote was dropped"


def test_offset_unit_is_recorded_per_row():
    """Three mechanisms exist and a row must say which one it is.

    A loader applying character indexing to byte offsets produces spans that are
    silently wrong and drift further into the answer — measured at 70 of 107
    citations on the provider that uses bytes.
    """
    oa = OpenAI(api_key="k").parse(load("grounded_ok")).citations
    an = Anthropic(api_key="k").parse(load("grounded_ok", AN)).citations
    assert {c["offset_unit"] for c in oa} == {"char"}
    assert {c["offset_unit"] for c in an} == {"quoted"}


def test_both_providers_return_the_same_dataclass():
    """One shape above the provider boundary. If a provider's distinctive
    structure could not survive `parse()`, normalising at analysis would have
    failed and this would be a gateway library with extra steps."""
    from machine_psych.providers.base import ParsedResponse
    oa = OpenAI(api_key="k").parse(load("grounded_ok"))
    an = Anthropic(api_key="k").parse(load("grounded_ok", AN))
    assert isinstance(oa, ParsedResponse) and isinstance(an, ParsedResponse)


def test_tier3_differs_by_capability_not_by_provider_name():
    """The columns each provider fills follow from the capability table.

    Anthropic has a retrieval set and this one does not; this one has readable
    reasoning and Anthropic does not. Neither parser branches on the provider
    name, which is why the readable-reasoning change was one table edit.
    """
    oa = OpenAI(api_key="k").parse(load("reasoning_summary"))
    an = Anthropic(api_key="k").parse(load("grounded_ok", AN))
    assert an.n_sources_retrieved is not None and oa.n_sources_retrieved is None
    assert oa.thought_text is not None and an.thought_text is None


# ═══════════════════════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════════════════════

def test_store_is_false_by_default(provider):
    """The endpoint otherwise keeps the response server-side for resumption,
    making a corpus depend on provider state rather than being self-contained."""
    assert provider.build("q", {"model": "openai/gpt-5.6-sol"})["store"] is False


def test_reasoning_off_maps_to_a_level_not_a_different_key(provider):
    """`none` is a real effort level here, measured at 0 reasoning tokens.

    Anthropic's off arm is a different key entirely (`thinking: disabled`), which
    is why the intent layer maps rather than passes through.
    """
    body = provider.build("q", {"model": "openai/gpt-5.6-sol", "reasoning": "off"})
    assert body["reasoning"]["effort"] == "none"
    assert "summary" not in body["reasoning"], "no summary to request when off"


def test_summary_is_requested_when_reasoning_is_on(provider):
    """Opt-in, so the capability would read as available while every record
    showed nothing if this were omitted."""
    body = provider.build("q", {"model": "openai/gpt-5.6-sol", "reasoning": "high"})
    assert body["reasoning"] == {"effort": "high", "summary": "detailed"}


def test_verbosity_and_max_tool_calls(provider):
    body = provider.build("q", {"model": "openai/gpt-5.6-sol", "verbosity": "low",
                                "search": True, "max_tool_calls": 1})
    assert body["text"] == {"verbosity": "low"}
    assert body["max_tool_calls"] == 1


def test_search_tool_is_not_mutated(provider):
    provider.build("q", {"model": "openai/gpt-5.6-sol",
                         "search": {"search_context_size": "low"}})
    from machine_psych.providers.openai import SEARCH_TOOL
    assert SEARCH_TOOL == {"type": "web_search"}


def test_escape_hatch_and_meta_keys(provider):
    body = provider.build("q", {"model": "openai/gpt-5.6-sol",
                                "future_param": 7, "repetitions": 5})
    assert body["future_param"] == 7
    assert "repetitions" not in body


def test_passback_includes_reasoning_items(provider):
    """Reasoning items carry `encrypted_content` the model uses to continue; a
    turn assembled without them is a different turn."""
    body = load("grounded_ok")
    items = provider.passback(body)
    assert items == body["output"]
    assert any(i["type"] == "reasoning" for i in items)


def test_headers_require_a_key():
    with pytest.raises(RuntimeError, match="set_api_key"):
        OpenAI().headers()
