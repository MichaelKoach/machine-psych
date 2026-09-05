"""Gemini provider tests, and the three-provider contract.

Gemini differs from the other two in more ways than they differ from each other,
so this is where the design either holds or does not. Three citation mechanisms
now exist; three token-accounting conventions; three cross-turn behaviours; and
one provider that cannot express a reasoning-off arm at all.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import pytest

from machine_psych.providers import Anthropic, Gemini, OpenAI

HERE = pathlib.Path(__file__).parent
GE, OA, AN = (HERE / "fixtures" / n for n in ("gemini", "openai", "anthropic"))


def load(name: str, where=GE) -> dict:
    return json.loads((where / f"{name}.json").read_text())["response"]


@pytest.fixture
def provider():
    return Gemini(api_key="test-key")


# ═══════════════════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fixture,expected", [
    ("ungrounded_ok", "ok"),
    ("grounded_ok", "ok"),
    ("ungrounded_with_tool", "ok"),
    ("readable_reasoning", "ok"),
    ("multiturn_t1", "ok"),
    ("incomplete_no_output", "truncated"),
    ("unavailable_503", "unavailable"),
    ("unavailable_api_error", "unavailable"),
    ("rejects_bad_field", "error"),
    ("rejects_bad_model", "error"),
])
def test_status(provider, fixture, expected):
    assert provider.status(load(fixture))[0] == expected


def test_capacity_failure_arrives_as_api_error_not_503(provider):
    """Keying on the status code alone means retry never fires.

    Capacity failures here come back as `code: "api_error"` with a prose message.
    Measured at roughly 1 in 6 on the newest model, with a 44-record battery
    needing 105 calls — so a classifier that misses this shape loses records to
    exactly the failure retry exists for.
    """
    body = load("unavailable_api_error")
    assert body["error"]["code"] == "api_error"
    assert provider.status(body)[0] == "unavailable"


def test_a_400_never_retries(provider):
    """Five attempts at a malformed request wastes a minute per record."""
    assert provider.status(load("rejects_bad_field"))[0] == "error"


# ═══════════════════════════════════════════════════════════════════════════════
# Byte offsets — the fixture is built so character indexing FAILS
# ═══════════════════════════════════════════════════════════════════════════════

def test_citations_use_byte_offsets(provider):
    """REGRESSION 3, and the fixture contains non-ASCII so it cannot pass vacuously.

    An ASCII-only fixture gives identical results under both conventions and
    proves nothing. This answer has en-dashes, so every span diverges — measured
    on real data as 34/35 byte matches against 2/35 character, and 70 of 107
    citations wrong on a second record.

    The failure is silent: quotes come back wrong by a character or two and drift
    further into the answer, with no error.
    """
    body = load("grounded_ok")
    answer = provider.parse(body).answer_text
    answer_bytes = answer.encode("utf-8")
    assert len(answer_bytes) > len(answer), "fixture is ASCII — it tests nothing"

    for c in provider.parse(body).citations:
        assert c["quote"] == answer_bytes[c["start"]:c["end"]].decode("utf-8")


def test_character_indexing_would_be_wrong_after_the_first_non_ascii(provider):
    """The counterfactual, asserted — but not for EVERY citation.

    Offsets only diverge once a multi-byte character has appeared earlier in the
    text. On the measured fixture the first non-ASCII character sits at position
    921 and 57 of 64 citations fall after it. An earlier version of this test
    asserted all of them, which was true of a synthetic answer that began with an
    en-dash and is not true of a real one.

    The seven that agree are the ones before position 921, and they would pass
    under either convention — which is exactly why a fixture must contain
    non-ASCII EARLY enough to matter, and why "all" was the wrong assertion
    rather than a stricter one.
    """
    body = load("grounded_ok")
    parsed = provider.parse(body)
    answer = parsed.answer_text
    ab = answer.encode("utf-8")
    first_multibyte = next((i for i, ch in enumerate(answer) if ord(ch) > 127), None)
    assert first_multibyte is not None, "fixture is ASCII — it tests nothing"

    wrong = [c for c in parsed.citations
             if answer[c["start"]:c["end"]]
             != ab[c["start"]:c["end"]].decode("utf-8", "replace")]
    assert wrong, "no citation diverges — the fixture cannot detect the wrong convention"
    assert all(c["end"] > first_multibyte for c in wrong)


def test_absent_start_index_is_read_as_zero(provider):
    """REGRESSION 4, no longer reproducible from a live response.

    Gemini omitted `start_index` when it was 0, so the first citation of every
    grounded record had no start at all and `ann["start_index"]` raised. Measured
    2026-08-28: 64 annotations, none omit the key, and the minimum start is 390 —
    the API now always sends it.

    The handling is kept and tested SYNTHETICALLY rather than deleted. The
    behaviour was real, the omit-when-zero convention still governs other fields
    on this provider, and `.get(k) or 0` costs nothing. Deleting a guard because
    the current API stopped needing it is how a fixed bug returns.
    """
    synthetic = {"end_index": 12, "url": "https://x", "title": "x"}
    assert Gemini._citations(
        [{"type": "model_output",
          "content": [{"text": "Some answer.", "annotations": [synthetic]}]}],
        "byte")[0]["start"] == 0

    live = load("grounded_ok")["steps"][-1]["content"][0]["annotations"][0]
    if "start_index" not in live:
        pytest.fail("the API omits it again — restore the live assertion")


def test_many_to_many_is_preserved(provider):
    """One span citing multiple sources arrives as repeated rows.

    Measured 20 of 35 supports mapping to more than one source. Flattened into
    rows rather than nested, which is the shape a DataFrame wants and which also
    lets the same side table hold a provider whose citations are one-to-one.
    """
    spans = Counter((c["start"], c["end"])
                    for c in provider.parse(load("grounded_ok")).citations)
    assert any(v > 1 for v in spans.values())


def test_urls_are_redirects(provider):
    """They resolve to real pages, and the redirect yields the address even when
    the fetch is blocked — so resolution and retrieval are separate steps for the
    scraping layer. `title` carries the real domain; the URL's netloc is always
    Google's."""
    for s in provider.parse(load("grounded_ok")).sources:
        assert s["is_redirect"]
        assert "vertexaisearch" in s["redirect_host"]
        assert s["title"] and "vertexaisearch" not in s["title"]


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional grounding — the only provider where this is a real question
# ═══════════════════════════════════════════════════════════════════════════════

def test_tool_provided_does_not_mean_search_ran(provider):
    """Providing the tool is a PERMISSION, not a condition.

    So "search on" is not a study condition here, and `grounded` is how an
    analysis knows which arm a record actually landed in. On the other two
    providers the tool means search happened, which is why the column is None
    there rather than False.
    """
    request = json.loads((GE / "ungrounded_with_tool.json").read_text())["_request"]
    assert request.get("tools"), "fixture did not provide the tool — it tests nothing"
    assert provider.parse(load("ungrounded_with_tool")).grounded is False
    assert provider.parse(load("grounded_ok")).grounded is True


def test_grounded_is_none_on_providers_where_it_is_meaningless():
    an = Anthropic(api_key="k").parse(load("grounded_ok", AN))
    oa = OpenAI(api_key="k").parse(load("grounded_ok", OA))
    assert an.grounded is None and oa.grounded is None


# ═══════════════════════════════════════════════════════════════════════════════
# Token accounting
# ═══════════════════════════════════════════════════════════════════════════════

def test_billed_and_processed_input_differ(provider):
    """REGRESSION 18. The only provider that separates them, by two orders of
    magnitude on grounded calls — 326 billed against 32,606 processed.

    Comparing the billed figure across providers measures billing policy rather
    than context volume, which is what made a 40x gap look architectural when it
    was an invoice.
    """
    parsed = provider.parse(load("grounded_ok"))
    assert parsed.in_tok_billed != parsed.in_tok_processed
    assert parsed.in_tok_processed > parsed.in_tok_billed * 10


def test_n_queries_from_grounding_tool_count(provider):
    """REGRESSION 19, third variant. Not from counting steps.

    `total_tool_use_tokens` is NOT this figure — it reads 0 on search calls and
    populates only for url_context, which is why it looked like an unused field
    and why reading it would report zero searches on every grounded record.
    """
    body = load("grounded_ok")
    steps = sum(1 for s in body["steps"] if s.get("type") == "google_search_call")
    parsed = provider.parse(body)
    assert parsed.n_queries != steps, "fixture no longer disagrees"
    assert parsed.n_queries == sum(
        c["count"] for c in body["usage"]["grounding_tool_count"])


def test_thinking_tokens_are_separate_from_output(provider):
    """Kept separate and added to the total here; counted INSIDE output on the
    other two. Which is why `answer_chars` rather than `out_tok` is the
    comparable length measure."""
    body = load("grounded_ok")
    u = body["usage"]
    assert u["total_tokens"] > u["total_input_tokens"] + u["total_output_tokens"]


# ═══════════════════════════════════════════════════════════════════════════════
# Thought text
# ═══════════════════════════════════════════════════════════════════════════════

def test_thought_text_comes_from_summary_not_content(provider):
    """The key is ABSENT without `thinking_summaries: "auto"`.

    A loader reading `content` reports zero thought text on every record, which
    looks like the API withholding reasoning rather than the extractor looking in
    the wrong place. That cost a full shakedown round.
    """
    # Asserts that reasoning text ARRIVES, not what it says. An earlier version
    # asserted a phrase from a synthetic fixture, which made the test a check on
    # my own invented content rather than on the parser.
    text = provider.parse(load("readable_reasoning")).thought_text
    assert text and len(text) > 50
    assert provider.parse(load("no_summaries")).thought_text is None


def test_summaries_are_requested_by_default(provider):
    """Free — measured 3,458 thought tokens with summaries visible against 4,098
    without — so there is no reason for a record to be missing them."""
    body = provider.build("q", {"model": "gemini/gemini-3.7-flash"})
    assert body["generation_config"]["thinking_summaries"] == "auto"


# ═══════════════════════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════════════════════

def test_generatecontent_fields_are_refused(provider):
    """An unknown field is a 400 on EVERY call, which reads as a provider outage.

    That mistake cost a full round of "all models unavailable" before anyone
    looked at the latency — sub-second failure is the tell.
    """
    with pytest.raises(ValueError, match="generateContent"):
        provider.build("q", {"model": "gemini/gemini-3.7-flash",
                             "include_thoughts": True})


def test_user_turn_is_a_step_not_a_role():
    """`{"role": "user", ...}` returns: "use step_list input format instead of
    turn_list"."""
    assert Gemini(api_key="k").user_turn("hi") == {"type": "user_input",
                                                  "content": "hi"}


def test_passback_returns_every_step(provider):
    """All model-generated steps must be resent exactly as received; they carry
    signatures required to continue. An earlier version appended only the output,
    which the other endpoint tolerates and this one does not."""
    body = load("grounded_ok")
    assert provider.passback(body) == body["steps"]


def test_store_is_false(provider):
    """Defaults to TRUE here. A research corpus should be self-contained and
    should not leave client material on a provider's servers."""
    assert provider.build("q", {"model": "gemini/gemini-3.7-flash"})["store"] is False


def test_max_tokens_default_accounts_for_the_combined_budget(provider):
    """The budget covers thinking AND output, with thinking measured taking ~96%.
    A 4096 default would truncate the answer in a way that reads as a model
    failure."""
    body = provider.build("q", {"model": "gemini/gemini-3.7-flash"})
    assert body["generation_config"]["max_output_tokens"] >= 8192


# ═══════════════════════════════════════════════════════════════════════════════
# THE THREE-PROVIDER CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

def _all_three():
    return (Anthropic(api_key="k").parse(load("grounded_ok", AN)),
            OpenAI(api_key="k").parse(load("grounded_ok", OA)),
            Gemini(api_key="k").parse(load("grounded_ok", GE)))


def test_three_citation_mechanisms_one_shape():
    """Quoted text, character offsets, byte offsets — one side table.

    If any needed a special case downstream, normalising at analysis would have
    failed. Two mechanisms could have been a coincidence; three is the design
    working.
    """
    an, oa, ge = _all_three()
    cols = [sorted(p.citations[0]) for p in (an, oa, ge)]
    assert cols[0] == cols[1] == cols[2]
    assert {p.citations[0]["offset_unit"] for p in (an, oa, ge)} == {
        "quoted", "char", "byte"}


def test_quote_is_populated_by_all_three():
    """The comparable column. Offsets are provenance and differ in unit; the
    quote is the object every analysis actually compares."""
    for parsed in _all_three():
        assert all(c["quote"] for c in parsed.citations)


def test_every_provider_returns_the_same_dataclass():
    from machine_psych.providers.base import ParsedResponse
    assert all(isinstance(p, ParsedResponse) for p in _all_three())


def test_tier3_columns_follow_capabilities_not_provider_names():
    """Each provider fills exactly what its capabilities say.

    Anthropic has the retrieval set; Gemini and OpenAI have readable reasoning;
    only Gemini has meaningful `grounded`. No parser branches on a provider name,
    which is why a capability changing was a one-line table edit.
    """
    an, oa, ge = _all_three()
    assert an.n_sources_retrieved is not None
    assert oa.n_sources_retrieved is None and ge.n_sources_retrieved is None
    assert ge.grounded is not None
    assert an.grounded is None and oa.grounded is None


def test_no_parser_branches_on_provider_identity():
    import inspect
    for cls in (Anthropic, OpenAI, Gemini):
        src = inspect.getsource(cls)
        for name in ("anthropic", "openai", "gemini"):
            assert f'== "{name}"' not in src and f"== '{name}'" not in src


def test_answer_chars_is_comparable_where_out_tok_is_not():
    """Three token conventions: two count thinking inside output, one keeps it
    separate and adds it to the total. Character length is the same measurement
    everywhere."""
    for parsed in _all_three():
        assert parsed.answer_chars == len(parsed.answer_text or "")
