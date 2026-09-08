"""Anthropic provider tests, run against recorded fixtures.

Every fixture is a real response, and several were collected specifically to make
the wrong and the right behaviour differ — a fixture where they coincide passes
vacuously and tests nothing.

The fixtures are also the only artifact in this project that can disagree with the
design document. On the day they were collected, four things the document asserted
turned out to be wrong, and each would have passed a synthetic suite built from it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from machine_psych.providers import Anthropic

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "anthropic"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())["response"]


@pytest.fixture
def provider():
    return Anthropic(api_key="test-key")


# ═══════════════════════════════════════════════════════════════════════════════
# Status — five states
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fixture,expected", [
    ("ungrounded_ok", "ok"),
    ("grounded_ok", "ok"),
    ("multiturn_t1", "ok"),
    ("truncated_partial", "truncated"),
    ("incomplete_empty", "incomplete"),
    ("incomplete_ends_on_tool", "incomplete"),
    ("rejects_temperature", "error"),
    ("rejects_unknown_param", "error"),
    ("rejects_bad_model", "error"),
    ("overloaded", "unavailable"),
])
def test_status(provider, fixture, expected):
    assert provider.status(load(fixture))[0] == expected


def test_truncated_is_not_incomplete(provider):
    """The fixture must carry TEXT, or it passes under both predicates.

    Truncation here returns a partial answer — measured at 8, 132, 944 and 4,338
    characters across four budgets. A fixture truncated to nothing would satisfy
    a predicate that calls this `incomplete` and one that calls it `truncated`,
    and would therefore test neither.
    """
    body = load("truncated_partial")
    status, _ = provider.status(body)
    parsed = provider.parse(body)
    assert status == "truncated"
    assert parsed.answer_chars > 0, "fixture has no partial text — it tests nothing"


def test_no_answer_is_a_thinking_block_not_an_empty_array(provider):
    """The shape has changed twice and the fixture records where it is now.

    August: `max_tokens: 5` returned `content: []`. After thinking became
    default: a thinking block with empty thinking text, because thinking runs
    first and consumes the budget. A predicate written against the older shape
    would call this `ok` with a null answer.
    """
    body = load("incomplete_empty")
    assert body["content"], "an empty array would be the pre-thinking shape"
    assert provider.status(body)[0] == "incomplete"
    assert provider.parse(body).answer_text is None


def test_status_and_parse_agree_on_whether_there_is_an_answer(provider):
    """They did not, and real data was required to show it.

    Anthropic now emits text BEFORE tool calls. This fixture has one such block —
    "I'll research current CRM options..." — then two tool calls and nothing
    after. A `status` that checks for any text ANYWHERE calls that `truncated`
    while `parse` applies the answer heuristic and reports zero answer
    characters: a record claiming a partial answer and containing none.

    Both now use the same extraction. Any divergence between them is a bug by
    construction rather than by inspection.
    """
    body = load("incomplete_ends_on_tool")
    assert body["content"][-1]["type"] == "server_tool_use"
    assert any(b["type"] == "text" for b in body["content"]), (
        "no pre-tool text — this fixture no longer exercises the divergence")

    parsed = provider.parse(body)
    assert parsed.answer_chars == 0, "the text is process, not answer"
    assert parsed.n_process_blocks == 1
    assert provider.status(body)[0] == "incomplete"


@pytest.mark.parametrize("fixture", [
    "grounded_ok", "truncated_partial", "incomplete_empty",
    "incomplete_ends_on_tool", "multiturn_t1", "ungrounded_ok"])
def test_status_never_claims_an_answer_parse_cannot_find(provider, fixture):
    body = load(fixture)
    status = provider.status(body)[0]
    has_answer = provider.parse(body).answer_chars > 0
    assert (status in ("ok", "truncated")) == has_answer


def test_only_transient_errors_retry(provider):
    """Retrying an invalid_request_error five times wastes a minute per record
    and never succeeds."""
    assert provider.status(load("overloaded"))[0] == "unavailable"
    for name in ("rejects_temperature", "rejects_unknown_param", "rejects_bad_model"):
        assert provider.status(load(name))[0] == "error"


def test_error_shape_is_uniform_across_status_codes(provider):
    """400 and 404 differ only in `error.type`, so one classifier covers both."""
    for name in ("rejects_unknown_param", "rejects_bad_model"):
        body = load(name)
        assert body["type"] == "error"
        assert "type" in body["error"] and "message" in body["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# Parse
# ═══════════════════════════════════════════════════════════════════════════════

def test_n_queries_comes_from_the_usage_figure(provider):
    """REGRESSION 19, weakened by an API change and recorded as weakened.

    This once had a fixture where the two DISAGREED — 16 `web_search_requests`
    against 3 `server_tool_use` blocks, because blocks batched requests. Measured
    2026-08-28 they are 1:1 on every fixture, so the disagreement is no longer
    reproducible on demand and this test can only assert the source.

    A test that cannot fail for the right reason is worth marking rather than
    deleting: the rule still holds on OpenAI, where the item count and the
    request count differ, and Anthropic could revert.
    """
    body = load("grounded_ok")
    parsed = provider.parse(body)
    reported = body["usage"]["server_tool_use"]["web_search_requests"]
    assert parsed.n_queries == reported
    blocks = sum(1 for b in body["content"] if b.get("type") == "server_tool_use")
    if parsed.n_queries == blocks:
        pytest.skip("usage and block count now agree — this fixture cannot "
                    "distinguish a parser that counts blocks")


def test_thinking_is_not_a_tool_boundary(provider):
    """The answer heuristic must not cut at the model's own reasoning.

    `thinking` is neither text nor a tool, so a naive "last non-text block"
    search would treat it as a boundary. Since thinking now fires on every call
    by default, that would truncate the answer on every record — the fixture has
    a thinking block followed by 25 text blocks, and the wrong logic returns 0.
    """
    parsed = provider.parse(load("multiturn_t1"))
    body = load("multiturn_t1")
    expected = sum(1 for b in body["content"] if b.get("type") == "text")
    assert parsed.n_answer_blocks == expected, (
        "text blocks were lost — treating thinking as a tool boundary gives 0")
    assert parsed.answer_chars > 0


def test_zero_is_not_absent(provider):
    """A count of zero is a measurement; None means the concept does not apply.

    An earlier version wrote `sum(...) or None` throughout and destroyed the
    distinction on exactly the records where it matters — a reasoning-off arm
    would report None thinking steps, indistinguishable from a provider that
    cannot think at all.
    """
    grounded = provider.parse(load("grounded_ok"))
    ungrounded = provider.parse(load("ungrounded_ok"))
    incomplete = provider.parse(load("incomplete_empty"))

    # NOT asserted as 0. An August measurement found zero process blocks on every
    # grounded record and that was written down as a finding — "the model searches
    # first, then writes". Measured 2026-08-28 it is 1, because Anthropic now
    # emits text BEFORE tool calls. The point of this test is that the count is a
    # NUMBER rather than None, not what the number happens to be.
    assert grounded.n_process_blocks is not None
    assert isinstance(grounded.n_process_blocks, int)
    assert ungrounded.n_thought_steps == 0, "produced no thinking, not inapplicable"
    assert incomplete.n_answer_blocks == 0, "produced no answer, not inapplicable"
    assert ungrounded.n_sources_retrieved is None, "no retrieval set EXISTS here"
    assert grounded.n_sources_retrieved > 0


def test_citations_carry_the_quote_not_an_offset(provider):
    """A third mechanism alongside byte and character offsets.

    This provider gives `cited_text` directly, which is what the analysis rule
    wants — compare quotes, not offsets — with no arithmetic to get wrong.
    """
    cites = provider.parse(load("grounded_ok")).citations
    assert cites
    for c in cites:
        assert c["offset_unit"] == "quoted"
        assert c["start"] is None and c["end"] is None
        assert c["quote"] and c["url"]


def test_retrieval_set_includes_uncited_sources(provider):
    """The capability that exists on no other provider.

    OpenAI and Gemini report only what they cited, so "what distinguishes a cited
    source from an uncited one" is answerable here and nowhere else — and that
    question is what the scraping layer is being built to answer.
    """
    sources = provider.parse(load("grounded_ok")).sources
    assert any(s["cited"] for s in sources)
    assert any(not s["cited"] for s in sources), "no uncited sources — the fixture is not exercising the retrieval set"
    assert all(s["retrieved"] for s in sources)


def test_page_age_is_captured(provider):
    """How fresh the sources behind a recommendation are — a direct question for
    this research, and one no other provider answers."""
    sources = provider.parse(load("grounded_ok")).sources
    with_age = [s for s in sources if s["page_age"]]
    # NOT all of them — 33 of 41 results carried it on the fixture measured. A
    # page with no discoverable date has none, which is itself informative and
    # would be destroyed by filling it.
    assert with_age, "no page_age at all — the capability may have been removed"
    assert all("page_age" in s for s in sources), "the column must exist on every row"


def test_usage_is_read_from_the_response(provider):
    body = load("grounded_ok")
    parsed = provider.parse(body)
    assert parsed.in_tok_billed == body["usage"]["input_tokens"]
    assert parsed.in_tok_processed == body["usage"]["input_tokens"]
    assert parsed.thinking_tok == body["usage"]["output_tokens_details"]["thinking_tokens"]


def test_answer_extraction_is_declared_heuristic(provider):
    """The only inferred answer of the three providers, and the column says so.

    Analysis should know it is holding an inference rather than something the API
    stated.
    """
    assert provider.parse(load("grounded_ok")).answer_extraction == "heuristic"


# ═══════════════════════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════════════════════

def test_reasoning_off_disables_thinking(provider):
    body = provider.build("q", {"model": "anthropic/claude-sonnet-5",
                                "reasoning": "off"})
    assert body["thinking"] == {"type": "disabled"}
    assert "output_config" not in body


def test_reasoning_level_uses_effort_not_budget_tokens(provider):
    """The surface moved on 2026-08-27.

    `thinking: {"type": "enabled", "budget_tokens": N}` — what every earlier note
    assumed — now returns 400: "not supported for this model. Use
    thinking.type.adaptive and output_config.effort". This test exists so a
    revert to the old mapping fails loudly.
    """
    body = provider.build("q", {"model": "anthropic/claude-sonnet-5",
                                "reasoning": "high"})
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in json.dumps(body)


def test_no_thinking_key_when_reasoning_unspecified(provider):
    """Thinking is on by default now; omitting the key is the default arm."""
    body = provider.build("q", {"model": "anthropic/claude-sonnet-5"})
    assert "thinking" not in body


def test_escape_hatch_passes_through(provider):
    """Unknown parameters are sent as-is. The API rejects them and names the
    field, and duplicating that check here would be a second thing to keep in
    sync with a moving API."""
    body = provider.build("q", {"model": "anthropic/claude-sonnet-5",
                                "future_param": 7})
    assert body["future_param"] == 7


def test_meta_keys_never_reach_the_request(provider):
    body = provider.build("q", {"model": "anthropic/claude-sonnet-5",
                                "repetitions": 5, "note": "a comment"})
    assert "repetitions" not in body and "note" not in body


def test_search_tool_is_not_mutated(provider):
    """A dict merged into the module-level tool constant would leak into every
    subsequent call in the process."""
    provider.build("q", {"model": "anthropic/claude-sonnet-5",
                         "search": {"max_uses": 3}})
    from machine_psych.providers.anthropic import SEARCH_TOOL
    assert "max_uses" not in SEARCH_TOOL


# ═══════════════════════════════════════════════════════════════════════════════
# Passback
# ═══════════════════════════════════════════════════════════════════════════════

def test_passback_is_verbatim_including_signatures(provider):
    """A thinking block whose signature is stripped invalidates the turn, and
    dropping them changes what the model is continuing from."""
    body = load("grounded_ok")
    items = provider.passback(body)
    assert len(items) == 1 and items[0]["role"] == "assistant"
    assert items[0]["content"] == body["content"]
    assert any(b.get("signature") for b in items[0]["content"])


def test_passback_of_an_empty_response_is_empty(provider):
    assert provider.passback({"content": []}) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Transport
# ═══════════════════════════════════════════════════════════════════════════════

def test_headers_require_a_key():
    with pytest.raises(RuntimeError, match="set_api_key"):
        Anthropic().headers()


def test_url_is_constant(provider):
    assert provider.url_for({}) == "https://api.anthropic.com/v1/messages"


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture audit, 2026-09-05
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_discriminating_fixtures_still_discriminate():
    """Several fixtures exist to make a WRONG parser fail. If the API changes so
    that right and wrong produce the same answer, the fixture stops testing
    anything — and nothing announces that.

    `usage_contradicts_blocks` already went that way: it was collected at 16
    requests against 3 blocks and is now 6 against 6, so the test using it skips.
    This checks the two that still work, so a future refresh cannot quietly kill
    them too.
    """
    partial = load("truncated_partial")
    text = "".join(b.get("text", "") for b in partial["content"]
                   if b.get("type") == "text")
    assert partial["stop_reason"] == "max_tokens"
    assert text, (
        "truncated_partial has no text — `truncated` and `incomplete` become "
        "indistinguishable and the fixture tests nothing")

    body = load("grounded_ok")
    results = [r for b in body["content"]
               if b.get("type") == "web_search_tool_result"
               for r in (b.get("content") or [])]
    cited = {c.get("url") for b in body["content"] if b.get("type") == "text"
             for c in (b.get("citations") or [])}
    assert len(results) > len(cited), (
        "every retrieved source was cited — the RETRIEVAL SET is what makes this "
        "provider unique, and a fixture where it equals the citation set cannot "
        "show the difference")
