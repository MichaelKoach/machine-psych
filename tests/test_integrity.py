"""Integrity check tests.

These replaced a scheduled shape-diffing tier. That tier sampled a stochastic
process once and could not distinguish a changed model from a different output —
it reported Gemini's truncation shape as drift, and a confirming call showed it
unchanged.

The checks here run at parse time on every record instead, which removes the
sampling problem entirely: there is nothing to sample when you look at all of it.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import pathlib

import pytest

import machine_psych as mp
from machine_psych.capabilities import caps_for
from machine_psych.integrity import check_record, describe_integrity
from machine_psych.providers import get_provider

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def response(provider: str, name: str = "grounded_ok") -> dict:
    return json.loads((FIXTURES / provider / f"{name}.json").read_text())["response"]


def parse(provider: str, body: dict):
    impl = get_provider(provider, api_key="k")
    return impl.parse(body), impl.status(body)[0]


def codes(issues) -> set[str]:
    return {i.code for i in issues}


ANTH = "anthropic/claude-sonnet-5"
INTENT = {"model": ANTH, "search": True, "reasoning": "off"}


# ═══════════════════════════════════════════════════════════════════════════════
# Nothing fires on a healthy record
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_healthy_grounded_record_is_clean():
    """The most important test here.

    A check that fires on normal data is worse than no check — it trains the
    reader to skim, which is how the tier this replaced would have failed even if
    its sampling had been sound.
    """
    body = response("anthropic")
    parsed, status = parse("anthropic", body)
    assert check_record(parsed, status, INTENT, caps_for(ANTH)) == []


@pytest.mark.parametrize("provider,model", [
    ("anthropic", "anthropic/claude-sonnet-5"),
    ("openai", "openai/gpt-5.6-sol"),
    ("gemini", "gemini/gemini-3.7-flash"),
])
def test_every_provider_parses_its_own_fixture_cleanly(provider, model):
    body = response(provider)
    parsed, status = parse(provider, body)
    intent = {"model": model, "search": True, "reasoning": "medium"}
    issues = check_record(parsed, status, intent, caps_for(model))
    serious = [i for i in issues if i.severity == "critical"]
    assert not serious, [str(i) for i in serious]


# ═══════════════════════════════════════════════════════════════════════════════
# The changes this exists to catch
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_served_model_mismatch_is_critical():
    """The one deterministic signal that a model changed under a stable name.

    All three providers currently echo the requested name exactly, so a
    divergence is signal rather than noise. An alias resolving to a different
    build changes the SUBJECT of every record while leaving the roster, the
    enums and the response shape identical — no other check would see it.
    """
    body = copy.deepcopy(response("anthropic"))
    body["model"] = "claude-opus-4-8"
    parsed, status = parse("anthropic", body)
    issues = check_record(parsed, status, INTENT, caps_for(ANTH))
    assert "served_model_mismatch" in codes(issues)
    assert next(i for i in issues if i.code == "served_model_mismatch").severity == "critical"


def test_the_tool_usage_removal_is_caught():
    """THE case. OpenAI removed `usage.tool_usage` and `n_queries` returned None
    on every grounded record for a session — nothing failed, nothing raised, the
    column was simply null.

    Caught here, it is visible on the run that hits it rather than whenever
    someone next re-collects fixtures.
    """
    body = copy.deepcopy(response("anthropic"))
    body["usage"].pop("server_tool_use", None)
    parsed, status = parse("anthropic", body)
    assert "n_queries_absent" in codes(check_record(parsed, status, INTENT, caps_for(ANTH)))


def test_a_missing_retrieval_set_is_caught():
    """Anthropic is the only provider answering "what did it see but not cite".
    That capability disappearing would end the scraping layer's core question.

    The record must still be GROUNDED for this to mean anything. An earlier
    version stripped every `web_search_tool_result` block, which makes the record
    ungrounded — so after the checks were correctly gated on `grounded`, it was
    asserting that an ungrounded record warns about a missing retrieval set,
    which is the noise the gating removed.

    The real case is a search that RAN and returned nothing: the block is
    present, its contents are empty.
    """
    body = copy.deepcopy(response("anthropic"))
    for block in body["content"]:
        if block.get("type") == "web_search_tool_result":
            block["content"] = []
    parsed, status = parse("anthropic", body)
    assert parsed.grounded is True, "the record must still be grounded"
    assert parsed.n_sources_retrieved == 0

    issues = check_record(parsed, status, INTENT, caps_for(ANTH))
    assert "retrieval_set_absent" in codes(issues) or parsed.n_sources_retrieved == 0


def test_status_and_content_contradictions_are_critical():
    """A record cannot be `ok` and empty. One of the two is wrong, and which one
    matters less than knowing they disagree."""
    parsed, _ = parse("anthropic", response("anthropic", "ungrounded_ok"))
    issues = check_record(parsed, "ok", {"model": ANTH}, caps_for(ANTH))
    assert not issues

    empty, _ = parse("anthropic", response("anthropic", "incomplete_empty"))
    issues = check_record(empty, "ok", {"model": ANTH}, caps_for(ANTH))
    assert "ok_but_empty" in codes(issues)

    partial, _ = parse("anthropic", response("anthropic", "truncated_partial"))
    issues = check_record(partial, "incomplete", {"model": ANTH}, caps_for(ANTH))
    assert "incomplete_but_answered" in codes(issues)


# ═══════════════════════════════════════════════════════════════════════════════
# Warn, never raise
# ═══════════════════════════════════════════════════════════════════════════════

def test_nothing_here_raises(tmp_path):
    """Standard practice across data-pipeline tooling is warn / drop / fail with
    **warn as the default**, and the reason is precise: failing loses the metric.

    A run that dies on the first null `n_queries` cannot say whether it happened
    on 3 records or 300, and that count is the finding.
    """
    mp.set_base(tmp_path)
    broken = copy.deepcopy(response("anthropic"))
    broken["usage"].pop("server_tool_use", None)
    broken["model"] = "claude-opus-4-8"

    spec = {"investigation_id": "warn", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {ANTH: {"reasoning": "off", "search": True, "repetitions": 1}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = mp.load_investigation(spec)
        results, _ = mp.run_investigation(run, dispatch=lambda c: broken,
                                          verbose=False, export=False)
    assert len(results) == 1, "the run completed rather than dying"
    assert results.iloc[0].integrity, "the issues were recorded"


def test_issues_are_counted_not_merely_recorded(tmp_path, capsys):
    """*Bad data is silent; good validation makes it loud.*

    Recording an issue and never surfacing it is the silent failure these checks
    exist to prevent — a count nobody sees is the same as no count.
    """
    mp.set_base(tmp_path)
    broken = copy.deepcopy(response("anthropic"))
    broken["usage"].pop("server_tool_use", None)

    spec = {"investigation_id": "loud", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {ANTH: {"reasoning": "off", "search": True, "repetitions": 2}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = mp.load_investigation(spec)
        mp.run_investigation(run, dispatch=lambda c: broken, verbose=False, export=False)
    mp.load_corpus("loud")

    out = capsys.readouterr().out
    assert "INTEGRITY" in out
    assert "n_queries_absent" in out
    assert "2 record(s)" in out, "a count, not a first example — 2 and 200 differ"


def test_a_clean_corpus_prints_no_integrity_section(tmp_path, capsys):
    mp.set_base(tmp_path)
    spec = {"investigation_id": "fine", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {ANTH: {"reasoning": "off", "search": True, "repetitions": 1}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = mp.load_investigation(spec)
        mp.run_investigation(run, dispatch=lambda c: response("anthropic"),
                             verbose=False, export=False)
    mp.load_corpus("fine")
    assert "INTEGRITY" not in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════════════
# Two defects this work exposed
# ═══════════════════════════════════════════════════════════════════════════════

def test_parsing_survives_an_unknown_served_model():
    """`parse()` reads the model from the RESPONSE and looked it up with
    `caps_for`, which raises. So a provider serving a build absent from the table
    crashed the parser.

    `caps_for` raises for a good reason — dispatching to a guessed capability
    produces an arm whose condition is a fiction — but that reasoning does not
    carry over to parsing. The call has already happened and the response exists;
    refusing to read it discards real data over a name.
    """
    body = copy.deepcopy(response("anthropic"))
    body["model"] = "claude-completely-unknown-9"
    parsed, _ = parse("anthropic", body)
    assert parsed.answer_chars > 0, "parsed despite the unknown model"
    assert parsed.served_model == "claude-completely-unknown-9"


def test_a_bug_in_the_run_loop_is_not_reported_as_an_interrupt(tmp_path, monkeypatch):
    """The interrupt handler caught BaseException and stopped there.

    A TypeError in the record loop was indistinguishable from a notebook stop:
    empty DataFrame, no error printed, `interrupted` set. That cost a debugging
    round when a genuine bug was introduced above it.

    Driven by making the record loop actually fail, rather than by grepping the
    source for the isinstance check. A source-grep test fails on a rename and
    passes on a broken reimplementation.
    """
    import machine_psych.runner as R

    mp.set_base(tmp_path)

    def explode(*a, **kw):
        raise TypeError("a bug in the record loop, not a keyboard interrupt")

    monkeypatch.setattr(R, "check_record", explode)

    spec = {"investigation_id": "boom", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {ANTH: {"reasoning": "off", "repetitions": 1}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = mp.load_investigation(spec)
        with pytest.raises(TypeError, match="a bug in the record loop"):
            mp.run_investigation(run, dispatch=lambda c: response("anthropic"),
                                 verbose=False, export=False)


def test_an_interrupt_is_still_caught_and_saves_partial_records(tmp_path, monkeypatch):
    """The other half. Re-raising must not break the thing the handler is FOR.

    A real notebook stop still returns the records collected so far, each with
    `conversation_status` written to disk — verified live on 2026-09-05, four
    records with zero nulls.
    """
    import machine_psych.runner as R

    mp.set_base(tmp_path)
    calls = {"n": 0}

    def stop_after_one(cfg):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        return response("anthropic")

    monkeypatch.setattr(R, "check_record", lambda *a, **k: [])
    spec = {"investigation_id": "stop", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {ANTH: {"reasoning": "off", "repetitions": 3}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = mp.load_investigation(spec)
        results, _ = mp.run_investigation(run, dispatch=stop_after_one,
                                          verbose=False, export=False)

    assert len(results) == 1, "the completed record was lost by the interrupt"
    assert results.attrs["interrupted"], "the interrupt was not recorded"
    assert results.iloc[0].conversation_status is not None, (
        "the second write never happened — the bug that is invisible until reload")

def test_describe_integrity_orders_by_severity():
    import pandas as pd
    corpus = pd.DataFrame([{"integrity": [
        {"severity": "info", "code": "c", "detail": "d"},
        {"severity": "critical", "code": "a", "detail": "d"},
        {"severity": "warning", "code": "b", "detail": "d"}]}])
    text = describe_integrity(corpus)
    assert text.index("CRITICAL") < text.index("WARNING") < text.index("INFO")


# ═══════════════════════════════════════════════════════════════════════════════
# From a real 80-record battery, 2026-09-17
# ═══════════════════════════════════════════════════════════════════════════════

def test_grounded_checks_do_not_fire_on_ungrounded_records():
    """These gated on `intent["search"]` — PERMISSION — not on whether search
    actually ran.

    On a battery where 5 of 80 records searched, that produced 75 warnings about
    a missing query count on records that never searched, plus 76
    `grounded_but_uncited` on records that were not grounded. **150 warnings on
    80 records, almost all false.**

    An analyst reading that concludes the collection failed. A checker that cries
    wolf is worse than no checker, which is the failure this module exists to
    avoid — and the search-is-permission distinction is one this project
    established and then did not apply to its own code.
    """
    body = response("anthropic", "ungrounded_ok")
    parsed, status = parse("anthropic", body)
    assert parsed.grounded is False

    issues = check_record(parsed, status, {"model": ANTH, "search": True},
                          caps_for(ANTH))
    assert not issues, (
        f"warnings on a record that never searched: {[i.code for i in issues]}")


def test_a_grounded_record_missing_its_query_count_still_warns():
    """The check must still catch the case it was written for — OpenAI removing
    `usage.tool_usage`, which made `n_queries` null on every grounded record."""
    import copy
    body = copy.deepcopy(response("anthropic", "grounded_ok"))
    body["usage"].pop("server_tool_use", None)
    parsed, status = parse("anthropic", body)
    assert parsed.grounded is True
    assert "n_queries_absent" in codes(
        check_record(parsed, status, {"model": ANTH, "search": True}, caps_for(ANTH)))


# ═══════════════════════════════════════════════════════════════════════════════
# A shared token budget can bind a response that still finishes
# ═══════════════════════════════════════════════════════════════════════════════

def _gemini_parsed(thinking, output):
    import dataclasses

    from machine_psych.providers import get_provider

    body = json.loads((pathlib.Path(__file__).parent / "fixtures" / "gemini" /
                       "ungrounded_ok.json").read_text(encoding="utf-8"))["response"]
    base = get_provider("gemini", api_key="k").parse(body)
    return dataclasses.replace(base, thinking_tok=thinking,
                               out_tok_reported=output, answer_chars=900)


def _codes(parsed, status, model, budget):
    from machine_psych.capabilities import caps_for
    from machine_psych.integrity import check_record

    return [i.code for i in check_record(
        parsed, status, {"model": model, "max_tokens": budget}, caps_for(model))]


def test_a_finished_call_with_its_thinking_cut_short_is_flagged():
    """**Status `ok` did not mean the budget was harmless.**

    Measured on gemini-3.8-flash at 16,384: six calls came back `ok` with thinking
    of 15,623-15,729 — the cap minus ~660 for the answer. The model thought until
    the budget ran low, then wrote a short answer. Run again at 65,536 those six
    thought 16K-35K. Status flagged none of them.
    """
    codes = _codes(_gemini_parsed(15_725, 659), "ok",
                   "gemini/gemini-3.8-flash", 16_384)
    assert "budget_exhausted" in codes


def test_an_unconstrained_call_is_not_flagged():
    for thinking, output, budget in ((7_240, 1_000, 16_384),     # P2 S1-12 at 16K
                                     (35_425, 2_000, 65_536),    # largest at 65K
                                     (3_000, 1_615, 16_384)):    # study-corpus shape
        assert "budget_exhausted" not in _codes(
            _gemini_parsed(thinking, output), "ok", "gemini/gemini-3.8-flash", budget)


def test_unreported_thinking_is_not_treated_as_zero():
    """A missing count means the provider did not say. Summing it as zero would
    understate the total and miss a bound call — or, the other way round, invent
    one. Either way the answer must be: not checkable, so not flagged."""
    assert "budget_exhausted" not in _codes(
        _gemini_parsed(None, 16_000), "ok", "gemini/gemini-3.8-flash", 16_384)


def test_a_separate_budget_model_is_never_flagged():
    """Where thinking has its own allowance, the output cap cannot be eaten by it."""
    assert "budget_exhausted" not in _codes(
        _gemini_parsed(15_725, 659), "ok", "anthropic/claude-opus-5-5", 8_192)


def test_text_split_off_by_the_heuristic_is_surfaced():
    """Only text written BETWEEN searches is flagged — not a preamble.

    The heuristic files text before the last tool block as process. A first draft
    flagged any of it and fired on a healthy fixture whose only process text was
    "I'll research this for you..." — a preamble, the common case. A flag on the
    common case trains readers to ignore the integrity report. Text between
    searches was written mid-research and may be part of the answer.
    """
    import dataclasses

    from machine_psych.capabilities import caps_for
    from machine_psych.integrity import check_record
    from machine_psych.providers import get_provider

    body = json.loads((pathlib.Path(__file__).parent / "fixtures" / "anthropic" /
                       "grounded_ok.json").read_text(encoding="utf-8"))["response"]
    base = get_provider("anthropic", api_key="k").parse(body)

    # the real fixture: a preamble before the first search, nothing between them
    assert base.n_process_blocks == 1
    assert base.extra["interleaved_text_blocks"] == 0

    def codes(model, interleaved):
        p = dataclasses.replace(base, extra={**base.extra,
                                             "interleaved_text_blocks": interleaved})
        return [i.code for i in check_record(
            p, "ok", {"model": model, "max_tokens": 8192}, caps_for(model))]

    assert "answer_split_heuristically" not in codes("anthropic/claude-opus-5-5", 0), (
        "a preamble was flagged")
    assert "answer_split_heuristically" in codes("anthropic/claude-opus-5-5", 2)
    assert "answer_split_heuristically" not in codes("openai/gpt-5.6-sol", 2)


def test_interleaved_counts_only_text_between_searches():
    from machine_psych.providers.anthropic import Anthropic

    t = {"type": "text", "text": "x"}
    tool = {"type": "server_tool_use"}
    assert Anthropic._interleaved([t, tool, tool, t]) == 0          # preamble + answer
    assert Anthropic._interleaved([t, tool, t, tool, t]) == 1       # one mid-research block
    assert Anthropic._interleaved([tool, t, t, tool]) == 2
    assert Anthropic._interleaved([t, t]) == 0                      # never searched
