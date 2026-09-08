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
from machine_psych.integrity import Issue, check_record, describe_integrity
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
    assert [i for i in issues if i.code == "served_model_mismatch"][0].severity == "critical"


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
    That capability disappearing would end the scraping layer's core question."""
    body = copy.deepcopy(response("anthropic"))
    body["content"] = [b for b in body["content"]
                       if b.get("type") != "web_search_tool_result"]
    parsed, status = parse("anthropic", body)
    assert "retrieval_set_absent" in codes(check_record(parsed, status, INTENT, caps_for(ANTH)))


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
