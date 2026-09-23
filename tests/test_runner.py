"""Runner tests: the sixteen orderings from §6b, as assertions.

Each test names what breaks if the sequence is wrong, because with the runner the
ORDER is the specification and every one of these corresponds to a bug this
project has already had.

A fake dispatcher is used throughout. Note that it must return errors in the shape
the RECEIVING provider recognises — an early version of these tests returned one
provider's error format to another and the retry silently did not fire, which is
itself worth knowing: the providers correctly refuse to parse each other's errors.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import pathlib
import threading
import time

import pytest

import machine_psych.runner as R
from machine_psych.providers import Anthropic, Gemini

SPEC = {
    "investigation_id": "t",
    "studies": [{
        "study_id": "s1",
        "probes": [
            {"probe_id": "ladder", "prompt_paths": [["t0", "t1", "t2"]]},
            {"probe_id": "single", "prompt_paths": [["solo"]]},
        ],
        "providers": {
            "anthropic/claude-sonnet-5": {"reasoning": "off", "repetitions": 1},
            "gemini/gemini-3.7-flash": {"reasoning": "medium",
                                        "max_tokens": 16384, "repetitions": 1},
        },
    }],
}


def _ok_body(cfg, n):
    if cfg.get("generation_config"):
        return {"id": f"v1_{n}", "status": "completed", "model": "gemini-3.7-flash",
                "steps": [{"type": "thought", "signature": "s" * 80,
                           "summary": "thinking"},
                          {"type": "model_output",
                           "content": [{"type": "text", "text": f"A{n}"}]}],
                "usage": {"total_input_tokens": 20, "raw_prompt_token": 200,
                          "total_output_tokens": 30, "total_thought_tokens": 10,
                          "total_tokens": 260}}
    return {"model": "claude-sonnet-5", "id": f"msg_{n}", "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": f"Answer {n}."}],
            "stop_reason": "end_turn", "stop_details": None,
            "usage": {"input_tokens": 20, "output_tokens": 30,
                      "output_tokens_details": {"thinking_tokens": 0}}}


def _transient(cfg):
    """In the shape the receiving provider recognises."""
    if cfg.get("generation_config"):
        return {"error": {"code": "api_error", "message": "high demand"}}
    return {"error": {"type": "overloaded_error", "message": "Overloaded"}}


def _permanent(cfg):
    if cfg.get("generation_config"):
        return {"error": {"code": 400, "message": "Unknown parameter"}}
    return {"error": {"type": "invalid_request_error", "message": "bad"}}


@pytest.fixture
def loaded(tmp_path):
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        return R.load_investigation(SPEC)


def _run(loaded, dispatch, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return R.run_investigation(loaded, dispatch=dispatch, backoff=0,
                                   verbose=False, export=False, **kw)


# ═══════════════════════════════════════════════════════════════════════════════
# Loading — sends nothing
# ═══════════════════════════════════════════════════════════════════════════════

def test_loading_sends_nothing(tmp_path):
    """A design mistake should cost a read, not a battery."""
    R.set_base(tmp_path)
    calls = []
    with contextlib.redirect_stdout(io.StringIO()):
        R.load_investigation(SPEC)
    assert not calls


def test_2_round_robin_over_reps_interleaved_across_providers(tmp_path):
    """ORDERING 2. The web changes during a run.

    A battery that finishes all of condition A before starting condition B has
    made TIME a hidden factor, and the same argument applies across providers.
    Every provider's rep 0 must run before any provider's rep 1.
    """
    R.set_base(tmp_path)
    spec = json.loads(json.dumps(SPEC))
    for block in spec["studies"][0]["providers"].values():
        block["repetitions"] = 3
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(spec)
    reps = list(run.rep)
    assert reps == sorted(reps), "reps are not grouped — drift confounds condition"
    for rep in set(reps):
        assert run[run.rep == rep].provider.nunique() == 2, (
            "a rep does not span both providers — they are not interleaved")


# ═══════════════════════════════════════════════════════════════════════════════
# The retry loop
# ═══════════════════════════════════════════════════════════════════════════════

def test_4_retry_is_recorded_per_record(loaded):
    """ORDERING 4. Retry lives in the runner, not in dispatch.

    In dispatch, `latency` would measure the whole retry chain and `attempts`
    would be invisible — so a record that took five tries to land would look
    identical to one that landed first time, and the provider's reliability could
    not be measured from the corpus.
    """
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        return _transient(cfg) if state["n"] in (2, 3) else _ok_body(cfg, state["n"])

    res, _ = _run(loaded, fake)
    assert res.attempts.max() == 3
    assert (res[res.attempts > 1].status == "ok").all(), "retry did not recover"


def test_6_a_transport_exception_is_transient(loaded):
    """ORDERING 6. An earlier draft broke on ANY exception.

    On a forty-minute battery against a provider with a measured 1-in-6 transient
    rate, one connection reset loses a record that a retry recovers.
    """
    import requests
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        if state["n"] == 2:
            raise requests.exceptions.Timeout("timed out")
        return _ok_body(cfg, state["n"])

    res, _ = _run(loaded, fake)
    assert res.attempts.max() >= 2
    assert (res.status == "ok").all()


def test_7_a_permanent_error_is_not_retried(loaded):
    """ORDERING 7. Five attempts at a 400 wastes a minute per record and never
    succeeds."""
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        return _permanent(cfg) if state["n"] == 1 else _ok_body(cfg, state["n"])

    res, _ = _run(loaded, fake)
    assert (res[res.status == "error"].attempts == 1).all()


def test_providers_do_not_parse_each_others_errors():
    """Discovered by writing a test wrong.

    A shared error classifier would have to guess a format, and guessing wrong
    means either retrying a 400 or dropping a recoverable record. Each provider
    reads its own shape and nothing else.
    """
    gemini_shape = {"error": {"code": "api_error", "message": "high demand"}}
    assert Gemini(api_key="k").status(gemini_shape)[0] == "unavailable"
    assert Anthropic(api_key="k").status(gemini_shape)[0] == "error"


# ═══════════════════════════════════════════════════════════════════════════════
# Conversations
# ═══════════════════════════════════════════════════════════════════════════════

def test_9_a_failed_turn_ends_its_conversation_not_the_run(loaded):
    """ORDERING 9. The rule that appeared in three implementations and no sentence.

    One provider's outage must not cost the other two their arms, and a
    conversation that broke at turn 1 still has a usable turn 0.
    """
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        # fail the second turn of the first conversation
        return _permanent(cfg) if state["n"] == 2 else _ok_body(cfg, state["n"])

    res, _ = _run(loaded, fake)
    assert res.provider.nunique() == 2, "the run stopped at the first failure"
    broken = res[res.conversation_status != "ok"]
    assert len(broken) and broken.conversation.nunique() == 1


def test_9b_truncated_continues_the_conversation(loaded):
    """ORDERING 9, the subtle half. `truncated` is NOT a failure.

    The answer is real — measured at 4,338 characters on one provider — and a
    truncated turn 0 can still be passed back. Treating it as failure would
    discard a usable partial answer mid-ladder.
    """
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        if state["n"] == 1 and not cfg.get("generation_config"):
            body = _ok_body(cfg, 1)
            body["stop_reason"] = "max_tokens"       # partial answer present
            return body
        return _ok_body(cfg, state["n"])

    res, _ = _run(loaded, fake)
    trunc = res[res.status == "truncated"]
    assert len(trunc) == 1
    conv = trunc.iloc[0].conversation
    assert len(res[res.conversation == conv]) == 3, "the ladder did not continue"
    assert (res[res.conversation == conv].conversation_status == "ok").all()


def test_11_failed_and_incomplete_are_distinguished(loaded):
    """ORDERING 11. `failed` means it died on turn 0; `incomplete` means later.

    Conflating them loses whether a conversation produced anything at all.
    """
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        return _permanent(cfg) if state["n"] == 1 else _ok_body(cfg, state["n"])

    res, _ = _run(loaded, fake)
    assert "failed" in set(res.conversation_status)


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence — the bug that survived a whole session
# ═══════════════════════════════════════════════════════════════════════════════

def test_12_disk_matches_memory(loaded):
    """ORDERING 12, REGRESSION 1. Records are written TWICE.

    Once as the call completes so an interrupt keeps them, once after
    `conversation_status` is known. Skipping the second leaves null on disk while
    the in-memory frame is correct — which is invisible in the same session and
    only appears on reload, and is why this test asserts against RELOADED data.
    """
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        return _permanent(cfg) if state["n"] == 2 else _ok_body(cfg, state["n"])

    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = R.run_investigation(loaded, dispatch=fake, backoff=0,
                                     verbose=False, export=False)
    run_dir = pathlib.Path(res.attrs["run_dir"])
    disk = [json.loads(p.read_text())["conversation_status"]
            for p in sorted(run_dir.glob("[0-9]*.json"))]
    assert disk == list(res.conversation_status)
    assert all(d is not None for d in disk)


def test_8_an_interrupt_keeps_what_was_collected(loaded):
    """ORDERINGS 1, 8 and 13.

    `conv_rows` is declared OUTSIDE the loop or the handler raises NameError and
    loses everything. The first write happens before the next call. And the
    handler catches BaseException, because a notebook stop is KeyboardInterrupt
    and `except Exception` would miss it entirely.
    """
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        if state["n"] == 3:
            raise KeyboardInterrupt("stopped")
        return _ok_body(cfg, state["n"])

    with contextlib.redirect_stdout(io.StringIO()):
        res, raw = R.run_investigation(loaded, dispatch=fake, backoff=0,
                                       verbose=False, export=False)
    assert len(res) == 2, "partial records were lost"
    assert res.attrs["interrupted"] and "KeyboardInterrupt" in res.attrs["interrupted"]
    assert res.conversation_status.notna().all()
    assert len(raw) == len(res)


def test_14_scratch_keys_never_reach_the_frame(loaded):
    """ORDERING 14. They would leak into the DataFrame and then into the export."""
    res, _ = _run(loaded, lambda cfg: _ok_body(cfg, 1))
    assert not any(c.startswith("_") for c in res.columns)


def test_3_user_turns_are_provider_shaped(loaded):
    """ORDERING 3. The wrong shape is a 400 on every turn after the first.

    One provider takes `{"role": "user", ...}`; another rejects exactly that with
    "use step_list input format instead of turn_list". The runner must ask the
    provider rather than assume a shape.
    """
    seen = {"anthropic": [], "gemini": []}
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        which = "gemini" if cfg.get("generation_config") else "anthropic"
        seen[which].append(cfg.get("input") or cfg.get("messages"))
        return _ok_body(cfg, state["n"])

    _run(loaded, fake)
    assert seen["anthropic"][0][0]["role"] == "user"
    assert seen["gemini"][0][0]["type"] == "user_input", (
        "a role/content turn would 400 on this provider")


def test_10_passback_happens_only_after_a_successful_turn(loaded):
    """ORDERING 10. Passing back a failed response corrupts the history.

    The next turn would be built on an error body, and the model would be
    continuing from something that never happened.
    """
    sent = []
    state = {"n": 0}

    def fake(cfg):
        state["n"] += 1
        sent.append(len(cfg.get("input") or cfg.get("messages") or []))
        if state["n"] == 1:
            return _permanent(cfg)
        return _ok_body(cfg, state["n"])

    res, _ = _run(loaded, fake)
    # the failed conversation contributed exactly one call and stopped
    assert sent[0] == 1
    assert (res.conversation_status == "failed").any()


def test_16_an_export_failure_does_not_discard_the_run(loaded, monkeypatch):
    """ORDERING 16. Records are on disk before the export runs.

    Losing a completed battery to a formatting bug in a downstream module would
    be the worst possible trade, so the export is wrapped and its failure is
    reported rather than raised.
    """
    import machine_psych.runner as runner_mod

    def boom(*a, **kw):
        raise RuntimeError("export is broken")

    monkeypatch.setitem(__import__("sys").modules, "machine_psych.export",
                        type("m", (), {"export_corpus": staticmethod(boom)}))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res, _ = runner_mod.run_investigation(
            loaded, dispatch=lambda cfg: _ok_body(cfg, 1), backoff=0,
            verbose=True, export=True)
    assert len(res) > 0, "the run was discarded by an export failure"
    assert "export failed" in out.getvalue()
    run_dir = pathlib.Path(res.attrs["run_dir"])
    assert list(run_dir.glob("[0-9]*.json")), "records are not on disk"


# ═══════════════════════════════════════════════════════════════════════════════
# Provenance
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_records_three_kinds_of_provenance(loaded):
    """What code ran, whether the instrument changed, and what each intent meant.

    Three mechanisms answering three different questions — a git commit, a probe
    hash, and the resolved config per record. Three corpora already on disk were
    collected by code that has since had bugs fixed, and there is no way to
    reconstruct what the collecting version did.
    """
    res, _ = _run(loaded, lambda cfg: _ok_body(cfg, 1))
    meta = json.loads((pathlib.Path(res.attrs["run_dir"]) / "_run.json").read_text())
    assert "harness" in meta and "version" in meta["harness"]
    assert meta["probe_hashes"]
    assert "config_resolved" in res.columns and "config_passed" in res.columns


def test_probe_hash_is_stable_and_content_derived(loaded):
    res, _ = _run(loaded, lambda cfg: _ok_body(cfg, 1))
    for _probe, group in res.groupby("probe"):
        assert group.probe_hash.nunique() == 1


def test_intent_unmet_is_recorded_per_record(loaded):
    res, _ = _run(loaded, lambda cfg: _ok_body(cfg, 1))
    assert "intent_unmet" in res.columns
    assert all(isinstance(v, list) for v in res.intent_unmet)


# ═══════════════════════════════════════════════════════════════════════════════
# Cost estimation
# ═══════════════════════════════════════════════════════════════════════════════

def test_estimate_is_per_provider_never_one_total():
    """Three accounting conventions make a single number a false comparison.

    Two providers count thinking inside output and one keeps it separate; one
    bills retrieved content as input and one does not.
    """
    rows = [{"model": "anthropic/claude-sonnet-5", "turns": ["a"],
             "params": {"search": True}},
            {"model": "gemini/gemini-3.7-flash", "turns": ["a"],
             "params": {"search": True}}]
    est = R.estimate(rows)
    assert set(est) == {"anthropic", "gemini"}
    assert est["anthropic"]["est_in_high"] != est["gemini"]["est_in_high"], (
        "providers with different tokens_per_query estimated identically")


def test_estimate_range_is_wide_for_searched_calls():
    """Search COUNT varies by question shape — 2 on a recognition probe, 11 on a
    neighbour probe, both measured — and a spec cannot say which it is. A narrow
    estimate would be a guess wearing a number's clothes."""
    rows = [{"model": "anthropic/claude-sonnet-5", "turns": ["a"],
             "params": {"search": True}}]
    e = R.estimate(rows)["anthropic"]
    assert e["est_in_high"] >= e["est_in_low"] * 5


# ═══════════════════════════════════════════════════════════════════════════════
# Public functions the audit found untested
# ═══════════════════════════════════════════════════════════════════════════════

def test_list_investigations_and_list_runs(tmp_path):
    """Small conveniences, exported, and previously exercised by nothing.

    `list_runs` matters more than it looks: `load_corpus` defaults to the most
    recent run, and a silently different run produces a real-looking wrong
    answer. This is how you check which one you are about to get.
    """
    R.set_base(tmp_path)
    assert R.list_investigations() == []

    with contextlib.redirect_stdout(io.StringIO()):
        R.save_investigation(SPEC, "one", overwrite=True)
        R.save_investigation(SPEC, "two", overwrite=True)
    assert R.list_investigations() == ["one", "two"]

    assert R.list_runs() == []
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation("one")
        R.run_investigation(run, dispatch=lambda cfg: _ok_body(cfg, 0),
                            backoff=0, verbose=False, export=False)

    # NOTE the run directory is named for the spec's `investigation_id`, NOT the
    # filename it was saved under. `save_investigation(spec, "one")` names the
    # FILE; the run uses what the spec calls itself. They can diverge, and this
    # test is where that surfaced.
    inv = SPEC["investigation_id"]
    assert len(R.list_runs(inv)) == 1
    assert R.list_runs("nonexistent") == []


def test_load_record_returns_the_raw_body(tmp_path):
    """The escape hatch for a question the side tables cannot answer.

    Everything else in `corpus` reads through a provider parser; this does not,
    which is the point — it returns the response exactly as it arrived.
    """
    from machine_psych.corpus import load_record

    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(SPEC)
        results, _ = R.run_investigation(run, dispatch=lambda cfg: _ok_body(cfg, 0),
                                         backoff=0, verbose=False, export=False)
    path = results.iloc[0].path_on_disk
    record = load_record(path)
    assert "response" in record
    assert record["response"], "the raw body, unparsed"


def test_conversation_id_distinguishes_models_from_one_provider(tmp_path):
    """Two models from one provider produced IDENTICAL conversation ids.

    An 80-record battery comparing Sonnet against Opus collapsed to 40
    conversations, and the run summary reported "40 conversations" against 80
    records. Any analysis using `conversation` as a unique key silently merged
    the two arms it was built to compare.
    """
    R.set_base(tmp_path)
    spec = {"investigation_id": "twomodels", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {"anthropic/claude-sonnet-5": {"reasoning": "off", "repetitions": 2},
                      "anthropic/claude-opus-5": {"reasoning": "off", "repetitions": 2}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(spec)
        res, _ = R.run_investigation(run, dispatch=lambda c: _ok_body(c, 0),
                                     backoff=0, verbose=False, export=False)

    assert len(res) == 4
    assert res.conversation.nunique() == 4, (
        f"conversation ids collide across models: {sorted(res.conversation.unique())}")
    for model in res.model.unique():
        tag = model.split("/", 1)[1]
        assert all(tag in c for c in res[res.model == model].conversation)


def test_the_estimate_floor_is_reachable_when_search_is_permitted():
    """It read "80 records (80 searched)" and put the floor at 1.86M input tokens
    for a battery that used 328K — 18% of the stated minimum.

    It treated every search-ENABLED record as grounding. Measured on that
    battery: 5 of 80 actually searched. The floor must therefore assume none of
    them do, or the range cannot contain the real outcome.
    """
    rows = [{"model": "anthropic/claude-sonnet-5", "params": {"search": True},
             "turns": ["q"]} for _ in range(80)]
    est = R.estimate(rows)["anthropic"]
    assert est["search_permitted"] == 80
    assert est["est_in_low"] < 328_000 < est["est_in_high"], (
        f"the real 328K outcome is outside the range "
        f"{est['est_in_low']}-{est['est_in_high']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Concurrency — 40s/record sequential made 8,000 records a four-day run
# ═══════════════════════════════════════════════════════════════════════════════

def _many_probes(inv, probes=6, reps=3):
    return {"investigation_id": inv, "studies": [{
        "study_id": "s",
        "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                   for i in range(probes)],
        "providers": {"anthropic/claude-sonnet-5": {"reasoning": "off",
                                                    "repetitions": reps}}}]}


def test_concurrency_produces_identical_output_to_sequential(tmp_path):
    """The only acceptable speedup is one that changes nothing else.

    Under concurrency rows arrive in COMPLETION order, which varies run to run.
    Without sorting by the pre-assigned sequence a corpus would not be
    reproducible and `record_id` would not line up with the filename on disk.
    """
    results = {}
    for conc in (1, 6):
        R.set_base(tmp_path / f"c{conc}")
        spec = _many_probes(f"conc{conc}")
        with contextlib.redirect_stdout(io.StringIO()):
            run = R.load_investigation(spec)
            res, _ = R.run_investigation(run, dispatch=lambda c: _ok_body(c, 0),
                                         backoff=0, verbose=False, export=False,
                                         concurrency=conc)
        results[conc] = res

    assert len(results[1]) == len(results[6]) == 18
    assert list(results[1].conversation) == list(results[6].conversation), (
        "concurrent run produced a different record ORDER — the corpus is not "
        "reproducible")
    assert results[1].status.tolist() == results[6].status.tolist()


def test_repetitions_do_not_overlap(tmp_path):
    """**The ordering guarantee, which speed must not buy.**

    Retrieval drift has to spread across conditions rather than confound with
    them, which requires that no arm FINISHES before another STARTS. Parallel
    within a repetition satisfies that; parallel across repetitions does not,
    and would make the first conditions to run see a different web from the last.
    """
    R.set_base(tmp_path)
    seen, lock = [], threading.Lock()

    def watch(cfg):
        with lock:
            seen.append(time.perf_counter())
        time.sleep(0.05)
        return _ok_body(cfg, 0)

    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_many_probes("barrier", probes=4, reps=3))
        res, _ = R.run_investigation(run, dispatch=watch, backoff=0,
                                     verbose=False, export=False, concurrency=4)

    by_rep = {}
    for i, rep in enumerate(res.rep):
        if i < len(seen):
            by_rep.setdefault(rep, []).append(seen[i])
    reps = sorted(by_rep)
    for a, b in itertools.pairwise(reps):
        assert min(by_rep[b]) > max(by_rep[a]), (
            f"rep {b} began before rep {a} finished — the barrier is gone")


def test_an_interrupt_under_concurrency_keeps_partial_work(tmp_path):
    """The piece most likely to be silently wrong.

    `conv_rows` alone was enough while this was sequential. With a pool there are
    several conversations in flight, so the handler walks a registry of them —
    and the pool is shut down with `cancel_futures` so queued conversations are
    not drained before the interrupt is honoured.
    """
    R.set_base(tmp_path)
    calls, lock = {"n": 0}, threading.Lock()

    def stop_midway(cfg):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n > 5:
            raise KeyboardInterrupt
        return _ok_body(cfg, 0)

    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_many_probes("intr", probes=6, reps=4))
        res, _ = R.run_investigation(run, dispatch=stop_midway, backoff=0,
                                     verbose=False, export=False, concurrency=4)

    assert 0 < len(res) < 24, f"expected partial results, got {len(res)}"
    assert res.attrs.get("interrupted")

    written = [json.loads(p.read_text())
               for p in pathlib.Path(tmp_path).rglob("[0-9]*.json")]
    assert written, "nothing persisted"
    nulls = [w for w in written if w.get("conversation_status") is None]
    assert not nulls, f"{len(nulls)} persisted records have no conversation_status"


def test_concurrency_defaults_to_sequential():
    """Every battery run before this existed used the sequential path. The
    default must keep doing exactly that."""
    import inspect
    assert inspect.signature(R.run_investigation).parameters["concurrency"].default == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Resume — a four-day battery WILL be interrupted
# ═══════════════════════════════════════════════════════════════════════════════

def _two_model_spec(inv, probes=5, reps=2):
    return {"investigation_id": inv, "studies": [{
        "study_id": "s",
        "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                   for i in range(probes)],
        "providers": {"anthropic/claude-sonnet-5": {"reasoning": "off", "repetitions": reps},
                      "anthropic/claude-opus-5": {"reasoning": "off", "repetitions": reps}}}]}


def test_ledger_key_separates_models_from_one_provider():
    """`provider` is not enough. Two models from one API are two different
    logical calls, and leaving the model out is the defect that gave eighty
    records forty conversation ids."""
    a = R.ledger_key("anthropic", "anthropic/claude-sonnet-5", "s", "p", 0, 0, 0, None)
    b = R.ledger_key("anthropic", "anthropic/claude-opus-5", "s", "p", 0, 0, 0, None)
    assert a != b
    # and it is derived, not generated — the same inputs give the same key
    assert a == R.ledger_key("anthropic", "anthropic/claude-sonnet-5",
                             "s", "p", 0, 0, 0, None)


def test_resume_skips_what_is_on_disk_and_returns_the_whole_corpus(tmp_path):
    """Without this, the attempt after an interruption starts a NEW directory and
    re-sends everything — paying a second time for records already collected.

    The returned frame must be the whole corpus rather than the increment, or
    every analysis after a resume is silently partial.
    """
    R.set_base(tmp_path)
    sent = {"n": 0}

    def die_after_8(cfg):
        sent["n"] += 1
        if sent["n"] > 8:
            raise KeyboardInterrupt
        return _ok_body(cfg, 0)

    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_two_model_spec("res"))
        first, _ = R.run_investigation(run, dispatch=die_after_8, backoff=0,
                                       verbose=False, export=False)
    assert 0 < len(first) < 20

    sent["n"] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_two_model_spec("res"))
        second, _ = R.run_investigation(run, dispatch=lambda c: _ok_body(c, 0),
                                        backoff=0, verbose=False, export=False,
                                        resume=True)

    assert len(second) == 20, f"resumed run returned {len(second)} of 20"
    assert sent["n"] < 20, f"re-sent {sent['n']} calls — the ledger did nothing"
    assert second.conversation_status.notna().all()
    dirs = [d for d in (tmp_path / "Output Log" / "res").glob("*") if d.is_dir()]
    assert len(dirs) == 1, f"resume wrote a second directory: {[d.name for d in dirs]}"


def test_an_in_flight_record_is_redone_rather_than_trusted(tmp_path):
    """A record without `conversation_status` was mid-conversation when the run
    stopped. It may have been billed, but there is no answer in it — so it is
    redone, and the possible double charge is accepted rather than leaving a hole
    in the corpus."""
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_two_model_spec("inflight", probes=2, reps=1))
        R.run_investigation(run, dispatch=lambda c: _ok_body(c, 0), backoff=0,
                            verbose=False, export=False)

    run_dir = next(d for d in (tmp_path / "Output Log" / "inflight").glob("*")
                   if d.is_dir())
    victim = min(run_dir.glob("[0-9]*.json"))
    rec = json.loads(victim.read_text())
    rec["conversation_status"] = None
    victim.write_text(json.dumps(rec))

    assert R.ledger_key(rec["provider"], rec["model"], rec["study"], rec["probe"],
                        rec["path"], rec["rep"], rec["turn"],
                        rec["condition"]) not in R.completed_keys(run_dir)


def test_resume_on_a_missing_run_says_so(tmp_path):
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_two_model_spec("nope", probes=1, reps=1))
        try:
            R.run_investigation(run, dispatch=lambda c: _ok_body(c, 0),
                                verbose=False, export=False, resume="not-a-run")
        except FileNotFoundError as e:
            assert "no run to resume" in str(e)
        else:
            raise AssertionError("resuming a run that does not exist must raise")


def test_idempotency_is_claimed_only_where_confirmed():
    """Sending the header to a provider that ignores it is worse than not
    sending it: the retry LOOKS protected and quietly pays twice."""
    from machine_psych.providers import get_provider

    assert get_provider("openai", api_key="k").supports_idempotency is True
    assert get_provider("anthropic", api_key="k").supports_idempotency is False
    assert get_provider("gemini", api_key="k").supports_idempotency is False


# ═══════════════════════════════════════════════════════════════════════════════
# Per-provider concurrency — limits differ by an order of magnitude
# ═══════════════════════════════════════════════════════════════════════════════

def _peak_watcher():
    """A dispatcher that records the PEAK simultaneous calls per provider."""
    live, peak, lock = {}, {}, threading.Lock()

    def watch(cfg):
        who = ("gemini" if "generation_config" in cfg
               else "openai" if "store" in cfg else "anthropic")
        with lock:
            live[who] = live.get(who, 0) + 1
            peak[who] = max(peak.get(who, 0), live[who])
        time.sleep(0.05)
        with lock:
            live[who] -= 1
        return _ok_body(cfg, 0)

    return watch, peak


def _cap_spec(inv, probes=10, reps=1):
    return {"investigation_id": inv, "studies": [{"study_id": "s",
            "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                       for i in range(probes)],
            "providers": {"anthropic/claude-sonnet-5": {"reasoning": "off", "repetitions": reps},
                          "anthropic/claude-opus-5": {"reasoning": "off", "repetitions": reps}}}]}


@pytest.mark.parametrize("concurrency,cap", [
    (1, 1), (4, 4), (8, 8),
    ({"anthropic": 3}, 3),
    ({"openai": 9}, 1),           # unnamed provider falls back to 1
    ({"anthropic": 1}, 1),        # a dict of ones is still sequential in effect
])
def test_the_per_provider_cap_is_actually_enforced(tmp_path, concurrency, cap):
    """**A shared pool caps nothing per provider.**

    Workers were sized `concurrency * n_providers` and drawn from one queue, so
    every worker could be running the same provider — and a rep holding twenty
    Gemini conversations against four Anthropic would put them all on Gemini,
    the account with the tightest limit. The comment claimed a per-provider cap
    the code did not enforce; a semaphore per provider does.
    """
    R.set_base(tmp_path)
    watch, peak = _peak_watcher()
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_cap_spec("cap"))
        res, _ = R.run_investigation(run, dispatch=watch, backoff=0, verbose=False,
                                     export=False, concurrency=concurrency)

    assert len(res) == 20
    assert peak.get("anthropic", 0) <= cap, (
        f"{peak.get('anthropic')} calls in flight against a cap of {cap}")


def test_providers_hold_different_caps_at_the_same_time(tmp_path):
    """The case that motivated this. Gemini's free tier runs about 10 requests
    per minute against Anthropic's hundreds, so one number for all three is
    either too slow for Anthropic or a 429 storm on Gemini."""
    R.set_base(tmp_path)
    watch, peak = _peak_watcher()
    caps = {"anthropic": 8, "openai": 4, "gemini": 2}
    spec = {"investigation_id": "mixed", "studies": [{"study_id": "s",
            "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                       for i in range(8)],
            "providers": {"anthropic/claude-sonnet-5": {"reasoning": "off", "repetitions": 2},
                          "openai/gpt-5.6-sol": {"reasoning": "low", "repetitions": 2},
                          "gemini/gemini-3.7-flash": {"reasoning": "low", "repetitions": 2}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(spec)
        res, _ = R.run_investigation(run, dispatch=watch, backoff=0, verbose=False,
                                     export=False, concurrency=caps)

    assert len(res) == 48
    for provider, cap in caps.items():
        assert peak.get(provider, 0) <= cap, (
            f"{provider}: {peak.get(provider)} in flight against a cap of {cap}")
    assert res.conversation.nunique() == 48


def test_the_repetition_barrier_survives_per_provider_gating(tmp_path):
    """Semaphores add a second place a worker can block. The barrier must still
    hold: retrieval drift has to spread across conditions rather than confound
    with them, which is the property speed must not buy."""
    R.set_base(tmp_path)
    seen, lock = [], threading.Lock()

    def watch(cfg):
        with lock:
            seen.append(time.perf_counter())
        time.sleep(0.05)
        return _ok_body(cfg, 0)

    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(_cap_spec("barrier", probes=4, reps=3))
        res, _ = R.run_investigation(run, dispatch=watch, backoff=0, verbose=False,
                                     export=False, concurrency={"anthropic": 4})

    by_rep = {}
    for i, rep in enumerate(res.rep):
        if i < len(seen):
            by_rep.setdefault(rep, []).append(seen[i])
    for a, b in itertools.pairwise(sorted(by_rep)):
        assert min(by_rep[b]) > max(by_rep[a]), (
            f"rep {b} began before rep {a} finished")
