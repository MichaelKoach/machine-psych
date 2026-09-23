"""The outage latch.

Before it existed, the retry budget was five attempts with backoff four — about
forty seconds of tolerance. Any outage longer than that failed the conversation,
then the next, then the next, as fast as the network could refuse them. Under
concurrency sixteen workers each burned their own budget against the same
outage.
"""

from __future__ import annotations

import pytest

from machine_psych.outage import OutageGaveUp, OutageLatch

PROVIDERS = ("anthropic", "openai", "gemini")


def _latch(down=(), recover_after=0, ceiling=12 * 3600, **kw):
    """A latch whose probe fails for `down` until called `recover_after` times."""
    clock, calls = {"t": 0.0}, {"n": 0}

    def probe(p):
        calls["n"] += 1
        return True if p not in down else calls["n"] > recover_after

    latch = OutageLatch(PROVIDERS, probe=probe, ceiling=ceiling,
                        sleep=lambda d: clock.__setitem__("t", clock["t"] + d),
                        clock=lambda: clock["t"], **kw)
    return latch, clock


def _strike(latch, provider, n=3):
    for i in range(n):
        latch.record_failure(provider, f"{provider}-conv{i}", True)


# ═══════════════════════════════════════════════════════════════════════════════
# Scope — network down, or one provider down
# ═══════════════════════════════════════════════════════════════════════════════

def test_one_provider_failing_does_not_stop_the_others():
    latch, _ = _latch(down={"anthropic"}, recover_after=2)
    _strike(latch, "anthropic")

    assert latch.history[-1]["scope"] == "provider"
    assert latch.history[-1]["providers"] == ["anthropic"]
    assert latch._open["openai"].is_set(), "a healthy provider was held"


def test_every_provider_failing_is_read_as_a_network_outage():
    """**Scope is decided by PROBING, not by strike counts.**

    An earlier version compared how many providers had reached the threshold. In
    a real outage all of them fail at once but one gets there first, so a network
    outage was diagnosed as three separate provider outages — wrong in the log
    and wasteful in probes. Probing the others settles it on evidence.
    """
    latch, _ = _latch(down=set(PROVIDERS), recover_after=6)
    _strike(latch, "anthropic")

    assert latch.history[-1]["scope"] == "network"
    assert sorted(latch.history[-1]["providers"]) == sorted(PROVIDERS)


# ═══════════════════════════════════════════════════════════════════════════════
# What counts as evidence
# ═══════════════════════════════════════════════════════════════════════════════

def test_http_errors_are_not_evidence_about_the_network():
    """A 500 or a 429 means the network carried the request and the provider
    answered badly. Different problem, different fix, and latching on it would
    hold the run for exactly the reason it should not."""
    latch, _ = _latch()
    for i in range(10):
        latch.record_failure("anthropic", f"conv{i}", False)
    assert not latch.history


def test_strikes_count_conversations_not_attempts():
    """One conversation retrying five times is one piece of evidence about the
    network, not five. Counting attempts would latch on a single unlucky call."""
    latch, _ = _latch()
    for _ in range(9):
        latch.record_failure("anthropic", "the-same-conversation", True)
    assert not latch.history


def test_a_success_clears_the_strikes():
    """Without this, three failures spread across an entire battery would
    eventually latch a perfectly healthy run."""
    latch, _ = _latch()
    latch.record_failure("anthropic", "c1", True)
    latch.record_failure("anthropic", "c2", True)
    latch.record_success("anthropic")
    latch.record_failure("anthropic", "c3", True)
    assert not latch.history


# ═══════════════════════════════════════════════════════════════════════════════
# Recovery and the ceiling
# ═══════════════════════════════════════════════════════════════════════════════

def test_recovery_releases_the_workers():
    latch, clock = _latch(down={"anthropic"}, recover_after=3)
    _strike(latch, "anthropic")

    assert latch._open["anthropic"].is_set(), "workers still held after recovery"
    assert latch.history[-1]["recovered"] is not None
    assert clock["t"] > 0, "recovered without ever waiting"


def test_the_ceiling_gives_up_cleanly_and_points_at_resume():
    """Twelve hours covers a router reboot, an overnight ISP outage or a machine
    that slept. Past that the run stops so `resume=True` can continue it — the
    ceiling is a handoff, not a loss."""
    latch, clock = _latch(down={"anthropic"}, recover_after=10 ** 9)
    with pytest.raises(OutageGaveUp) as exc:
        _strike(latch, "anthropic")

    assert 11 < clock["t"] / 3600 < 13, f"waited {clock['t'] / 3600:.1f}h"
    assert "resume=True" in str(exc.value)
    assert latch._open["anthropic"].is_set(), (
        "workers left waiting on an event nobody will set again")


def test_a_probe_that_raises_does_not_escape_into_a_worker():
    """An exception here would kill a run that was only waiting."""
    clock = {"t": 0.0}
    latch = OutageLatch(("anthropic",),
                        probe=lambda p: (_ for _ in ()).throw(RuntimeError("dns")),
                        ceiling=60,
                        sleep=lambda d: clock.__setitem__("t", clock["t"] + d),
                        clock=lambda: clock["t"])
    with pytest.raises(OutageGaveUp):
        _strike(latch, "anthropic")


def test_waiting_is_free_when_nothing_is_wrong():
    latch, _ = _latch()
    for p in PROVIDERS:
        latch.wait(p)          # returns immediately or the test hangs
    latch.wait("a-provider-not-in-this-run")


# ═══════════════════════════════════════════════════════════════════════════════
# The probe itself
# ═══════════════════════════════════════════════════════════════════════════════

def test_any_http_answer_counts_as_reachable():
    """**A 429 is emphatic proof the provider is up.** The question is whether
    the network carries a request and something answers, not whether the answer
    is useful."""
    import requests

    from machine_psych.providers import get_provider

    class Limited:
        status_code, headers = 429, {}
        def json(self):
            return {"error": {"type": "rate_limit_error"}}

    real = requests.post
    requests.post = lambda *a, **k: Limited()
    try:
        assert get_provider("anthropic", api_key="k").reachable(
            "anthropic/claude-sonnet-5") is True
    finally:
        requests.post = real


def test_a_connection_error_is_not_reachable():
    import requests

    from machine_psych.providers import get_provider

    real = requests.post

    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError("no route to host")

    requests.post = refuse
    try:
        assert get_provider("anthropic", api_key="k").reachable(
            "anthropic/claude-sonnet-5") is False
    finally:
        requests.post = real


# ═══════════════════════════════════════════════════════════════════════════════
# In the runner
# ═══════════════════════════════════════════════════════════════════════════════

def _flaky_run(tmp_path, outage_len, ceiling=600):
    """A 12-record battery that meets a connection outage partway through."""
    import contextlib
    import io
    import json as _json
    import pathlib as _pl

    import requests

    import machine_psych as mp
    from machine_psych import paths
    from machine_psych.providers import get_provider

    paths.set_api_key("anthropic", "k")
    ok = _json.loads((_pl.Path(__file__).parent / "fixtures" / "anthropic" /
                      "ungrounded_ok.json").read_text())["response"]
    n = {"c": 0}

    class Ok:
        status_code, headers = 200, {}
        def json(self):
            return ok

    def flaky(*a, **k):
        n["c"] += 1
        if 4 <= n["c"] < 4 + outage_len:
            raise requests.exceptions.ConnectionError("network down")
        return Ok()

    clock = {"t": 0.0}
    latch = OutageLatch(
        ("anthropic",), ceiling=ceiling,
        probe=lambda name: get_provider("anthropic", api_key="k").reachable(
            "anthropic/claude-sonnet-5"),
        sleep=lambda d: clock.__setitem__("t", clock["t"] + d),
        clock=lambda: clock["t"])

    real = requests.post
    requests.post = flaky
    try:
        mp.set_base(tmp_path)
        spec = {"investigation_id": "o", "studies": [{"study_id": "s",
                "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                           for i in range(6)],
                "providers": {"anthropic/claude-sonnet-5":
                              {"reasoning": "off", "repetitions": 2}}}]}
        with contextlib.redirect_stdout(io.StringIO()):
            run = mp.load_investigation(spec)
            res, _ = mp.run_investigation(run, backoff=0, retries=5,
                                          verbose=False, export=False, latch=latch)
    finally:
        requests.post = real
    return res, latch


def test_a_short_outage_is_absorbed_without_latching(tmp_path):
    """The retry layer handles brief drops. The latch is for the case retries
    cannot reach, and firing it on every blip would stall a healthy run."""
    res, latch = _flaky_run(tmp_path, outage_len=6)
    assert len(res) == 12
    assert (res.status == "ok").sum() >= 10
    assert not latch.history, "latched on an outage the retries already covered"


def test_a_long_outage_stops_the_run_instead_of_burning_it(tmp_path):
    """**The behaviour this exists for.**

    Before the latch, an outage past the ~40s retry budget failed the
    conversation, then the next, then the next — the whole remaining plan, as
    fast as the network could refuse it. Now the run stops, the records already
    collected stay on disk, and `resume=True` continues it.
    """
    res, latch = _flaky_run(tmp_path, outage_len=500, ceiling=600)

    assert latch.history, "never latched"
    assert res.attrs.get("interrupted"), "did not report itself stopped"
    assert 0 < len(res) < 12, f"expected a partial run, got {len(res)}"

    import pathlib as _pl
    on_disk = list(_pl.Path(tmp_path).rglob("[0-9]*.json"))
    assert len(on_disk) == len(res), "records in memory and on disk disagree"


def test_giving_up_is_a_stop_rather_than_a_traceback(tmp_path):
    """`OutageGaveUp` reaching the caller would discard everything collected in
    memory and print a traceback for a condition the run handled correctly. It
    is grouped with KeyboardInterrupt, not with bugs."""
    res, _ = _flaky_run(tmp_path, outage_len=500, ceiling=1)
    assert res.attrs.get("interrupted")
