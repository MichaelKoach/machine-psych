"""Pacing against the limits each provider reports.

A fixed concurrency number is a guess. This project tried to measure one and
could not: batteries at 8, 16 and 32 all returned 158 records a minute, because
the battery held six conversations per repetition and the extra workers were
idle. **That was the design ceiling, not a rate limit** — the real one was never
found.
"""

from __future__ import annotations

import pytest

from machine_psych.governor import COLD_START, Governor, pool_key
from machine_psych.ratelimit import RateLimits


@pytest.fixture
def clock():
    t = {"now": 0.0}
    return t


@pytest.fixture
def gov(clock):
    return Governor(clock=lambda: clock["now"],
                    sleep=lambda d: clock.__setitem__("now", clock["now"] + d))


# ═══════════════════════════════════════════════════════════════════════════════
# Pacing
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_sustained_rate_matches_the_margin_adjusted_limit(gov, clock):
    """**Bursting first is correct, not a bug.**

    Anthropic's limiter is a token bucket: capacity accumulates, so a short burst
    above the sustained rate is allowed if the capacity is there. Mirroring the
    algorithm means predicting the same answer the server will give. What must
    hold is the SUSTAINED rate once the bucket is empty.
    """
    gov.observe("anthropic", RateLimits(rpm_limit=30, rpm_remaining=30))

    sent = []
    for _ in range(40):
        gov.acquire("anthropic", est_in=2_000, est_out=1_000)
        sent.append(clock["now"])
        gov.release("anthropic")

    burst = [t for t in sent if t == 0.0]
    assert len(burst) == 24, f"burst was {len(burst)}, expected 30 * 0.8"

    after = sent[len(burst):]
    elapsed = after[-1] - after[0]
    rate = (len(after) - 1) / (elapsed / 60)
    assert rate <= 30, f"sustained {rate:.1f}/min against a stated 30"
    assert rate == pytest.approx(24, rel=0.15), (
        f"sustained {rate:.1f}/min, expected the margin-adjusted 24")


def test_the_tightest_counter_wins(gov, clock):
    """Which counter binds changes with the prompt: requests-per-minute on short
    ungrounded calls, input-tokens-per-minute on grounded ones pulling 30K of
    retrieved text. A limiter watching only one would sail past the other."""
    gov.observe("anthropic", RateLimits(rpm_limit=10_000, rpm_remaining=10_000,
                                        input_tpm_limit=100_000,
                                        input_tpm_remaining=100_000))
    for _ in range(20):
        gov.acquire("anthropic", est_in=20_000, est_out=1_000)
        gov.release("anthropic")

    # 400K of input against an 80K-per-minute allowance is minutes of waiting,
    # while 20 requests against 10,000 RPM is nothing.
    assert clock["now"] > 100, (
        f"only waited {clock['now']:.0f}s — input TPM did not bind")


def test_a_combined_pool_is_not_counted_twice(gov, clock):
    """OpenAI reports ONE token pool covering input and output. Splitting it
    across the separate input and output buckets would count the same capacity
    twice and pace at half the allowed rate."""
    gov.observe("openai", RateLimits(rpm_limit=10_000, rpm_remaining=10_000,
                                     tpm_limit=120_000, tpm_remaining=120_000))
    b = gov._buckets["openai"]
    assert "tpm" in b
    assert "in_tpm" not in b and "out_tpm" not in b


# ═══════════════════════════════════════════════════════════════════════════════
# Headers correct the estimate
# ═══════════════════════════════════════════════════════════════════════════════

def test_a_lower_header_is_trusted_and_a_higher_one_is_not(gov):
    """A header reports what was remaining when that response was GENERATED. By
    the time it arrives other calls may have consumed more, so a higher number
    would hand back capacity already spent. Anthropic also rounds remaining to
    the nearest thousand, which inflates it."""
    gov.observe("anthropic", RateLimits(rpm_limit=100, rpm_remaining=100))
    gov.observe("anthropic", RateLimits(rpm_limit=100, rpm_remaining=5))
    low = gov._buckets["anthropic"]["rpm"]._level
    assert low <= 5

    gov.observe("anthropic", RateLimits(rpm_limit=100, rpm_remaining=100))
    assert gov._buckets["anthropic"]["rpm"]._level <= low + 1, (
        "a higher header handed back spent capacity")


def test_no_headers_means_no_pacing_rather_than_no_traffic(gov):
    """Gemini sends no rate-limit headers at all. A governor that blocked
    whenever it knew nothing would stop that provider entirely."""
    assert gov.acquire("gemini", 1_000, 1_000) == 0.0
    gov.observe("gemini", None)
    assert gov.acquire("gemini", 1_000, 1_000) == 0.0


def test_cold_start_is_conservative(gov):
    """Guessing high and apologising later means a 429 storm on the first
    repetition of a four-day battery."""
    assert gov._ceiling("openai") == COLD_START
    gov.observe("openai", RateLimits(rpm_limit=500, rpm_remaining=500))
    assert gov._ceiling("openai") > COLD_START


def test_one_oversized_call_is_sent_rather_than_deadlocking(gov, clock):
    """A single call larger than the whole margin-adjusted bucket would wait for
    capacity that never arrives. It is sent instead — one overshoot costs a
    retry, a deadlock costs the run."""
    gov.observe("anthropic", RateLimits(rpm_limit=60, rpm_remaining=60,
                                        input_tpm_limit=1_000,
                                        input_tpm_remaining=1_000))
    gov.acquire("anthropic", est_in=500_000, est_out=1_000)   # must return
    assert clock["now"] < 3_600


# ═══════════════════════════════════════════════════════════════════════════════
# In the runner
# ═══════════════════════════════════════════════════════════════════════════════

def test_auto_is_accepted_and_builds_a_governor(tmp_path):
    import contextlib
    import io
    import json
    import pathlib

    import requests

    import machine_psych as mp
    from machine_psych import paths

    paths.set_api_key("anthropic", "k")
    ok = json.loads((pathlib.Path(__file__).parent / "fixtures" / "anthropic" /
                     "ungrounded_ok.json").read_text(encoding="utf-8"))["response"]

    class Resp:
        status_code = 200
        headers = {"anthropic-ratelimit-requests-limit": "50",
                   "anthropic-ratelimit-requests-remaining": "48"}
        def json(self):
            return ok

    real = requests.post
    requests.post = lambda *a, **k: Resp()
    try:
        mp.set_base(tmp_path)
        spec = {"investigation_id": "auto", "studies": [{"study_id": "s",
                "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                           for i in range(4)],
                "providers": {"anthropic/claude-sonnet-5":
                              {"reasoning": "off", "repetitions": 2}}}]}
        with contextlib.redirect_stdout(io.StringIO()):
            run = mp.load_investigation(spec)
            res, _ = mp.run_investigation(run, verbose=False, export=False,
                                          concurrency="auto")
    finally:
        requests.post = real

    assert len(res) == 8
    assert res.status.eq("ok").all()
    assert res.ratelimit_rpm_limit.notna().all(), (
        "auto ran but the headers it paces on were not recorded")


# ═══════════════════════════════════════════════════════════════════════════════
# Pools — found by auditing the assumptions rather than the code
# ═══════════════════════════════════════════════════════════════════════════════

def test_anthropic_families_draw_on_separate_pools():
    """**Anthropic pools by model FAMILY.** Opus models share one combined limit
    and Sonnet models share a separate one.

    Keyed on the provider, a Sonnet-versus-Opus study tracked two independent
    budgets as one and paced both to whichever was busier — precisely the study
    shape this harness exists for.
    """
    assert pool_key("anthropic", "anthropic/claude-sonnet-5") == "anthropic:sonnet"
    assert pool_key("anthropic", "anthropic/claude-opus-5") == "anthropic:opus"
    assert (pool_key("anthropic", "anthropic/claude-sonnet-5")
            != pool_key("anthropic", "anthropic/claude-opus-5"))


def test_unconfirmed_pooling_stays_one_pool_per_provider():
    """Splitting a pool that is actually shared would overshoot it. Same
    reasoning that keeps `supports_idempotency` False where unverified: claiming
    headroom that does not exist is worse than leaving some unused."""
    assert pool_key("openai", "openai/gpt-5.6-sol") == "openai"
    assert pool_key("gemini", "gemini/gemini-3.7-flash") == "gemini"


def test_a_busy_pool_does_not_throttle_its_sibling(gov):
    gov.observe("anthropic:sonnet", RateLimits(rpm_limit=50, rpm_remaining=5))
    gov.observe("anthropic:opus", RateLimits(rpm_limit=50, rpm_remaining=50))
    busy = gov._buckets["anthropic:sonnet"]["rpm"]._level
    idle = gov._buckets["anthropic:opus"]["rpm"]._level
    assert idle > busy * 3, f"opus at {idle:.0f} was dragged down by sonnet at {busy:.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# The estimate is learned, not assumed
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_estimate_does_not_assume_a_search_enabled_call_will_search(gov):
    """**The assumption this project spent a session disproving**, reintroduced
    in the governor's first draft.

    It reserved `tokens_per_query x 2` — 23,200 tokens — whenever `search` was
    allowed. Measured: 5 of 80 search-enabled records actually searched, so 94%
    of calls cost about 2,000. That over-reservation paced a 500K input limit at
    21 calls a minute instead of 250.
    """
    cold = gov.estimate_in("anthropic:sonnet", search_allowed=True)
    assert cold < 23_200 / 3, f"cold estimate {cold} still assumes grounding"

    for _ in range(5):
        gov.record_cost("anthropic:sonnet", 2_100)
    assert gov.estimate_in("anthropic:sonnet", True) == pytest.approx(2_100, rel=0.01)


def test_the_estimate_follows_a_change_in_shape(gov):
    """A battery that moves from ungrounded to grounded probes must adapt rather
    than average over a shape it has left behind."""
    for _ in range(10):
        gov.record_cost("anthropic:sonnet", 2_000)
    low = gov.estimate_in("anthropic:sonnet", True)
    for _ in range(20):
        gov.record_cost("anthropic:sonnet", 40_000)
    assert gov.estimate_in("anthropic:sonnet", True) > low * 5


def test_costs_are_tracked_per_pool_not_globally(gov):
    gov.record_cost("anthropic:sonnet", 2_000)
    gov.record_cost("anthropic:sonnet", 2_000)
    gov.record_cost("anthropic:sonnet", 2_000)
    gov.record_cost("anthropic:opus", 40_000)
    gov.record_cost("anthropic:opus", 40_000)
    gov.record_cost("anthropic:opus", 40_000)
    assert gov.estimate_in("anthropic:sonnet", True) < 5_000
    assert gov.estimate_in("anthropic:opus", True) > 30_000


def test_no_slot_leaks_across_retries(tmp_path):
    """**A leaked slot narrows the run permanently, and worst when it matters.**

    Release happened only on the paths that LEFT the retry loop, so every retry
    acquired without releasing. A battery with transient failures lost a slot per
    retry: 16 of 32 records leaked one, and the narrowing compounds — a run with
    a flaky connection would end up effectively sequential.

    The pairing is now structural, in a `finally`, rather than placed on the
    paths someone remembered.
    """
    import contextlib
    import io
    import json
    import pathlib

    import requests

    import machine_psych as mp
    import machine_psych.runner as RN
    from machine_psych import paths

    paths.set_api_key("anthropic", "k")
    ok = json.loads((pathlib.Path(__file__).parent / "fixtures" / "anthropic" /
                     "ungrounded_ok.json").read_text(encoding="utf-8"))["response"]

    made, original = {}, RN.Governor

    class Spy(original):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made["g"] = self

    n = {"c": 0}

    class Ok:
        status_code = 200
        headers = {"anthropic-ratelimit-requests-limit": "60",
                   "anthropic-ratelimit-requests-remaining": "55"}
        def json(self):
            return ok

    def flaky(*a, **k):
        n["c"] += 1
        if n["c"] % 3 == 0:
            raise requests.exceptions.ConnectionError("blip")
        if n["c"] % 5 == 0:
            class Bad(Ok):
                status_code = 500
                def json(self):
                    return {"error": {"type": "api_error", "message": "boom"}}
            return Bad()
        return Ok()

    real = requests.post
    requests.post = flaky
    RN.Governor = Spy
    try:
        mp.set_base(tmp_path)
        spec = {"investigation_id": "leak", "studies": [{"study_id": "s",
                "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                           for i in range(8)],
                "providers": {"anthropic/claude-sonnet-5": {"reasoning": "off", "repetitions": 2},
                              "anthropic/claude-opus-5": {"reasoning": "off", "repetitions": 2}}}]}
        with contextlib.redirect_stdout(io.StringIO()):
            run = mp.load_investigation(spec)
            res, _ = mp.run_investigation(run, backoff=0, retries=2, verbose=False,
                                          export=False, concurrency="auto")
    finally:
        requests.post = real
        RN.Governor = original

    assert len(res) == 32
    leaked = {k: v for k, v in made["g"]._in_flight.items() if v != 0}
    assert not leaked, f"slots never released: {leaked}"


def test_a_provider_that_reports_nothing_is_not_stuck_at_cold_start(gov):
    """**Found by auditing the briefing, not by any test.**

    Gemini sends no rate-limit headers. The governor only learned a provider's
    limits from headers, so Gemini never left cold start and ran two at a time
    for an entire battery — against a real limit of 1,000 RPM. And because every
    repetition waits for its slowest provider, one arm stuck at two throttled
    the whole study under `concurrency="auto"`.

    "Not heard from yet" and "answered, reports nothing" are different states.
    """
    from machine_psych.ratelimit import parse_headers

    assert gov._ceiling("gemini") == COLD_START          # not heard from
    gov.observe("gemini", parse_headers("gemini", {}))   # answered, silent
    assert gov._ceiling("gemini") > COLD_START, (
        "a provider that reports no limits is stranded at cold start")


def test_the_silent_ceiling_is_tunable():
    """Eight is a guess, because the provider will not say. It must be raisable
    once the console shows the real number."""
    from machine_psych.ratelimit import parse_headers

    g = Governor(silent_ceiling=32)
    g.observe("gemini", parse_headers("gemini", {}))
    assert g._ceiling("gemini") == 32


def test_a_provider_that_does_report_is_unaffected(gov):
    gov.observe("anthropic:sonnet", RateLimits(rpm_limit=5000, rpm_remaining=5000))
    assert "anthropic:sonnet" not in gov._silent
    assert gov._ceiling("anthropic:sonnet") > 8


# ═══════════════════════════════════════════════════════════════════════════════
# Declared limits and daily caps
# ═══════════════════════════════════════════════════════════════════════════════

def test_declared_limits_build_the_same_buckets_headers_would():
    """Gemini sends no rate-limit headers, so without a declaration the governor
    could not learn its limits at all — and fell back to a fixed ceiling 125x
    under the real 1,000 RPM."""
    g = Governor(declared={"gemini": {"rpm": 1000, "input_tpm": 2_000_000}})
    assert sorted(g._buckets["gemini"]) == ["in_tpm", "rpm"]
    assert g._ceiling("gemini") > COLD_START


def test_the_silent_ceiling_defaults_to_the_hard_cap():
    """At paid-tier limits, 64 in flight is ~128 requests a minute against 1,000.
    An overshoot is cheap now — 429s retry on `retry-after`."""
    from machine_psych.governor import MAX_IN_FLIGHT, SILENT_CEILING
    assert SILENT_CEILING == MAX_IN_FLIGHT == 64


def test_a_daily_cap_resets_at_midnight_pacific_not_locally():
    """Gemini's RPD resets at midnight PACIFIC. Reasoning in local time gets the
    reset wrong by up to a day."""
    from datetime import datetime, timedelta, timezone

    from machine_psych.governor import DailyCap, DailyCapReached

    t = {"now": datetime(2026, 9, 23, 22, 0, tzinfo=timezone(timedelta(hours=-7)))}
    cap = DailyCap(10, margin=1.0, now=lambda: t["now"])
    for _ in range(10):
        cap.take()
    with pytest.raises(DailyCapReached):
        cap.take()
    t["now"] += timedelta(hours=3)          # past midnight Pacific
    cap.take()
    assert cap.used == 1


def test_a_daily_cap_is_a_window_not_a_bucket():
    """A continuously refilling model of a per-day quota claims capacity that does
    not exist until midnight. Waiting an hour must not buy anything back."""
    from datetime import datetime, timedelta, timezone

    from machine_psych.governor import DailyCap, DailyCapReached

    t = {"now": datetime(2026, 9, 23, 9, 0, tzinfo=timezone(timedelta(hours=-7)))}
    cap = DailyCap(5, margin=1.0, now=lambda: t["now"])
    for _ in range(5):
        cap.take()
    t["now"] += timedelta(hours=6)          # same day
    with pytest.raises(DailyCapReached):
        cap.take()


# ═══════════════════════════════════════════════════════════════════════════════
# In a real battery
# ═══════════════════════════════════════════════════════════════════════════════

def _gemini_run(tmp_path, inv, limits, body_for=None, resume=None, reps=2):
    import contextlib
    import io
    import json
    import pathlib

    import requests

    import machine_psych as mp
    from machine_psych import paths

    paths.set_api_key("gemini", "k")
    fx = pathlib.Path(__file__).parent / "fixtures" / "gemini"
    ok = json.loads((fx / "ungrounded_ok.json").read_text(encoding="utf-8"))["response"]
    grd = json.loads((fx / "grounded_ok.json").read_text(encoding="utf-8"))["response"]
    n = {"c": 0}

    class Resp:
        def __init__(self, body):
            self.status_code, self.headers, self._b = 200, {}, body
        def json(self):
            return self._b

    def post(*a, **k):
        n["c"] += 1
        return Resp(body_for(n["c"], ok, grd) if body_for else ok)

    real = requests.post
    requests.post = post
    try:
        mp.set_base(tmp_path)
        spec = {"investigation_id": inv, "studies": [{"study_id": "s",
                "probes": [{"probe_id": f"p{i}", "prompt_paths": [[f"q{i}"]]}
                           for i in range(10)],
                "providers": {"gemini/gemini-3.7-flash":
                              {"reasoning": "low", "repetitions": reps}}}]}
        with contextlib.redirect_stdout(io.StringIO()):
            run = mp.load_investigation(spec)
            res, _ = mp.run_investigation(run, verbose=False, export=False,
                                          concurrency="auto", limits=limits,
                                          resume=resume)
    finally:
        requests.post = real
    return res, n["c"]


def test_the_daily_cap_is_a_clean_stop_that_resume_continues(tmp_path):
    """A quota spent is a handoff to `resume=True`, not a fault — the mistake
    first made with OutageGaveUp, where the exception discarded everything in
    memory and printed a traceback for a condition the run handled correctly."""
    first, _ = _gemini_run(tmp_path, "cap", {"gemini": {"rpd": 10}})
    assert 0 < len(first) < 20
    assert first.attrs.get("interrupted")

    second, _ = _gemini_run(tmp_path, "cap", None, resume=True)
    assert len(second) == 20


def test_a_same_day_resume_does_not_grant_a_second_day_of_quota(tmp_path):
    """**Measured before the fix: 18 calls sent against a real cap of 10.**

    A battery stopped by a wifi drop and resumed an hour later built a fresh
    governor whose daily counters read zero. Today's calls are now counted from
    the records already on disk.
    """
    _gemini_run(tmp_path, "same", {"gemini": {"rpd": 10}}, reps=3)
    res, sent = _gemini_run(tmp_path, "same", {"gemini": {"rpd": 10}},
                            resume=True, reps=3)
    assert len(res) <= 10, f"{len(res)} records against a cap of 10"
    assert sent == 0, f"resume sent {sent} more calls on a spent quota"


def test_the_grounding_cap_counts_only_calls_that_grounded(tmp_path):
    """Whether a call grounds is the model's decision — 5 of 80 search-enabled
    calls actually searched on one battery. Counting every search-enabled call
    against a grounding cap would stop a run that had barely used it."""
    res, _ = _gemini_run(
        tmp_path, "grd", {"gemini": {"grounding_rpd": 4}},
        body_for=lambda i, ok, grd: grd if i % 4 == 0 else ok)
    grounded = int(res.grounded.fillna(False).sum())
    assert len(res) > grounded, "ungrounded calls were counted against the cap"
    assert res.attrs.get("interrupted")


def test_queueing_is_reported_even_though_no_limit_fires():
    """Measured on gemini-3.7-flash's free tier: latency rose from 27s to 354s
    with queue depth, and one request passed 500s. No 429 ever said so."""
    from machine_psych.governor import QueueWatch

    q = QueueWatch()
    warnings = [q.record("gemini", x) for x in [30] * 8 + [35] * 4 + [120] * 8]
    fired = [w for w in warnings if w]
    assert len(fired) == 1, "queueing not reported, or reported repeatedly"
    assert "QUEUEING" in fired[0]
