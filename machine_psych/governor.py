"""Run at the provider's actual rate limit, with a margin.

A fixed concurrency number is a guess. This project tried to measure one and
failed: batteries at concurrency 8, 16 and 32 all returned 158 records a minute
because the battery held six conversations per repetition, so sixteen and
thirty-two had idle workers. **That was the design ceiling, not a rate limit** —
the real ceiling was never found, and anything picked by hand is therefore
arbitrary.

Every provider states its limits on every response. This reads them and runs
underneath them.

### Token buckets, because that is what the server uses

Anthropic's limiter is a token bucket: capacity refills continuously rather than
resetting on a clock edge, and a short burst above the sustained rate is fine if
capacity has accumulated. Mirroring the algorithm means predicting the same
answer the server will give, rather than approximating it with a fixed window
that is wrong at every boundary.

One bucket per counter, because **the tightest one wins and which one that is
changes with the prompt**: requests-per-minute binds on short ungrounded calls,
input-tokens-per-minute on grounded ones that pull 30K of retrieved text.

### Predictive, because headers arrive too late to be reactive

A header describes what was remaining when a response came back. With sixteen
requests in flight the limit can be breached before the first response says
anything, so a purely reactive limiter is always one round trip behind.

The estimate gates the send; the header corrects the estimate. Neither alone is
enough.

### Cold start is conservative

Before the first response there are no limits known, so the governor holds at a
low floor and opens up once the provider has stated the truth. Guessing high and
apologising later means a 429 storm on the first repetition of a four-day
battery.
"""

from __future__ import annotations

import threading
import time

__all__ = ["Bucket", "Governor", "pool_key"]


def pool_key(provider: str, model: str) -> str:
    """Which limit pool a model draws from.

    **Anthropic pools by model FAMILY**: Opus models share one combined limit and
    Sonnet models share a separate one. A Sonnet-versus-Opus study therefore has
    two independent budgets, and tracking them as one throttles both to whichever
    is busier — precisely the study shape this harness was built for.

    Split only where confirmed. OpenAI and Gemini stay as one pool per provider
    because their pooling was not verified, and splitting a shared pool would
    overshoot it — the same reasoning that keeps `supports_idempotency` False
    where unconfirmed.
    """
    if provider == "anthropic":
        name = model.split("/", 1)[-1]
        for family in ("opus", "sonnet", "haiku", "fable"):
            if family in name:
                return f"anthropic:{family}"
    return provider

# Fraction of a stated limit to actually use. Not tuned — chosen because the
# cost of being wrong is asymmetric: unused headroom wastes a little time, while
# overshooting produces retries that cost more time than the headroom saved.
DEFAULT_MARGIN = 0.8

# Until a provider has reported its limits, this many calls may be in flight.
# Enough to get a header back quickly, small enough that being wrong is cheap.
COLD_START = 2

# For a provider that has ANSWERED but reports no limits at all. Gemini sends no
# rate-limit headers, so without this it never leaves cold start and runs two
# at a time for an entire battery — against a real limit of 1,000 RPM. And
# because every repetition waits for its slowest provider, one arm stuck at two
# throttles the whole study. Found by an audit of the briefing, not by a test.
#
# 64 — the same as the hard cap, because at the observed paid-tier limits
# (1,000 RPM) sixty-four in flight is ~128 requests a minute and nowhere near the
# ceiling. An overshoot is also cheap now: a 429 retries on `retry-after`, and
# the outage latch does not fire on HTTP errors.
#
# Lower it for a free-tier project. And note the caveat that has nothing to do
# with rate limits: gemini-3.7-flash was measured QUEUEING rather than rejecting
# under concurrency on the free tier, latency climbing from 27s to 354s with
# queue depth. If that holds on a paid tier, concurrency buys waiting rather than
# throughput — and no limit ever fires to say so. `QueueWatch` below reports it.
SILENT_CEILING = 64

# A hard ceiling regardless of what the limits allow. Guards against a provider
# reporting something implausible and the governor opening a thousand sockets.
MAX_IN_FLIGHT = 64


def _pacific_now():
    """Now, in the timezone that Gemini's daily quotas reset in.

    RPD resets at midnight PACIFIC, not local midnight. A battery that reasons in
    its own timezone gets the reset wrong by up to a day.

    `zoneinfo` needs the `tzdata` package on Windows and silently has no zones
    without it. The fallback is a fixed UTC-8, which is wrong by an hour during
    daylight time — acceptable, because it errs toward stopping EARLY: a cap
    hit an hour too soon costs an hour; a cap missed costs a wall of 429s.
    """
    from datetime import datetime, timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:
        return datetime.now(timezone(timedelta(hours=-8)))


class DailyCapReached(RuntimeError):
    """A per-day limit was reached. A STOP, not a fault.

    Raised so the run ends cleanly with every record on disk. Nothing inside a
    run can wait out a daily reset — it can be twenty hours away — so this hands
    off to `resume=True` the way the outage latch does.
    """


class DailyCap:
    """A fixed window that resets at midnight Pacific.

    **Not a token bucket.** Per-minute limits refill continuously and a bucket
    models them; a per-day quota resets on a clock edge, and a continuously
    refilling model of it would claim capacity that does not exist until
    midnight.
    """

    def __init__(self, limit: int, margin: float = 0.95, label: str = "", now=_pacific_now):
        self.limit = max(1, int(limit * margin))
        self.label = label
        self._now = now
        self._day = now().date()
        self.used = 0

    def take(self, n: int = 1) -> None:
        today = self._now().date()
        if today != self._day:
            self._day, self.used = today, 0
        if self.used + n > self.limit:
            raise DailyCapReached(
                f"{self.label} daily limit reached ({self.used} of {self.limit} "
                f"usable). Records already collected are on disk — rerun with "
                f"resume=True after midnight Pacific.")
        self.used += n


class QueueWatch:
    """Reports when a provider queues requests rather than rejecting them.

    A rate limit announces itself with a 429. Queueing does not: latency just
    climbs, and concurrency buys waiting instead of throughput. Measured on
    gemini-3.7-flash's free tier, latency rose from 27s to 354s with queue depth
    and one request exceeded 500s. No limit ever fired.

    Watches, never controls. Pacing on latency is a different design with its own
    failure modes, and a warning is enough to tell someone to lower concurrency.
    """

    def __init__(self, baseline_n: int = 8, ratio: float = 3.0):
        self._n = baseline_n
        self._ratio = ratio
        self._seen: dict[str, list[float]] = {}
        self.warned: set[str] = set()

    def record(self, pool: str, latency: float) -> str | None:
        xs = self._seen.setdefault(pool, [])
        xs.append(latency)
        if pool in self.warned or len(xs) < self._n * 2:
            return None
        base = sorted(xs[:self._n])[self._n // 2]
        recent = sorted(xs[-self._n:])[self._n // 2]
        if base > 0 and recent > base * self._ratio:
            self.warned.add(pool)
            return (f"{pool}: median latency rose from {base:.0f}s to "
                    f"{recent:.0f}s — the provider may be QUEUEING rather than "
                    f"rate limiting, and no 429 will say so. Lower concurrency "
                    f"for it; more in flight is buying waiting, not throughput.")
        return None


class Bucket:
    """One counter's worth of capacity, refilling continuously.

    `capacity` is the stated limit times the margin. `refill` is that capacity
    per second, since every provider states its limits per minute.
    """

    def __init__(self, capacity: float, per_second: float, clock=time.monotonic):
        self.capacity = capacity
        self.refill = per_second
        self._level = capacity
        self._last = clock()
        self._clock = clock

    def _tick(self) -> None:
        now = self._clock()
        self._level = min(self.capacity,
                          self._level + (now - self._last) * self.refill)
        self._last = now

    def wait_for(self, amount: float) -> float:
        """Seconds until `amount` is available. 0 when it already is.

        An amount larger than the whole bucket would wait forever, so it is
        clamped: one call that cannot fit under the margin is still sent rather
        than deadlocking the run.
        """
        self._tick()
        want = min(amount, self.capacity)
        if self._level >= want:
            return 0.0
        return (want - self._level) / self.refill if self.refill > 0 else 0.0

    def take(self, amount: float) -> None:
        self._tick()
        self._level = max(0.0, self._level - amount)

    def observe(self, remaining: float) -> None:
        """Correct the level from a header.

        Trusted only when it is LOWER than the local estimate. A header reports
        what was remaining when that response was generated, so by the time it
        arrives other calls may have consumed more — taking a higher number at
        face value would hand back capacity that is already spent. Anthropic also
        rounds remaining to the nearest thousand, which inflates it slightly.
        """
        self._tick()
        self._level = min(self._level, remaining)


class Governor:
    """Per-provider pacing from the limits each provider reports.

    Thread-safe. One instance per run, shared by every worker.
    """

    def __init__(self, margin: float = DEFAULT_MARGIN, clock=time.monotonic,
                 sleep=time.sleep, max_in_flight: int = MAX_IN_FLIGHT,
                 silent_ceiling: int = SILENT_CEILING,
                 declared: dict | None = None):
        self.margin = margin
        self._clock = clock
        self._sleep = sleep
        self._max = max_in_flight
        self._silent_ceiling = silent_ceiling
        # Pools that have ANSWERED with no limits — distinct from pools not yet
        # heard from. The first get `silent_ceiling`, the second cold start.
        self._silent: set[str] = set()
        self._daily: dict[str, DailyCap] = {}
        self._grounding: dict[str, DailyCap] = {}
        self.queue = QueueWatch()
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[str, Bucket]] = {}
        self._in_flight: dict[str, int] = {}
        self._known: set[str] = set()
        self._observed: dict[str, tuple[float, int]] = {}
        self.waited: dict[str, float] = {}
        # LAST, so every structure `declare` touches already exists. It ran
        # before `_lock` was created in the first draft and raised on
        # construction.
        for pool, lim in (declared or {}).items():
            self.declare(pool, **lim)

    def declare(self, pool: str, rpm=None, input_tpm=None, output_tpm=None,
                tpm=None, rpd=None, grounding_rpd=None) -> None:
        """Limits read off a console, for a provider that does not report them.

        Builds the SAME buckets a header would. Gemini sends no rate-limit
        headers, so without this the governor could not learn its limits at all
        and fell back to a fixed ceiling — 125x under the real 1,000 RPM.

        `rpd` and `grounding_rpd` become daily caps, which headers never carry
        for any provider. **The grounding cap is the one that bites:** Gemini 3
        models allow 1,500 grounded requests a day, and that did not rise at
        Tier 2.

        A declared limit is a claim about someone's account on some day. Headers,
        where a provider sends them, override it on the first response.
        """
        from .ratelimit import RateLimits
        limits = RateLimits(rpm_limit=rpm, rpm_remaining=rpm,
                            input_tpm_limit=input_tpm, input_tpm_remaining=input_tpm,
                            output_tpm_limit=output_tpm,
                            output_tpm_remaining=output_tpm,
                            tpm_limit=tpm, tpm_remaining=tpm)
        if not limits.empty:
            self.observe(pool, limits)
        with self._lock:
            if rpd:
                self._daily[pool] = DailyCap(rpd, label=f"{pool} requests/day")
            if grounding_rpd:
                self._grounding[pool] = DailyCap(
                    grounding_rpd, label=f"{pool} search grounding")

    def count_day(self, pool: str) -> None:
        """One request against the daily cap. Raises before the cap, not after."""
        cap = self._daily.get(pool)
        if cap is not None:
            with self._lock:
                cap.take()

    def count_grounded(self, pool: str) -> None:
        """One GROUNDED request, counted after the fact.

        Only after, because whether a call grounds is the model's decision —
        measured, 5 of 80 search-enabled calls actually searched. So this cannot
        reserve in advance; it stops the run once the day's allowance is spent,
        before the next grounded call is refused.
        """
        cap = self._grounding.get(pool)
        if cap is not None:
            with self._lock:
                cap.take()

    def seed_today(self, run_dir) -> dict[str, int]:
        """Count calls already made TODAY toward the daily caps.

        **Without this a same-day resume grants a second day's quota.** A battery
        that stops on a wifi drop at 2pm and resumes at 3pm starts a fresh
        governor, whose daily counters read zero — measured: 18 calls sent
        against a real cap of 10.

        Each record's file is written when its call completes, so its
        modification time, read in Pacific, says which quota day it spent.
        Calls from an earlier day are ignored: that quota has already reset.
        """
        import json
        import pathlib
        from datetime import datetime

        today = _pacific_now()
        tz = today.tzinfo
        spent: dict[str, int] = {}
        grounded: dict[str, int] = {}
        for f in pathlib.Path(run_dir).glob("[0-9]*.json"):
            try:
                when = datetime.fromtimestamp(f.stat().st_mtime, tz).date()
                if when != today.date():
                    continue
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            pool = pool_key(rec.get("provider", ""), rec.get("model", ""))
            spent[pool] = spent.get(pool, 0) + 1
            if rec.get("grounded"):
                grounded[pool] = grounded.get(pool, 0) + 1
        with self._lock:
            for pool, n in spent.items():
                if pool in self._daily:
                    self._daily[pool].used += n
            for pool, n in grounded.items():
                if pool in self._grounding:
                    self._grounding[pool].used += n
        return spent

    # ── what a worker calls ────────────────────────────────────────────────

    def estimate_in(self, pool: str, search_allowed: bool) -> float:
        """Expected input tokens for the next call on this pool.

        **Not `tokens_per_query x searches`.** That estimate assumed a
        `search: True` call would search, which is the assumption this project
        spent a session disproving: 5 of 80 search-enabled records actually
        searched. Reserving 23K tokens for calls that cost 2K paced a 500K limit
        at 21 calls a minute instead of 250.

        A running mean of what calls on this pool ACTUALLY cost gets there
        without assuming anything. Seeded low, because the common case is
        ungrounded and the margin plus the headers absorb an early under-estimate
        far more cheaply than a 12x over-reservation absorbs a correct one.
        """
        with self._lock:
            seen = self._observed.get(pool)
            if seen and seen[1] >= 3:
                return seen[0] / seen[1]
        return 6_000.0 if search_allowed else 2_000.0

    def record_cost(self, pool: str, actual_in: float | None) -> None:
        """Feed back what a completed call really cost."""
        if not actual_in:
            return
        with self._lock:
            total, n = self._observed.get(pool, (0.0, 0))
            # A window rather than all history, so a battery that switches from
            # ungrounded to grounded probes adapts instead of averaging over a
            # shape it has left behind.
            if n >= 40:
                total, n = total * 0.5, n * 0.5
            self._observed[pool] = (total + actual_in, n + 1)

    def acquire(self, pool: str, est_in: float, est_out: float) -> float:
        """Block until this call fits under the margin. Returns seconds waited.

        Loops rather than sleeping once: other workers consume capacity while
        this one waits, so the amount needed must be rechecked after sleeping.
        """
        waited = 0.0
        while True:
            with self._lock:
                if self._in_flight.get(pool, 0) >= self._ceiling(pool):
                    delay = 0.05
                else:
                    delay = self._delay(pool, est_in, est_out)
                    if delay <= 0:
                        self._spend(pool, est_in, est_out)
                        self._in_flight[pool] = self._in_flight.get(pool, 0) + 1
                        self.waited[pool] = self.waited.get(pool, 0.0) + waited
                        return waited
            self._sleep(min(delay, 5.0))
            waited += min(delay, 5.0)

    def release(self, pool: str) -> None:
        with self._lock:
            self._in_flight[pool] = max(0, self._in_flight.get(pool, 1) - 1)

    def observe(self, pool: str, limits) -> None:
        """Take what a response reported.

        Limits are re-read on every response rather than once, because a tier
        upgrade or a pool-side change appears here and nowhere else.
        """
        if limits is None:
            return
        with self._lock:
            if getattr(limits, "empty", False):
                self._silent.add(pool)
                return
            b = self._buckets.setdefault(pool, {})

            def declare(name, limit, remaining):
                if limit:
                    per_sec = (limit * self.margin) / 60.0
                    if name not in b or b[name].capacity != limit * self.margin:
                        b[name] = Bucket(limit * self.margin, per_sec, self._clock)
                    self._known.add(pool)
                if remaining is not None and name in b:
                    b[name].observe(remaining * self.margin)

            declare("rpm", limits.rpm_limit, limits.rpm_remaining)
            declare("in_tpm", limits.input_tpm_limit, limits.input_tpm_remaining)
            declare("out_tpm", limits.output_tpm_limit, limits.output_tpm_remaining)
            # A pool with one combined pool (OpenAI) gets one bucket, not two
            # halves — splitting it would count the same capacity twice.
            declare("tpm", limits.tpm_limit, limits.tpm_remaining)

            # The header IS the correction. Subtracting actual usage as well
            # would count the same tokens twice — once from the estimate at
            # acquire, once from the reported remaining — and starve the run.

    # ── internals ──────────────────────────────────────────────────────────

    def _ceiling(self, pool: str) -> int:
        """How many may be in flight at once.

        Until the pool has reported anything, the cold-start floor. After
        that, whatever the tightest bucket's capacity allows, capped.
        """
        if pool not in self._known:
            if pool in self._silent:
                return self._silent_ceiling
            return COLD_START
        b = self._buckets.get(pool, {})
        rpm = b.get("rpm")
        if rpm is None:
            return min(self._max, 8)
        # **A sanity ceiling, not a target.** The right concurrency is
        # rate x latency (Little's Law) — at 40 requests a minute and 40s per
        # grounded call that is 27, not 40. Latency is not known here, and the
        # buckets do the real pacing, so this only stops the pool opening an
        # absurd number of sockets if a pool reports something implausible.
        return max(1, min(self._max, int(rpm.capacity)))

    def _delay(self, pool: str, est_in: float, est_out: float) -> float:
        b = self._buckets.get(pool)
        if not b:
            return 0.0
        waits = [b["rpm"].wait_for(1)] if "rpm" in b else []
        if "in_tpm" in b:
            waits.append(b["in_tpm"].wait_for(est_in))
        if "out_tpm" in b:
            waits.append(b["out_tpm"].wait_for(est_out))
        if "tpm" in b:
            waits.append(b["tpm"].wait_for(est_in + est_out))
        return max(waits) if waits else 0.0

    def _spend(self, pool: str, est_in: float, est_out: float) -> None:
        b = self._buckets.get(pool)
        if not b:
            return
        if "rpm" in b:
            b["rpm"].take(1)
        if "in_tpm" in b:
            b["in_tpm"].take(est_in)
        if "out_tpm" in b:
            b["out_tpm"].take(est_out)
        if "tpm" in b:
            b["tpm"].take(est_in + est_out)
