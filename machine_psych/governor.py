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

# A hard ceiling regardless of what the limits allow. Guards against a provider
# reporting something implausible and the governor opening a thousand sockets.
MAX_IN_FLIGHT = 64


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
                 sleep=time.sleep, max_in_flight: int = MAX_IN_FLIGHT):
        self.margin = margin
        self._clock = clock
        self._sleep = sleep
        self._max = max_in_flight
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[str, Bucket]] = {}
        self._in_flight: dict[str, int] = {}
        self._known: set[str] = set()
        self._observed: dict[str, tuple[float, int]] = {}
        self.waited: dict[str, float] = {}

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
