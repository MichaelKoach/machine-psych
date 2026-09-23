"""Pause the run when the network drops, rather than burning through it.

A four-day battery will meet a wifi drop, a router reboot, a sleeping laptop or
an ISP blip. Before this existed, the retry budget was five attempts with
backoff four — sleeps of 4, 8, 12 and 16 seconds, about forty seconds of
tolerance. **Any outage longer than that failed the conversation, and then the
next one, and the next**, as fast as the network could refuse them. Under
concurrency sixteen workers each burned their own budget against the same
outage.

### Why an outage is not like other failures

A rate limit or a 500 is a per-call condition: the next call might work. **An
outage is a global condition where nothing works**, and retrying is pointless
until it clears. Treating them identically is why the old behaviour failed
badly.

This is a circuit breaker inverted. A normal one fails fast to protect someone
waiting; nobody is waiting on a battery, so this fails SLOW — it holds every
worker until the world comes back, because the alternative is a ruined run.

### Network down, or one provider down

Distinguished by evidence rather than assumed. A connection error means the
request never got out or died in flight; an HTTP error means the network is
fine and the provider is not. When every provider in a run is failing at once,
the network is the common cause. When one is, it is that provider.

A provider outage pauses only that provider's workers, so a three-provider
battery keeps two thirds of its throughput.

### What it costs to be wrong

Holding when the network was fine wastes time and nothing else. Proceeding when
it was not burns the remaining plan. The asymmetry is the whole argument for
waiting rather than failing.
"""

from __future__ import annotations

import threading
import time

__all__ = ["OutageGaveUp", "OutageLatch"]

# Twelve hours. Long enough to cover a router reboot, an overnight ISP outage or
# a machine that slept; short enough that a run does not silently wait for days.
# Past this the run stops cleanly and `resume=True` picks it up, so the ceiling
# is a handoff rather than a loss.
DEFAULT_CEILING = 12 * 3600

# Probe spacing. Tight at first because most outages are brief, widening so a
# long one does not hammer a dead network — and capped, so recovery after eleven
# hours is noticed within a minute rather than an hour.
_PROBE_MIN = 5.0
_PROBE_MAX = 60.0


class OutageGaveUp(RuntimeError):
    """The ceiling passed without recovery. Raised so the run stops cleanly and
    the records already on disk stay resumable."""


class OutageLatch:
    """Holds workers while the network, or one provider, is unreachable.

    One latch per run, shared by every worker. Thread-safe by construction: all
    state changes happen under one lock, and waiting workers block on an
    `Event` rather than polling.
    """

    def __init__(self, providers, probe, ceiling: float = DEFAULT_CEILING,
                 threshold: int = 3, sleep=time.sleep, clock=time.monotonic):
        """`probe(name) -> bool` says whether a provider is reachable RIGHT NOW.

        `threshold` is how many connection errors, across DIFFERENT
        conversations, count as an outage rather than bad luck. Three because a
        single dropped request is ordinary and three in a row is not — and
        because the cost of being slightly late to latch is a few wasted calls,
        while the cost of latching on one blip is a stall.

        `sleep` and `clock` are injected so the tests do not take twelve hours.
        """
        self._providers = set(providers)
        self._probe = probe
        self._ceiling = ceiling
        self._threshold = threshold
        self._sleep = sleep
        self._clock = clock

        self._lock = threading.Lock()
        self._open = {p: threading.Event() for p in self._providers}
        for e in self._open.values():
            e.set()                      # set == traffic allowed
        self._strikes: dict[str, set] = {p: set() for p in self._providers}
        self._recovering: set[str] = set()
        self.history: list[dict] = []

    # ── what a worker calls ────────────────────────────────────────────────

    def wait(self, provider: str) -> None:
        """Block while this provider is latched. Returns immediately otherwise.

        Called before every dispatch. The uncontended path is one `Event.wait()`
        on a set event, which costs nothing measurable.
        """
        ev = self._open.get(provider)
        if ev is not None:
            ev.wait()

    def record_failure(self, provider: str, conversation: str,
                       is_connection_error: bool) -> None:
        """Report a failed call. Only connection errors count toward an outage.

        An HTTP error means the network carried the request and the provider
        answered badly, which is a different problem with a different fix. A
        connection error means the request never got out or died in flight, and
        that is the one thing a latch can help with.

        Strikes are counted per CONVERSATION, not per attempt. One conversation
        retrying five times is one piece of evidence about the network, not
        five — counting attempts would latch on a single unlucky call.
        """
        if not is_connection_error or provider not in self._strikes:
            return
        with self._lock:
            self._strikes[provider].add(conversation)
            if len(self._strikes[provider]) < self._threshold:
                return
            if provider in self._recovering:
                return
            # Latch THIS provider immediately — it has the evidence. Scope is
            # decided below, by probing, because a strike count cannot tell the
            # difference: in a real outage every provider fails at once, but one
            # of them reaches the threshold first. Deciding scope on whoever got
            # there first diagnosed a network outage as three separate provider
            # outages, which is wrong in the log and wasteful in probes.
            self._open[provider].clear()
            self._recovering.add(provider)
            started = self._clock()
            others = sorted(self._providers - {provider})

        # Probe the OTHERS, outside the lock. If none of them answer either, the
        # common cause is the network rather than any one provider.
        network = bool(others) and not any(self._safe_probe(p) for p in others)

        with self._lock:
            to_latch = set(self._providers) if network else {provider}
            for p in to_latch:
                if p not in self._recovering:
                    self._open[p].clear()
                    self._recovering.add(p)
            self.history.append({"scope": "network" if network else "provider",
                                 "providers": sorted(to_latch),
                                 "started": started, "recovered": None})

        # Probing happens OUTSIDE the lock: it makes network calls and can take
        # hours. Holding the lock would block every worker reporting a failure,
        # including the ones this is meant to help.
        self._recover(sorted(to_latch), started)

    def record_success(self, provider: str) -> None:
        """A call got through. Clears that provider's strikes.

        Without this, three failures spread across an entire battery would
        eventually latch a perfectly healthy run.
        """
        if provider in self._strikes:
            with self._lock:
                self._strikes[provider].clear()

    # ── recovery ───────────────────────────────────────────────────────────

    def _recover(self, latched, started) -> None:
        delay = _PROBE_MIN
        while True:
            self._sleep(delay)
            back = [p for p in latched if self._safe_probe(p)]
            if back:
                with self._lock:
                    for p in back:
                        self._strikes[p].clear()
                        self._recovering.discard(p)
                        self._open[p].set()
                    if self.history:
                        self.history[-1]["recovered"] = self._clock()
                    latched = [p for p in latched if p not in back]
                if not latched:
                    return

            if self._clock() - started > self._ceiling:
                # Release the workers before raising, or they wait forever on an
                # event nobody will set again.
                with self._lock:
                    for p in latched:
                        self._recovering.discard(p)
                        self._open[p].set()
                raise OutageGaveUp(
                    f"{', '.join(latched)} unreachable for "
                    f"{self._ceiling / 3600:.0f}h. Records already collected are "
                    f"on disk — rerun with resume=True to continue.")

            delay = min(delay * 1.5, _PROBE_MAX)

    def _safe_probe(self, provider: str) -> bool:
        """A probe that raises is a probe that failed. It must never propagate:
        an exception here would escape into a worker thread and kill a run that
        was only waiting."""
        try:
            return bool(self._probe(provider))
        except Exception:
            return False
