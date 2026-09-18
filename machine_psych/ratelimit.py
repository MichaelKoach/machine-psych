"""Rate-limit headers, normalised across three providers.

Every provider reports remaining capacity on every response, and this project
threw all of it away. That left one question — *how hard can a battery run* —
answerable only by guessing or by reading a blog post about someone else's
account.

**Three schemas, one shape.** Anthropic sends `anthropic-ratelimit-*` and splits
input from output tokens. OpenAI sends `x-ratelimit-*` and counts input and
output TOGETHER. Gemini sends neither, and reports quota only through its own
console.

### What the headers can and cannot tell you

They describe what was remaining when a response came back. With sixteen requests
in flight, the limit can be breached before the first response says anything —
so a purely reactive limiter is always one round trip behind. Headers correct an
estimate; they cannot replace one.

Three practical cautions, all from providers' own behaviour rather than from
theory:

- **Presence is not guaranteed.** A missing header must degrade to backoff, never
  raise. Gemini currently sends none at all.
- **Anthropic rounds remaining token counts to the nearest thousand**, so treating
  the number as exact overstates headroom on small requests.
- **Reset fields are formatted differently** — seconds, durations like `6m0s`, and
  RFC-3339 timestamps all appear. Parsed to seconds-from-now, or left None.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone

__all__ = ["RATE_LIMIT_COLUMNS", "RateLimits", "parse_headers"]


@dataclass(frozen=True)
class RateLimits:
    """What one response said about remaining capacity.

    Every field is optional. A provider that sends nothing produces an instance
    of all-None rather than an absence, so the columns exist on every record and
    `None` keeps its usual meaning: **the provider did not say**, which is
    different from zero remaining.
    """

    rpm_limit: int | None = None
    rpm_remaining: int | None = None
    input_tpm_limit: int | None = None
    input_tpm_remaining: int | None = None
    output_tpm_limit: int | None = None
    output_tpm_remaining: int | None = None
    # Combined, for providers that do not separate input from output. OpenAI
    # counts one pool; reading its number as either half would overstate both.
    tpm_limit: int | None = None
    tpm_remaining: int | None = None
    # Seconds to wait, from `retry-after` — the provider's own instruction, and
    # better than any backoff schedule we could invent.
    retry_after: float | None = None
    reset_in: float | None = None
    source: str | None = None

    def as_record(self) -> dict:
        """Flat `ratelimit_*` keys for the corpus."""
        return {f"ratelimit_{k}": v for k, v in asdict(self).items()}

    @property
    def empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self)
                   if f.name != "source")


RATE_LIMIT_COLUMNS = [f"ratelimit_{f.name}" for f in fields(RateLimits)]


def _int(headers: dict, *names) -> int | None:
    for n in names:
        v = headers.get(n) or headers.get(n.lower())
        if v is None:
            continue
        try:
            return int(float(str(v).strip()))
        except (TypeError, ValueError):
            continue
    return None


def _seconds(headers: dict, *names) -> float | None:
    """Reset fields in any of the three formats providers actually send."""
    for n in names:
        v = headers.get(n) or headers.get(n.lower())
        if v is None:
            continue
        s = str(v).strip()
        try:
            return float(s)                       # plain seconds
        except ValueError:
            pass
        # Go-style durations: `6m0s`, `1m30s`, `500ms`
        m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:(\d+)ms)?", s)
        if m and any(m.groups()):
            h, mi, sec, ms = (float(g) if g else 0.0 for g in m.groups())
            return h * 3600 + mi * 60 + sec + ms / 1000
        try:                                       # RFC-3339 timestamp
            when = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except ValueError:
            continue
    return None


def parse_headers(provider: str, headers) -> RateLimits:
    """Normalise one response's headers. Never raises.

    A malformed or absent header yields `None` for that field. Failing here
    would turn a reporting nicety into a dispatch error, which inverts the
    point: this exists so a battery can run closer to the limit, not so it can
    fall over further from it.
    """
    if not headers:
        return RateLimits(source=provider)
    try:
        h = {str(k).lower(): v for k, v in dict(headers).items()}
    except Exception:
        return RateLimits(source=provider)

    retry = _seconds(h, "retry-after")

    if provider == "anthropic":
        # Input and output are SEPARATE counters here, and either can trip alone.
        return RateLimits(
            rpm_limit=_int(h, "anthropic-ratelimit-requests-limit"),
            rpm_remaining=_int(h, "anthropic-ratelimit-requests-remaining"),
            input_tpm_limit=_int(h, "anthropic-ratelimit-input-tokens-limit"),
            input_tpm_remaining=_int(h, "anthropic-ratelimit-input-tokens-remaining"),
            output_tpm_limit=_int(h, "anthropic-ratelimit-output-tokens-limit"),
            output_tpm_remaining=_int(h, "anthropic-ratelimit-output-tokens-remaining"),
            retry_after=retry,
            reset_in=_seconds(h, "anthropic-ratelimit-requests-reset",
                              "anthropic-ratelimit-tokens-reset"),
            source=provider)

    if provider == "openai":
        # ONE token pool covering input and output together. Splitting it across
        # the separate fields would double-count the same capacity.
        return RateLimits(
            rpm_limit=_int(h, "x-ratelimit-limit-requests"),
            rpm_remaining=_int(h, "x-ratelimit-remaining-requests"),
            tpm_limit=_int(h, "x-ratelimit-limit-tokens"),
            tpm_remaining=_int(h, "x-ratelimit-remaining-tokens"),
            retry_after=retry,
            reset_in=_seconds(h, "x-ratelimit-reset-requests",
                              "x-ratelimit-reset-tokens"),
            source=provider)

    # Gemini reports quota through its console rather than headers. The generic
    # names are tried anyway: absence is cheap to check and a provider adding
    # them later should not need a code change.
    return RateLimits(
        rpm_limit=_int(h, "x-ratelimit-limit-requests", "ratelimit-limit"),
        rpm_remaining=_int(h, "x-ratelimit-remaining-requests", "ratelimit-remaining"),
        tpm_limit=_int(h, "x-ratelimit-limit-tokens"),
        tpm_remaining=_int(h, "x-ratelimit-remaining-tokens"),
        retry_after=retry,
        reset_in=_seconds(h, "ratelimit-reset", "x-ratelimit-reset"),
        source=provider)
