"""Rate-limit header capture.

Every provider reports remaining capacity on every response and this project
discarded all of it, which left "how hard can a battery run" answerable only by
guessing or by reading a blog post about someone else's account.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile

import machine_psych as mp
import machine_psych.runner as R
from machine_psych.providers import get_provider
from machine_psych.ratelimit import RATE_LIMIT_COLUMNS, RateLimits, parse_headers

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
ANTH = "anthropic/claude-sonnet-5"


def _body():
    return json.loads((FIXTURES / "anthropic" / "ungrounded_ok.json")
                      .read_text())["response"]


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing — three schemas, one shape
# ═══════════════════════════════════════════════════════════════════════════════

def test_anthropic_separates_input_from_output():
    """Anthropic runs input and output token counters INDEPENDENTLY, and either
    can trip alone. Folding them into one number would hide whichever is
    tighter."""
    r = parse_headers("anthropic", {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "43",
        "anthropic-ratelimit-input-tokens-remaining": "486000",
        "anthropic-ratelimit-output-tokens-remaining": "79000"})
    assert (r.rpm_limit, r.rpm_remaining) == (50, 43)
    assert r.input_tpm_remaining == 486_000
    assert r.output_tpm_remaining == 79_000
    assert r.tpm_remaining is None, "Anthropic has no combined pool"


def test_openai_reports_one_combined_pool():
    """OpenAI counts input and output TOGETHER. Reading its number as either half
    would overstate both."""
    r = parse_headers("openai", {
        "x-ratelimit-limit-requests": "500",
        "x-ratelimit-remaining-tokens": "1980000"})
    assert r.rpm_limit == 500
    assert r.tpm_remaining == 1_980_000
    assert r.input_tpm_remaining is None and r.output_tpm_remaining is None


def test_missing_headers_degrade_rather_than_raise():
    """Header presence is NOT guaranteed — Gemini sends none at all, and a
    gateway may strip them. Failing here would turn a reporting nicety into a
    dispatch error, which inverts the point: this exists so a battery can run
    closer to the limit, not fall over further from it."""
    for headers in ({}, None, {"unrelated": "value"}, {"x-ratelimit-limit-requests": "junk"}):
        r = parse_headers("gemini", headers)
        assert isinstance(r, RateLimits)
        assert r.rpm_limit is None


def test_reset_fields_parse_in_every_format_providers_send():
    """Seconds, Go-style durations and RFC-3339 timestamps all appear in the
    wild. `6m0s` is the one a naive float() silently loses."""
    assert parse_headers("openai", {"x-ratelimit-reset-tokens": "6m0s"}).reset_in == 360
    assert parse_headers("openai", {"x-ratelimit-reset-requests": "1.5"}).reset_in == 1.5
    assert parse_headers("anthropic", {"retry-after": "12"}).retry_after == 12


# ═══════════════════════════════════════════════════════════════════════════════
# Capture — the columns reach the corpus
# ═══════════════════════════════════════════════════════════════════════════════

def test_every_record_carries_the_columns_even_with_no_headers():
    """All-None rather than absent. `None` means "the provider did not say",
    which is not zero remaining — and a column that appears only sometimes cannot
    be analysed at all."""
    mp.set_base(tempfile.mkdtemp())
    spec = {"investigation_id": "rl", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {ANTH: {"reasoning": "off", "repetitions": 1}}}]}
    with contextlib.redirect_stdout(io.StringIO()):
        run = mp.load_investigation(spec)
        res, _ = mp.run_investigation(run, dispatch=lambda c: _body(),
                                      verbose=False, export=False)
    for col in RATE_LIMIT_COLUMNS:
        assert col in res.columns, f"{col} missing from the corpus"


def test_headers_reach_the_corpus_when_a_provider_sends_them():
    mp.set_base(tempfile.mkdtemp())
    impl = get_provider("anthropic", api_key="k")
    # **kw so the stub keeps matching when dispatch_full gains a parameter. An
    # earlier version pinned the signature; adding `idempotency_key` made the
    # call raise TypeError, which the runner caught as a dispatch failure and
    # reported as all-None headers rather than as a broken test.
    impl.dispatch_full = lambda cfg, timeout=1800, **kw: (200, _body(), parse_headers(
        "anthropic", {"anthropic-ratelimit-requests-limit": "50",
                      "anthropic-ratelimit-requests-remaining": "43",
                      "anthropic-ratelimit-input-tokens-limit": "500000",
                      "anthropic-ratelimit-input-tokens-remaining": "486000"}))

    spec = {"investigation_id": "hdr", "studies": [{
        "study_id": "s", "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {ANTH: {"reasoning": "off", "repetitions": 2}}}]}
    original = R._run_conversation
    R._run_conversation = lambda r, **kw: original(
        r, **{**kw, "provider": impl, "send": impl.dispatch})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run = mp.load_investigation(spec)
            res, _ = mp.run_investigation(run, verbose=False, export=False)
    finally:
        R._run_conversation = original

    assert res.ratelimit_rpm_remaining.iloc[0] == 43
    assert res.ratelimit_input_tpm_remaining.iloc[0] == 486_000
    assert (res.ratelimit_source == "anthropic").all()


def test_exactly_one_dispatch_variant_makes_the_request():
    """Three entry points, one POST. Copies are what drifted last time, in seven
    of ten shared functions.

    The chain is `dispatch` → `dispatch_with_status` → `dispatch_full`, so an
    earlier version of this test asserting each names `dispatch_full` directly
    was wrong about the structure. The property that matters is that only ONE of
    them holds the request.
    """
    import ast
    import inspect

    from machine_psych.providers.base import Provider

    posts = []
    for name in ("dispatch", "dispatch_with_status", "dispatch_full"):
        fn = ast.parse(inspect.getsource(getattr(Provider, name)).strip()).body[0]
        stmts = [n for n in fn.body
                 if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        code = " ".join(ast.unparse(n) for n in stmts)
        if "requests.post" in code:
            posts.append(name)

    assert posts == ["dispatch_full"], (
        f"expected only dispatch_full to POST, found {posts}")


def test_retry_honours_the_providers_own_retry_after():
    """The runner slept `backoff * attempt` and ignored the header saying exactly
    how long to wait.

    Under concurrency a fixed schedule is worse than useless: every worker wakes
    at the same instant and re-trips the same limit, which is the thundering herd
    the backoff exists to prevent. Jitter is applied regardless.
    """
    source = pathlib.Path(R.__file__).read_text()
    assert "limits.retry_after or backoff * attempt" in source
    assert "random.random()" in source, "no jitter — workers wake together"
