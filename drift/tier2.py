"""Tier 2 — do the parameters still work, and still mean what we recorded.

Two techniques, one tier, because they answer the same question about the same
parameters and neither would ever be run without the other.

**Rejection probes** send a deliberately bogus value and read the error. That is
the technique that recovered every enumeration in `ENUMS` originally, and it
costs one rejected call per parameter. It catches a value becoming illegal.

**Differential probes** send both arms of a claimed capability and assert the
response actually differs on the field that should move. It catches a value
becoming **inert** — and that is the more important half, because rejection is
loud and inertness is silent.

A parameter deprecated by being accepted and ignored produces: 200 on every call,
a record that parses cleanly, every test passing, a condition label reading
`reasoning: off`, and an arm identical to `reasoning: high`. The corpus looks
perfect and the manipulation did nothing.

That is not hypothetical — one provider already accepts `temperature` and
discards it, measured as six distinct answers at temperature 0.0, which is why
`sampling_silently_ignored` exists as a field. A parameter moving into that state
without announcement is what this tier is for.

**And this is the only tier that validates the capability table against
reality.** Every other check asks whether the response shape changed; this asks
whether what we wrote down is still true.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from machine_psych import paths  # noqa: E402
from machine_psych.capabilities import ENUMS, caps_for  # noqa: E402
from machine_psych.providers import get_provider  # noqa: E402

__all__ = ["run", "Finding"]

BOGUS = "__drift_probe_invalid__"

# A prompt short enough to be cheap and structured enough that thinking has
# something to do. Arithmetic rather than prose because the reasoning arms must
# differ in effort, not in verbosity.
PROBE = ("A firm bills three clients. A pays 40% of $50,000. B pays 1.5x what C "
         "pays. What does each pay?")


@dataclass
class Finding:
    kind: str          # "enum" | "inert" | "unexpected" | "error"
    model: str
    subject: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    checked: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Rejection probes — what is still LEGAL
# ═══════════════════════════════════════════════════════════════════════════════

def _dispatch(provider: str, body: dict) -> tuple[int, dict]:
    urls = {"anthropic": "https://api.anthropic.com/v1/messages",
            "openai": "https://api.openai.com/v1/responses",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/interactions"}
    key = paths.api_key(provider)
    headers = {"anthropic": {"x-api-key": key or "", "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
               "openai": {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
               "gemini": {"x-goog-api-key": key or "", "content-type": "application/json"}}[provider]
    r = requests.post(urls[provider], headers=headers, json=body, timeout=600)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"error": {"message": r.text[:300]}}


def _values_in_error(body: dict) -> list[str]:
    """Valid values named in a rejection message.

    Providers list the accepted set when they reject an enum, which is how these
    were recovered in the first place. The extraction is deliberately loose —
    quoted strings and comma lists — because three providers phrase it three
    ways and a strict parser would silently find nothing.
    """
    import re
    message = str((body.get("error") or {}).get("message") or body)
    found = re.findall(r"'([a-z_][a-z0-9_]*)'", message)
    found += re.findall(r'"([a-z_][a-z0-9_]*)"', message)
    # EXCLUDE the value we sent. Providers quote the offending input back —
    # "expected 'low' or 'high', got 'X'" — so an extractor that takes every
    # quoted token returns its own probe value and reports it as newly accepted.
    # Seven of nine findings on the first real run were this.
    return sorted(set(found) - {BOGUS})


def _probe_enums(provider: str, model: str, report: Report) -> None:
    """One rejected call per enumerated parameter."""
    short = model.split("/", 1)[1]
    for parameter in sorted(ENUMS.get(provider, {})):
        recorded = set(ENUMS[provider][parameter])
        body = _body_with(provider, short, parameter, BOGUS)
        if body is None:
            continue
        report.checked += 1
        status, response = _dispatch(provider, body)
        time.sleep(0.5)

        if status == 200:
            # A bogus value ACCEPTED is the loudest possible signal: the
            # parameter is no longer validated, which usually means it is no
            # longer read.
            report.findings.append(Finding(
                "unexpected", model, parameter,
                f"a bogus value was ACCEPTED (HTTP 200) — the parameter may no "
                f"longer be validated, which usually means it is no longer read"))
            continue

        returned = set(_values_in_error(response))
        if not returned:
            report.findings.append(Finding(
                "error", model, parameter,
                f"rejected, but no valid values named in the message: "
                f"{str((response.get('error') or {}).get('message'))[:120]}"))
            continue

        gone = recorded - returned
        new = returned - recorded - {parameter.split(".")[-1]}
        if gone:
            report.findings.append(Finding(
                "enum", model, parameter,
                f"recorded but NOT listed as valid: {sorted(gone)}"))
        if new:
            # "listed as valid" rather than "newly accepted". The message names
            # what the API says it accepts; whether we ever recorded it is a
            # separate question, and asserting acceptance from a rejection
            # message is one inference too many.
            report.findings.append(Finding(
                "enum", model, parameter,
                f"listed as valid but not in ENUMS: {sorted(new)}"))


def _body_with(provider: str, model: str, parameter: str, value):
    """A minimal request carrying one parameter set to `value`."""
    parts = parameter.split(".")
    if parameter in ("error.type", "role"):
        return None      # response-side enums; nothing to send
    if provider == "anthropic":
        body = {"model": model, "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}]}
    elif provider == "openai":
        body = {"model": model, "max_output_tokens": 64, "store": False, "input": "hi"}
    else:
        body = {"model": model, "store": False, "input": "hi",
                "generation_config": {"max_output_tokens": 64}}

    # A bare parameter name on Gemini belongs inside `generation_config`, not at
    # the top level. Sending it top-level gets rejected for being an UNKNOWN
    # FIELD, which the differential probe then read as "temperature is now
    # rejected" — a finding manufactured entirely by putting it in the wrong
    # place. Dotted names already carry their own path and are left alone.
    if provider == "gemini" and len(parts) == 1:
        parts = ["generation_config", parts[0]]

    node = body
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return body


# ═══════════════════════════════════════════════════════════════════════════════
# Differential probes — what is still EFFECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def _probe_differentials(provider: str, model: str, report: Report) -> None:
    """Both arms of each claimed capability, asserting the response moves.

    **Probes NEGATIVES as well as positives.** A capability recorded `False` that
    becomes `True` is otherwise invisible, and that has already happened once —
    OpenAI's `reasoning.summary` began populating after being recorded as empty,
    and it was caught by accident rather than by a check.
    """
    caps = caps_for(model)
    impl = get_provider(provider, api_key=paths.api_key(provider))
    short = model.split("/", 1)[1]

    def call(**intents):
        body = impl.build(PROBE, {"model": model, "max_tokens": 4096, **intents})
        body["model"] = short
        status, response = _dispatch(provider, body)
        time.sleep(0.5)
        if status != 200:
            return None
        return impl.parse(response)

    # ── reasoning_off ────────────────────────────────────────────────────────
    report.checked += 1
    off, high = call(reasoning="off"), call(reasoning="high")
    if off and high:
        off_tok = off.thinking_tok or 0
        high_tok = high.thinking_tok or 0
        actually_off = off_tok == 0 and high_tok > 0
        if caps.reasoning_off and not actually_off:
            report.findings.append(Finding(
                "inert", model, "reasoning_off",
                f"recorded True, but off={off_tok} high={high_tok} thinking "
                f"tokens — the off arm is not off, so a condition labelled "
                f"`reasoning: off` is a fiction"))
        if not caps.reasoning_off and actually_off:
            report.findings.append(Finding(
                "inert", model, "reasoning_off",
                f"recorded False, but off={off_tok} high={high_tok} — the "
                f"capability appears to have TURNED ON"))
    elif off is None and caps.reasoning_off:
        report.findings.append(Finding(
            "inert", model, "reasoning_off",
            "recorded True, but the off arm was rejected"))

    # ── readable_reasoning ───────────────────────────────────────────────────
    report.checked += 1
    result = call(reasoning="high")
    if result:
        has_text = bool(result.thought_text)
        if caps.readable_reasoning and not has_text:
            report.findings.append(Finding(
                "inert", model, "readable_reasoning",
                "recorded True, but no thought text came back"))
        if not caps.readable_reasoning and has_text:
            report.findings.append(Finding(
                "inert", model, "readable_reasoning",
                f"recorded False, but {len(result.thought_text)} characters of "
                f"thought text came back — TURNED ON"))

    # ── verbosity ────────────────────────────────────────────────────────────
    if caps.verbosity:
        report.checked += 1
        low, hi = call(verbosity="low"), call(verbosity="high")
        if low and hi and low.answer_chars and hi.answer_chars:
            ratio = hi.answer_chars / max(low.answer_chars, 1)
            if ratio < 1.3:
                report.findings.append(Finding(
                    "inert", model, "verbosity",
                    f"recorded True, but high/low = {ratio:.2f}x "
                    f"({low.answer_chars} vs {hi.answer_chars} chars) — "
                    f"measured 5.7x when characterised"))

    # ── sampling ─────────────────────────────────────────────────────────────
    report.checked += 1
    body = _body_with(provider, short, "temperature", 0.7)
    status, _ = _dispatch(provider, body)
    time.sleep(0.5)
    accepted = status == 200
    if caps.sampling_silently_ignored and not accepted:
        report.findings.append(Finding(
            "enum", model, "temperature",
            "recorded as silently ignored, but is now REJECTED — which is an "
            "improvement, and means a sampling sweep may be possible"))
    if not caps.sampling_silently_ignored and accepted and not caps.sampling_meaningful:
        report.findings.append(Finding(
            "enum", model, "temperature",
            "recorded as rejected, but is now ACCEPTED — check whether it does "
            "anything before believing it"))


# ═══════════════════════════════════════════════════════════════════════════════

def run(models: list[str] | None = None) -> int:
    """Probe each model. Returns 0 clean, 1 differences, 2 incomplete."""
    from machine_psych.capabilities import known_models
    models = models or [m for m in known_models()
                        if m in ("anthropic/claude-sonnet-5", "openai/gpt-5.6-sol",
                                 "gemini/gemini-3.7-flash")]
    report = Report()

    print("TIER 2 — do the parameters still work, and still mean what we recorded\n")

    for model in models:
        provider = model.split("/", 1)[0]
        if not paths.api_key(provider):
            report.unreachable.append(model)
            print(f"  {model}\n    UNREACHABLE  no API key\n")
            continue
        print(f"  {model}")
        try:
            _probe_enums(provider, model, report)
            _probe_differentials(provider, model, report)
        except Exception as exc:
            report.unreachable.append(model)
            print(f"    INCOMPLETE  {type(exc).__name__}: {exc}")
        mine = [f for f in report.findings if f.model == model]
        for finding in mine:
            print(f"    {finding.kind.upper():<11}{finding.subject}")
            print(f"                {finding.detail}")
        if not mine:
            print("    unchanged")
        print()

    print(f"  {report.checked} probes")
    if report.unreachable:
        print(f"  INCOMPLETE — not probed: {report.unreachable}")
        print("  No differences were found for those, because none were looked for.")
        return 2
    if report.findings:
        print(f"  {len(report.findings)} DIFFERENCES. Each one is the capability")
        print("  table disagreeing with the API — update the table and bump")
        print("  `measured_on`, or fix the code, but do not leave both standing.")
        return 1
    print("  Parameters behave as recorded. Response SHAPE is not checked here;")
    print("  that is tier 3.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(run(args or None))
