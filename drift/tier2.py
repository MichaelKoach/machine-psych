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

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from machine_psych import paths
from machine_psych.capabilities import ENUMS, caps_for
from machine_psych.providers import get_provider

__all__ = ["Finding", "run"]

BOGUS = "__drift_probe_invalid__"

# A value that is REAL SOMEWHERE but may not be valid here. This is the half the
# first version lacked, and it is the half that matters.
#
# Measured 2026-09-05: a bogus value returns the API-WIDE SCHEMA, identical
# across every model of a provider. A plausible value returns the PER-MODEL set,
# and those differ — `minimal` is listed in OpenAI's schema and rejected by
# gpt-5.6-sol, gpt-5.5 and gpt-6-astra alike, each naming a different valid set.
#
# Every enum in the table was recovered by the bogus technique, so every one
# described the schema rather than any model. Three of seven parameters probed
# turned out to carry per-model restrictions it could not see.
PLAUSIBLE = {
    "reasoning.effort": "minimal",          # real on Gemini, not on OpenAI models
    "reasoning.mode": "pro",                # accepted on sol, refused on 5.5
    "reasoning.summary": "concise",
    "text.verbosity": "low",
    "output_config.effort": "low",
    "thinking.type": "enabled",             # valid, and requires budget_tokens
    "generation_config.thinking_level": "minimal",   # rejected on 3.7 and 3.8
    "generation_config.thinking_summaries": "none",
}

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
    """POST through the PROVIDER, not through a second copy of its auth.

    This used to rebuild the URL map and the three header shapes locally — two
    more copies of things `Provider.url_for` and `Provider.headers` already own.
    `base.py` carries the lesson explicitly: copies are what drifted last time,
    in seven of ten shared functions.

    The reason the copy existed is that `dispatch` discards the HTTP status, and
    a rejection probe needs it. That is now `dispatch_with_status`.
    """
    impl = get_provider(provider, api_key=paths.api_key(provider))
    return impl.dispatch_with_status(body, timeout=600)


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

    # Gemini names its valid set UNQUOTED and in a DIFFERENT ORDER on each call —
    # "Allowed values are: medium, low, high" then "high, low, medium". Quoted
    # extraction alone finds nothing there.
    tail = re.search(r"(?:Allowed|Supported) values?(?: are)?:?\s*([^.]*)", message)
    if tail:
        for token in tail.group(1).split(","):
            # Strip the conjunction and the quotes. "…, 'xhigh', and 'max'"
            # yields "and 'xhigh'" as a chunk, and an uncleaned split reports
            # `and 'xhigh` as a valid enum value.
            token = re.sub(r"^\s*(?:and|or)\s+", "", token.strip())
            token = token.strip().strip("'\"").strip()
            if token and re.fullmatch(r"[a-z_][a-z0-9_]*", token):
                found.append(token)

    # EXCLUDE the value we sent, and the FIELD NAME. Providers quote the
    # offending input back — "expected 'low' or 'high', got 'X'" — and one quotes
    # the discriminator too: "Input tag '...' found using 'type' does not match".
    # An extractor taking every quoted token returns its own probe value and the
    # field name, and reports both as valid enum members.
    noise = {BOGUS, "type", "effort", "mode", "summary", "verbosity",
             "thinking_level", "thinking_summaries", "reasoning", "model"}
    return sorted({f for f in found if f and f not in noise})


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
                "a bogus value was ACCEPTED (HTTP 200) — the parameter may no "
                "longer be validated, which usually means it is no longer read"))
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
                f"in ENUMS but NOT in the schema: {sorted(gone)}"))
        if new:
            # "in the schema" rather than "accepted". The bogus message names
            # what the API validates against, which is not what this model takes.
            report.findings.append(Finding(
                "enum", model, parameter,
                f"in the schema but not in ENUMS: {sorted(new)}"))

        # ── the per-model set, which the bogus probe cannot reach ────────────
        plausible = PLAUSIBLE.get(parameter)
        if not plausible:
            continue
        report.checked += 1
        status, response = _dispatch(provider, _body_with(
            provider, short, parameter, plausible))
        time.sleep(0.5)
        if status == 200:
            continue
        message = str((response.get("error") or {}).get("message") or "")

        # A companion-field error means the value IS valid here — Anthropic's
        # `thinking.type: enabled` returns "thinking.enabled.budget_tokens:
        # Field required", which is acceptance with a condition, not rejection.
        if "required" in message.lower():
            continue

        # THE WHOLE PARAMETER MISSING is a different finding from a narrowed
        # value set, and a bigger one. `reasoning.mode` on gpt-5.5 returns
        # "`reasoning.mode` is not supported with this model" — no values are
        # named, so an extractor reads an empty set and reports "accepts
        # nothing", which reads as a narrowing rather than an absence.
        bare = parameter.split(".")[-1]
        if ("not supported with this model" in message
                or f"`{parameter}` is not supported" in message
                or (bare in message and "not supported" in message
                    and not _values_in_error(response))):
            report.findings.append(Finding(
                "unexpected", model, parameter,
                f"THE PARAMETER DOES NOT EXIST on this model — not a narrowed "
                f"enum. A spec setting it here is rejected outright: "
                f"{message[:110]}"))
            continue

        per_model = set(_values_in_error(response)) - {plausible}
        if not per_model:
            continue
        # Sorted, because at least one provider lists these in a DIFFERENT ORDER
        # on each call — "medium, low, high" then "high, low, medium".
        declared = set(ENUMS[provider][parameter])
        missing = declared - per_model - {plausible}
        if missing or plausible in declared:
            report.findings.append(Finding(
                "inert", model, parameter,
                f"PER-MODEL set is {sorted(per_model)}, narrower than the schema "
                f"{sorted(declared)} — a spec using "
                f"{sorted(declared - per_model) or [plausible]} here would be "
                f"rejected at the API"))


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
    print("  Parameters behave as recorded. Response SHAPE is not checked on a")
    print("  schedule — per-record checks run at parse time (integrity.py), and")
    print("  a full shape diff happens at a fixture refresh (see REFRESH.md).")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(run(args or None))
