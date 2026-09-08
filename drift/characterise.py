"""Adding a model to the capability table.

Four probes plus the enum sweep, producing a `ModelCaps` block to paste. Roughly
six calls.

Until now this was **prose in a docstring naming a file that did not exist** —
the unread-constant pattern applied to a procedure, where the table documents how
to add a model by pointing at something imaginary. The error message was fixed to
describe the probes; this is the script it described.

**It refuses to guess.** A probe that fails leaves its field as `None` with a
note, rather than defaulting. A defaulted capability produces an arm whose
condition is a fiction, which is worse than a blank because it looks like data —
and that is the whole reason `caps_for` raises on an unknown model instead of
returning something plausible.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from drift.tier2 import PROBE, _body_with, _dispatch, _values_in_error
from machine_psych import paths

__all__ = ["characterise"]

LONG = ("Compare six qualitative research platforms in detail: pricing, "
        "methodology and target customer for each. Be thorough.")


def characterise(model: str) -> dict:
    """Probe a model and return measured capability values.

    Every value is either measured or `None`. Nothing is inferred from a sibling
    model, because a family is not a guarantee — one Anthropic model cannot
    disable thinking while its siblings can, and that is exactly the kind of
    difference an inherited default would erase.
    """
    provider, short = model.split("/", 1)
    if not paths.api_key(provider):
        raise SystemExit(f"no API key set for {provider}")

    found: dict[str, object] = {}
    notes: list[str] = []

    def send(body):
        status, response = _dispatch(provider, body)
        time.sleep(0.5)
        return status, response

    def base(**extra):
        if provider == "anthropic":
            b = {"model": short, "max_tokens": 2048,
                 "messages": [{"role": "user", "content": PROBE}]}
        elif provider == "openai":
            b = {"model": short, "max_output_tokens": 2048, "store": False,
                 "input": PROBE}
        else:
            b = {"model": short, "store": False, "input": PROBE,
                 "generation_config": {"max_output_tokens": 8192}}
        b.update(extra)
        return b

    # ── 1. can reasoning be turned off ───────────────────────────────────────
    print("  [1/5] reasoning off …", flush=True)
    if provider == "anthropic":
        body = base(thinking={"type": "disabled"})
    elif provider == "openai":
        body = base(reasoning={"effort": "none"})
    else:
        body = base()
        body["generation_config"]["thinking_level"] = "off"
    status, response = send(body)
    if status == 200:
        found["reasoning_off"] = True
    elif status in (400, 422):
        found["reasoning_off"] = False
        notes.append(f"reasoning cannot be disabled: "
                     f"{str((response.get('error') or {}).get('message'))[:100]}")
    else:
        found["reasoning_off"] = None
        notes.append(f"reasoning_off UNMEASURED — HTTP {status}")

    # ── 2. is sampling meaningful, or accepted and ignored ───────────────────
    print("  [2/5] sampling …", flush=True)
    status, _ = send(_body_with(provider, short, "temperature", 0.7))
    if status in (400, 422):
        found["sampling_meaningful"] = False
        found["sampling_silently_ignored"] = False
        notes.append("temperature is rejected outright")
    elif status == 200:
        found["sampling_silently_ignored"] = None
        found["sampling_meaningful"] = None
        notes.append("temperature is ACCEPTED — whether it does anything is "
                     "UNMEASURED and needs repeated calls at two settings; "
                     "one provider accepts it and discards it")
    else:
        found["sampling_meaningful"] = None
        notes.append(f"sampling UNMEASURED — HTTP {status}")

    # ── 3. a grounded call: citations, retrieval set, page age, grounding ────
    print("  [3/5] grounded …", flush=True)
    tools = {"anthropic": [{"type": "web_search_20250305", "name": "web_search"}],
             "openai": [{"type": "web_search"}],
             "gemini": [{"type": "google_search"}]}[provider]
    status, response = send(base(tools=tools))
    if status == 200:
        found.update(_read_grounded(response))
    else:
        notes.append(f"grounded probe UNMEASURED — HTTP {status}")

    # ── 4. is the token budget combined with thinking ────────────────────────
    print("  [4/5] token budget …", flush=True)
    if provider == "anthropic":
        body = {"model": short, "max_tokens": 200,
                "messages": [{"role": "user", "content": LONG}]}
    elif provider == "openai":
        body = {"model": short, "max_output_tokens": 200, "store": False,
                "input": LONG}
    else:
        body = {"model": short, "store": False, "input": LONG,
                "generation_config": {"max_output_tokens": 200}}
    status, response = send(body)
    if status == 200:
        found.update(_read_budget(response, provider))
    else:
        notes.append(f"budget probe UNMEASURED — HTTP {status}")

    # ── 5. the enums ─────────────────────────────────────────────────────────
    print("  [5/5] enums …", flush=True)
    from machine_psych.capabilities import ENUMS
    enums: dict[str, list[str]] = {}
    for parameter in sorted(ENUMS.get(provider, {})):
        probe = _body_with(provider, short, parameter, "__drift_probe_invalid__")
        if probe is None:
            continue
        status, response = send(probe)
        values = _values_in_error(response)
        if values:
            enums[parameter] = values
    found["_enums"] = enums

    return {"model": model, "measured": found, "notes": notes}


def _read_grounded(response: dict) -> dict:
    """Citation mechanism, retrieval set, page age, conditional grounding."""
    out: dict[str, object] = {}
    blob = json.dumps(response)

    if "cited_text" in blob:
        out["citation_offsets"] = "quoted"
    elif "start_index" in blob:
        # Byte or character is NOT decidable from shape — both look identical on
        # ASCII. Recorded as unknown rather than guessed, because guessing wrong
        # produces spans that are silently wrong and drift further into the text.
        out["citation_offsets"] = None
    else:
        out["citation_offsets"] = None

    out["page_age"] = "page_age" in blob
    out["search"] = any(k in blob for k in
                        ("server_tool_use", "web_search_call", "google_search_call"))
    # A retrieval SET means URLs present that no citation references. Cannot be
    # read from one response without comparing the two lists, so it is left for
    # the caller rather than assumed False.
    out["retrieval_set"] = None
    return out


def _read_budget(response: dict, provider: str) -> dict:
    """What a small budget produced. REPORTS, does not conclude.

    An earlier version decided `combined_token_budget` from "thinking tokens
    present and no answer text", and got Gemini wrong — its truncated response
    has both, and it does have a combined budget. That was a verdict function
    computing a conclusion from insufficient evidence, which is the pattern this
    project has now flagged four times.

    So the field is left UNMEASURED and the observation is recorded beside it.
    Deciding it properly needs a sweep across budgets, not one call — and one
    call producing a plausible answer is exactly how a wrong capability gets into
    the table looking like data.
    """
    usage = response.get("usage") or {}
    thinking = (usage.get("output_tokens_details", {}).get("reasoning_tokens")
                or usage.get("total_thought_tokens")
                or usage.get("thinking_tokens") or 0)
    answer = _answer_text(response, provider)
    out: dict[str, object] = {
        "combined_token_budget": None,
        "thinking_in_output_tokens": None,
        "_budget_observation": (
            f"at a 200-token budget: {thinking} thinking tokens, "
            f"{len(answer)} answer characters, "
            f"status={response.get('stop_reason') or response.get('status')}"),
        "_truncation_shape": (
            "PARTIAL text present — this provider truncates mid-answer"
            if answer else "no answer text at all"),
    }
    return out


def _answer_text(response: dict, provider: str) -> str:
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in response.get("content", [])
                       if b.get("type") == "text")
    if provider == "openai":
        return "".join(c.get("text", "") for i in response.get("output", [])
                       if i.get("type") == "message"
                       for c in (i.get("content") or []))
    return "".join(c.get("text", "") for s in response.get("steps", [])
                   if s.get("type") == "model_output"
                   for c in (s.get("content") or []))


def _render(result: dict) -> str:
    """A pasteable ModelCaps block. Unmeasured fields are left as None with a note."""
    from datetime import datetime, timezone
    lines = [f'    "{result["model"]}": ModelCaps(']
    for key, value in sorted(result["measured"].items()):
        if key.startswith("_"):
            continue
        lines.append(f"        {key}={value!r},"
                     + ("   # UNMEASURED — see notes" if value is None else ""))
    lines.append(f'        measured_on="{datetime.now(timezone.utc).date().isoformat()}",')
    if result["notes"]:
        lines.append('        notes="' + " ".join(result["notes"])[:200] + '",')
    lines.append("    ),")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python drift/characterise.py provider/model")
    outcome = characterise(sys.argv[1])
    print()
    print("=" * 72)
    print(_render(outcome))
    print("=" * 72)
    if outcome["measured"].get("_enums"):
        print("\nENUMS returned by rejection:")
        for parameter, values in outcome["measured"]["_enums"].items():
            print(f"    {parameter}: {values}")
    if outcome["notes"]:
        print("\nNotes:")
        for note in outcome["notes"]:
            print(f"  - {note}")
    print("\nFields left as None are UNMEASURED, not False. Fill them by hand or")
    print("leave the model out of the table — a guessed capability produces an")
    print("arm whose condition is a fiction, which is worse than a blank because")
    print("it looks like data.")
