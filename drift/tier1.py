"""Tier 1 — which models exist.

The cheapest tier and the only one that costs nothing. One GET per provider,
diffed against `CAPABILITIES`.

**It is an alert, not an answer.** A new model appearing tells you `caps_for`
will raise on it, which is deliberate — a guessed capability produces an arm
whose condition is a fiction. It says nothing about whether the platform changed
underneath the models already recorded, and that is the larger risk:
`output_config.effort` replacing `thinking.budget_tokens` was not a new-model
event at all.

So a clean tier 1 means the roster is unchanged. It does not mean nothing moved.
"""

from __future__ import annotations

import sys

import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from machine_psych import paths  # noqa: E402
from machine_psych.capabilities import caps_for, known_models  # noqa: E402

__all__ = ["roster", "run"]

ENDPOINTS = {
    "anthropic": ("https://api.anthropic.com/v1/models",
                  lambda k: {"x-api-key": k, "anthropic-version": "2023-06-01"}),
    "openai":    ("https://api.openai.com/v1/models",
                  lambda k: {"Authorization": f"Bearer {k}"}),
    "gemini":    ("https://generativelanguage.googleapis.com/v1beta/models",
                  lambda k: {"x-goog-api-key": k}),
}


def roster(provider: str) -> tuple[list[str], str | None]:
    """Model ids this key can reach, and an error if the call failed.

    Returns the error rather than raising so one dead provider does not hide the
    other two — and so an INCOMPLETE run is distinguishable from a clean one.
    """
    url, headers = ENDPOINTS[provider]
    key = paths.api_key(provider)
    if not key:
        return [], f"no API key set for {provider}"
    try:
        r = requests.get(url, headers=headers(key), timeout=60)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if r.status_code != 200:
        return [], f"HTTP {r.status_code}: {r.text[:160]}"

    body = r.json()
    items = body.get("data") or body.get("models") or []
    ids = []
    for item in items:
        name = item.get("id") or item.get("name") or ""
        ids.append(name.split("/")[-1] if "/" in name else name)
    return sorted(set(filter(None, ids))), None


def run(providers: list[str] | None = None) -> int:
    """Diff live rosters against CAPABILITIES. Returns an exit code.

    Three states, and the third is why this returns a code rather than a bool:
    a run that could not reach a provider has found no differences for that
    provider, which must not read as a clean result.

        0  clean
        1  differences found
        2  incomplete — a provider could not be reached
    """
    providers = providers or sorted(ENDPOINTS)
    differences = incomplete = False

    print("TIER 1 — which models exist\n")

    for provider in providers:
        live, error = roster(provider)
        if error:
            incomplete = True
            print(f"  {provider}")
            print(f"    UNREACHABLE  {error}")
            print()
            continue

        recorded = {m.split("/", 1)[1] for m in known_models(provider)}
        new = sorted(set(live) - recorded)
        gone = sorted(recorded - set(live))

        print(f"  {provider}   {len(live)} live, {len(recorded)} recorded")
        for model in new:
            differences = True
            print(f"    NEW        {model}")
        for model in gone:
            differences = True
            print(f"    VANISHED   {model}  — recorded but no longer offered")
        if not new and not gone:
            print("    unchanged")
        print()

    # Staleness is reported alongside, because a roster that has not changed says
    # nothing about whether the models in it still behave as recorded, and this
    # is the only place anyone looks weekly.
    print("  measured_on:")
    for model in known_models():
        if providers and model.split("/")[0] not in providers:
            continue
        print(f"    {model:<34}{caps_for(model).measured_on}")

    print()
    if incomplete:
        print("  INCOMPLETE — a provider could not be reached. Differences may")
        print("  exist that this run did not look for.")
        return 2
    if differences:
        print("  DIFFERENCES FOUND. A new model needs characterising before it")
        print("  can be dispatched to; a vanished one breaks any spec naming it.")
        print("  Neither says whether the EXISTING models still behave as")
        print("  recorded — that is tier 2.")
        return 1
    print("  Roster unchanged. This does not mean nothing moved: a parameter")
    print("  going silently inert or a response shape shifting are invisible")
    print("  here. Tiers 2 and 3 look for those.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(run(args or None))
