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

import json
import pathlib
import sys

import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from machine_psych import paths  # noqa: E402
from machine_psych.capabilities import caps_for, known_models  # noqa: E402

__all__ = ["roster", "run"]

# Model ids this harness could never dispatch to. The endpoints return the whole
# catalogue — embeddings, speech, image, video, moderation — and diffing that
# against a table of ten text models produced 186 lines of output on the first
# real run, of which about four mattered.
#
# That is the noise failure tier 3 was designed against, walked into by tier 1.
# A detector that reports 180 non-events buries the ones that matter, and gets
# skimmed within a fortnight.
#
# Matching on SUBSTRINGS rather than an allowlist, because the point is to
# exclude what is definitely irrelevant, not to guess what is relevant. A new
# text model with an unfamiliar name must still come through.
NOT_DISPATCHABLE = (
    "embedding", "tts", "whisper", "transcribe", "moderation", "audio",
    "realtime", "image", "sora", "veo", "lyria", "nano-banana", "computer-use",
    "robotics", "-live", "aqa", "translate",
)

# Generations this project does not use. Unlike the above these ARE text models,
# so the filter is a scope decision rather than a fact — recorded separately so
# it can be relaxed without touching the line above.
OLD_GENERATIONS = (
    "gpt-3.5", "gpt-4", "davinci", "babbage", "o1", "o3", "o4-mini",
    "gemini-2.", "gemma", "claude-sonnet-4-5", "claude-haiku-4-5",
    "claude-opus-4-5",
)


def dispatchable(model_id: str) -> bool:
    """Could this harness plausibly send a text battery to it.

    False for anything whose name marks it as a different modality, and for
    generations this project does not use. Neither list is authoritative — a
    model that slips through is reported and can be ignored, which is the right
    direction to fail in.
    """
    lowered = model_id.lower()
    if any(mark in lowered for mark in NOT_DISPATCHABLE):
        return False
    if any(lowered.startswith(gen) or f"-{gen}" in lowered
           for gen in OLD_GENERATIONS):
        return False
    return True


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


ROSTER = pathlib.Path(__file__).parent / "roster.json"


def run(providers: list[str] | None = None, verbose: bool = False) -> int:
    """Diff live rosters against the approved roster. Returns an exit code.

    **Two comparisons, and conflating them was the first version's bug.**

    `CAPABILITIES` holds the ten models this project USES. Diffing a live roster
    against it reported `gpt-5.2` as NEW — which it is not; it is simply not one
    we characterised, and it would have reported as NEW on every run forever. 186
    lines of output on the first real run, of which four mattered.

    So: NEW and VANISHED are measured against an approved ROSTER — what the
    provider offered last time anyone looked. And separately, a recorded model
    that is no longer live is reported as BROKEN, because a spec naming it will
    fail.

        0  clean
        1  differences found
        2  incomplete — a provider could not be reached, or no roster approved
    """
    providers = providers or sorted(ENDPOINTS)
    approved = json.loads(ROSTER.read_text()) if ROSTER.exists() else None
    differences = incomplete = False

    print("TIER 1 — which models exist\n")
    if approved is None:
        print("  NO APPROVED ROSTER — so there is nothing to diff against, and")
        print("  the full dispatchable roster is listed below. Read it, then")
        print("  --approve. Later runs report only what CHANGED.\n")
        incomplete = True

    current: dict[str, list[str]] = {}

    for provider in providers:
        live, error = roster(provider)
        if error:
            incomplete = True
            print(f"  {provider}\n    UNREACHABLE  {error}\n")
            continue

        relevant = sorted(m for m in live if dispatchable(m))
        current[provider] = relevant
        was = set((approved or {}).get(provider, []))
        recorded = {m.split("/", 1)[1] for m in known_models(provider)}

        broken = sorted(recorded - set(live))
        # A model we USE disappearing is one event, not two. Reporting it as both
        # VANISHED and BROKEN doubles the line and buries which one matters — and
        # BROKEN is the one that matters, because a spec naming it will fail.
        new = sorted(set(relevant) - was) if approved else []
        gone = sorted(was - set(relevant) - set(broken)) if approved else []

        print(f"  {provider}   {len(relevant)} dispatchable "
              f"({len(live) - len(relevant)} filtered), {len(recorded)} in use")

        if approved is None:
            # WITHOUT a baseline there is no diff, and an earlier version printed
            # "unchanged" here while the header said "everything reads as new" —
            # two contradictory claims and nothing to read. That defeats the
            # approval gate: a person told to review before approving was shown
            # nothing to review, so approving could only be blind.
            #
            # So the full list is printed. It is the only run where the whole
            # roster is the thing being approved, and it should be seen.
            for model in relevant:
                marker = " (in use)" if model in recorded else ""
                print(f"    {model}{marker}")
            for model in broken:
                differences = True
                print(f"    BROKEN     {model}  — IN USE but not offered")
            print()
            continue

        for model in new:
            differences = True
            print(f"    NEW        {model}")
        for model in gone:
            differences = True
            print(f"    VANISHED   {model}")
        for model in broken:
            differences = True
            print(f"    BROKEN     {model}  — IN USE but no longer offered; any "
                  f"spec naming it will fail")
        if verbose:
            for model in sorted(recorded & set(live)):
                print(f"    in use     {model}")
        if not (new or gone or broken):
            print("    unchanged")
        print()

    if current:
        ROSTER.with_suffix(".current.json").write_text(
            json.dumps(current, indent=1, sort_keys=True))

    print("  measured_on — when each model in use was last CHECKED, which is not")
    print("  the same as whether checking again would find something different:")
    for model in known_models():
        if providers and model.split("/")[0] not in providers:
            continue
        print(f"    {model:<34}{caps_for(model).measured_on}")

    print()
    if incomplete:
        print("  INCOMPLETE — differences may exist that this run did not look for.")
        return 2
    if differences:
        print("  DIFFERENCES FOUND. A NEW model needs characterising before it can")
        print("  be dispatched to. A BROKEN one breaks any spec naming it. Neither")
        print("  says whether the models IN USE still behave as recorded — tier 2.")
        print("  Approve the new roster with --approve once you have read this.")
        return 1
    print("  Roster unchanged. This does not mean nothing moved: a parameter going")
    print("  silently inert or a response shape shifting are invisible here.")
    return 0


def approve(providers: list[str] | None = None) -> int:
    """Record the current roster as the baseline. Deliberate, never automatic."""
    current = {}
    for provider in providers or sorted(ENDPOINTS):
        live, error = roster(provider)
        if error:
            print(f"  {provider}: UNREACHABLE — {error}")
            print("  Refusing to approve a partial roster; the gap would read as")
            print("  a vanished model on the next run.")
            return 2
        current[provider] = sorted(m for m in live if dispatchable(m))
    ROSTER.write_text(json.dumps(current, indent=1, sort_keys=True))
    print(f"  approved {sum(len(v) for v in current.values())} models across "
          f"{len(current)} providers")
    print(f"  {ROSTER}")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--approve" in sys.argv:
        raise SystemExit(approve(args or None))
    raise SystemExit(run(args or None, verbose="--verbose" in sys.argv))
