"""Tier 3 — did the response shape or behaviour change.

The expensive tier and the only one that catches BEHAVIOUR — thinking becoming
default, truncation returning partial text, cross-turn behaviour shifting. Those
cannot be read off an endpoint.

It compares the SHAPE of freshly collected responses against an approved
baseline, using the Printer: scrub what is non-deterministic, compare everything
else exactly. **Scrubbing rather than selecting, because a gap in a scrub list
produces noise and a gap in a compare list produces a miss** — and noise is the
failure you notice.

### The approval gate

The baseline is an *approved* shape, not a snapshot. Approval testing is explicit
about why:

> people just update snapshots without really understanding what's going on.
> Therefore, despite having tests, bugs appear because there's no easy way to
> tell whether the change is legit.

So this **never writes the baseline as a side effect of running**. A run that
found differences exits non-zero and leaves the baseline untouched; moving it
takes a separate `--approve`, after a person has read the diff. A script that
overwrites what it compares against destroys the baseline by running.

### The canary

A detector that breaks and reports nothing looks exactly like a detector
reporting no changes. So every run injects one deliberate difference and requires
it to appear. If the canary is missing, the machinery is broken rather than the
APIs unchanged, and the run reports that instead of a clean result.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from drift.printer import compare, shape_of  # noqa: E402

__all__ = ["build_shape", "check", "approve", "CANARY"]

BASELINE = pathlib.Path(__file__).parent / "baseline.json"
FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# One deliberate difference, injected into every run and required to appear. A
# clean report that ALSO lacks the canary means the comparison is not working —
# which is indistinguishable from "nothing changed" without it.
CANARY = {"__drift_canary__": "if you cannot see this, the detector is broken"}


def build_shape(source: pathlib.Path | None = None) -> dict:
    """The shape of every fixture, grouped by provider.

    Derived from the fixtures on disk rather than stored separately, so the two
    cannot disagree. Tests keep frozen fixtures; drift keeps a shape; one
    approval regenerates the shape from whatever the fixtures currently are.
    """
    source = source or FIXTURES
    shapes: dict[str, dict] = {}
    for provider_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        bodies = []
        for path in sorted(provider_dir.glob("*.json")):
            record = json.loads(path.read_text())
            body = record.get("response")
            if isinstance(body, dict):
                bodies.append(body)
        if bodies:
            shapes[provider_dir.name] = shape_of(bodies)
    return shapes


def check(source: pathlib.Path | None = None, verbose: bool = False) -> int:
    """Compare fresh shapes against the approved baseline.

        0  clean, and the canary fired
        1  differences found
        2  incomplete — no baseline, or the canary did not fire
    """
    print("TIER 3 — did the response shape or behaviour change\n")

    if not BASELINE.exists():
        print(f"  NO BASELINE at {BASELINE}")
        print("  Nothing to compare against. Review the current shapes and run")
        print("  with --approve to establish one.")
        return 2

    baseline = json.loads(BASELINE.read_text())
    current = build_shape(source)

    # ── canary ───────────────────────────────────────────────────────────────
    canary_seen = False
    if baseline:
        provider = sorted(baseline)[0]
        spiked = json.loads(json.dumps(current.get(provider, {})))
        spiked.setdefault("values", {})["__drift_canary__"] = ["present"]
        canary_seen = bool(compare(baseline[provider], spiked))

    differences = False
    for provider in sorted(set(baseline) | set(current)):
        if provider not in current:
            differences = True
            print(f"  {provider}\n    VANISHED   no fixtures collected\n")
            continue
        if provider not in baseline:
            differences = True
            print(f"  {provider}\n    NEW        not in the approved baseline\n")
            continue

        diffs = compare(baseline[provider], current[provider])
        print(f"  {provider}")
        if diffs:
            differences = True
            for line in diffs[: None if verbose else 25]:
                print(line)
            if not verbose and len(diffs) > 25:
                print(f"    … {len(diffs) - 25} more — rerun with --verbose")
        else:
            print("    unchanged")
        print()

    if not canary_seen:
        print("  CANARY DID NOT FIRE. The comparison is not working, so a clean")
        print("  report here means nothing. Fix the detector before trusting it.")
        return 2
    if differences:
        print("  DIFFERENCES FOUND. Read them, decide whether each is expected,")
        print("  then `--approve` to move the baseline. Approving without reading")
        print("  is how a detector stops detecting.")
        return 1
    print("  Shape unchanged, canary fired. Note this compares the shape of the")
    print("  fixtures ON DISK — it is only as current as the last collection.")
    return 0


def approve(source: pathlib.Path | None = None) -> int:
    """Write the current shapes as the approved baseline. Deliberate, never automatic."""
    current = build_shape(source)
    BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True))
    total = sum(len(s.get("values", {})) for s in current.values())
    print(f"  approved {len(current)} providers, {total} value paths")
    print(f"  {BASELINE}")
    print()
    print("  This is now what future runs compare against. If it was approved")
    print("  without reading the diff, the next run compares against the drift.")
    return 0


if __name__ == "__main__":
    if "--approve" in sys.argv:
        raise SystemExit(approve())
    raise SystemExit(check(verbose="--verbose" in sys.argv))
