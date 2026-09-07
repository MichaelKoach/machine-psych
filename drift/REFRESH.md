# Refreshing the fixtures

**Re-collect on a schedule, not when something looks wrong.** By the time
something looks wrong, a corpus has usually already been collected against it.

This is the top of the testing pyramid — expensive, run rarely, and the only
thing that compares reality against what the project believes. The cheap checks
run on every parse (`machine_psych/integrity.py`) and every test run.

## Why it matters

Fixtures are the only artifact here that can contradict the documentation. The
capability table is data someone typed; the status logic is prose in a docstring.
A synthetic fixture is built from both and agrees with both by construction —
**only a recorded response can disagree.**

It has. On 2026-08-28 a swap from reconstructed to recorded fixtures produced
twelve test failures: four provider changes, one real code defect, and several
tests that turned out to be asserting invented content.

| provider | what changed | what it would have done |
|---|---|---|
| Anthropic | `thinking.type.enabled` needed a companion field | a control recorded as withdrawn |
| OpenAI | `usage.tool_usage` removed | `n_queries` silently null on every record |
| Gemini | truncates WITH partial text | a usable answer discarded |
| Gemini | `start_index` now always sent | nothing — the guard held |

## The procedure

**1. Collect into a scratch directory.** Never over the committed fixtures — the
diff is the point, and overwriting destroys the thing you are comparing against.

```bash
python drift/collect_fixtures.py --out /tmp/fresh
```

~40 calls, a few dollars. Needs `set_api_key` for all three providers.

**2. Diff the shapes.** This is what `drift/printer.py` is for.

```python
from drift.printer import compare, shape_of
import json, pathlib

def shapes(root):
    out = {}
    for d in sorted(p for p in pathlib.Path(root).iterdir() if p.is_dir()):
        bodies = [json.loads(f.read_text())["response"]
                  for f in sorted(d.glob("*.json"))]
        out[d.name] = shape_of([b for b in bodies if isinstance(b, dict)])
    return out

old, new = shapes("tests/fixtures"), shapes("/tmp/fresh")
for provider in sorted(old):
    for line in compare(old[provider], new.get(provider, {})):
        print(provider, line)
```

The Printer scrubs what is non-deterministic and compares everything else
exactly — **scrubbing rather than selecting, because a gap in a scrub list
produces noise and a gap in a compare list produces a miss.**

**3. Read the diff before replacing anything.** Two things will always appear
and are not changes:

- The synthetic failure fixtures, which the collector does not recreate. See
  `_NOT_COLLECTED.md`.
- `gemini/incomplete_no_output`, which sits on a boundary at a 5-token budget and
  has returned both a partial answer and none. **That variance is why a scheduled
  shape-diffing tier was cut** — it could not tell a changed model from a
  different output, and reported this fixture as drift when a confirming call
  showed it unchanged.

**4. Replace, keeping the synthetic files**, then run the suite.

```bash
cp -r /tmp/fresh/anthropic/* tests/fixtures/anthropic/   # etc.
python -m pytest tests/ -q
```

**A test failure here is usually a finding, not a bug.** Read each one and decide
whether the code is wrong or the fixture recorded a change. Both happen.

**5. Update `measured_on`** in `capabilities.py` for any model whose behaviour
you re-verified — and ONLY those. An unmeasured value is not the same as a wrong
one, and a blanket update makes the field report when someone last edited the
file rather than when anyone last checked the API.

## What this does not cover

Truncation behaviour, cross-turn behaviour, and anything else only visible at a
boundary are **unmonitored between refreshes**. They surface when a battery
produces something odd. That is a deliberate trade: a scheduled check on
boundary behaviour needs repeated sampling to separate drift from variance, and
the sampling costs more than the finding is worth.
