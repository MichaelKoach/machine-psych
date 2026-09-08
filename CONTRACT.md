# The contract

**This is the document to review. Not the code.**

The package is ~4,700 lines. This is the ~40 promises it makes. If these are the
right promises and the verification below holds, the implementation is a detail —
the same trade you already make with any third-party library. You do not read
`pandas`; you trust its contract.

That trade is only sound if the verification is independent of whoever wrote the
code. Here it partly is not, and where it is not is marked.

---

## How much to trust each layer

| layer | independent of the author? | what it caught |
|---|---|---|
| **ruff** (lint, CI) | **yes** — external tool | 114 findings first run, incl. a dead local a hand-written sweep missed |
| **coverage** (CI) | **yes** — measurement, not judgement | 91%; `paths.py` at 60% is the honest weak spot |
| **the 400 tests** | **no** — same author as the code | real bugs, but they share the author's blind spots |
| **the seven audit passes** | **no** | mostly the author's own earlier work |

**The tests are a self-portrait wherever they assert what the code does rather
than what this document demands.** That is the known limit. The deterministic
layer above exists because it does not have that limit.

---

## Blast radius: what to scrutinise

Not everything deserves equal attention. Ordered by what breaks if it is wrong.

### Tier 1 — a wrong answer that looks right

**`caps_for(model)` → `ModelCaps`**
Promises: every capability is MEASURED, never guessed. Raises on an unknown model
rather than defaulting.
*If wrong:* a study arm's condition is a fiction, and the corpus looks fine.
*Verified by:* `_validate_capabilities()` at import; `measured_on` per model;
tier 2 probes the live API against this table.
**Known gap:** three models carry `2026-09-05`; seven inherit older family dates
and are recorded rather than re-verified.

**`Provider.parse(body)` → `ParsedResponse`**
Promises: identical fields from every provider. `None` means the provider CANNOT;
`0` means it did not.
*If wrong:* every downstream number is wrong, silently.
*Verified by:* 46 recorded API responses, several collected specifically so a
WRONG parser fails — the Gemini fixture contains en-dashes because byte and
character offsets agree on ASCII.

**`Provider.status(body)` → one of five states**
Promises: `ok | truncated | incomplete | unavailable | error`, and status agrees
with `parse` about whether an answer exists.
*If wrong:* usable partial answers get discarded, or empty records counted as
successes.

**`resolve(model, passed, on_unmet)`**
Promises: an intent a model cannot honour is refused at LOAD, not at the API.
*If wrong:* a battery fails thirty seconds and one paid call later than it should.
*This was wrong until 2026-09-05* — `supports()` returned True for every reasoning
level without reading `reasoning_levels`.

### Tier 2 — recoverable, but expensive

**`run_investigation(run, ...)`** — sixteen documented ordering rules, all
load-bearing. 238 lines. **The known debt.**
**`load_corpus(investigation, run)`** — side tables re-derived from raw on every
load, so a parser fix repairs old corpora. 2.1 ms/record.
**`normalize(corpus)`** — returns a view AND a report of what it set aside. If a
Tier 3 finding cannot be reached after this, the whole design has failed.

### Tier 3 — read it and you will notice

`index`, `read`, `mentions`, `verdict`, `format_stability`, `export_corpus`.
Display and convenience. A bug here is visible in the output.

---

## The load-bearing invariants

Five claims the whole thing rests on. **These are the ones worth arguing with.**

**1. Absence is not zero.** `None` = the provider cannot; `0` = it did not. Never
fill a Tier 3 column — a fill is a claim.

**2. Normalise at analysis, never at collection.** Raw responses go to disk
unchanged; every narrow view is a function over them. If this fails, this is a
gateway library with extra steps.

**3. Measured, not assumed.** No family defaults. A guessed capability produces an
arm whose condition is a fiction, and that failure is silent because it looks like
data.

**4. Schema is not capability.** A bogus parameter value returns the API-wide
schema; a plausible one returns the per-model set. They differ.

**5. Check declarations on a schedule; check behaviour at the point of use.** A
scheduled check on something that varies needs a model of the variance first, and
that model costs more than the finding.

---

## What is NOT verified

Stated plainly, because an unverified thing presented as verified is the failure
this document exists to prevent.

- **The `unavailable` status branch** is tested synthetically on every provider.
  No genuine transient failure was ever captured in ~400 live calls.
- **Seven of ten models** have inherited capability dates, not fresh measurements.
- **`compare_capabilities()`** is a stub. Largely superseded by tier 2.
- **No research has been done with any of this.** Every finding to date is about
  the APIs, not about models as subjects.

---

## How to check this yourself, without reading the code

```bash
ruff check machine_psych drift tests          # external judgement
pytest tests/ -q --cov=machine_psych          # 400 tests, 91%
python drift/tier1.py                         # free: has anything changed?
python drift/tier2.py                         # ~40 calls: does the table still hold?
```

The last two are the only checks that ask the WORLD rather than the repo. Tier 2
is the one that would tell you the capability table has gone stale, and it is the
closest thing here to an independent audit.
