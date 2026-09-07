# drift

Detecting what the providers did since you last looked.

Not a test suite. These make live API calls, need keys, and cost money — which is
why they sit here rather than in `tests/`. The comparison logic is tested offline
in `tests/test_drift.py`.

## Why

Providers ship constantly. In one week during the build, four surfaces moved:

| provider | what changed | what it would have done |
|---|---|---|
| Anthropic | `thinking.type.enabled` → `output_config.effort` | every reasoning arm rejected |
| OpenAI | `usage.tool_usage` removed | `n_queries` silently `None` on every record |
| Gemini | truncates WITH partial text after all | a usable answer discarded |
| Gemini | `start_index` now always sent | nothing — the guard held |

**None of them broke loudly.** The question is not "did our code break" — bugs we
write, we find by using the thing. It is "what did they change," and those are
the ones we cannot control and will not notice.

## Running it

```bash
python drift/tier1.py --approve          # establish the roster baseline, once
python drift/tier1.py                    # thereafter. free
python drift/tier2.py                    # ~40 calls/provider
python drift/characterise.py openai/gpt-5.7    # ~6 calls, adds a model
python drift/collect_fixtures.py --out /tmp/fresh   # a refresh; see REFRESH.md
```

Both tiers exit `0` clean, `1` differences found, `2` **incomplete**. The third
matters: a run that could not reach a provider has found no differences for it,
and that must not read as a pass.

## The tiers

**Tier 1 — which models exist.** One GET per provider. Free.

    python drift/tier1.py --approve      # establish the roster baseline, once
    python drift/tier1.py                # thereafter

**Two comparisons, and conflating them was the first version's bug.** NEW and
VANISHED are measured against an approved ROSTER — what the provider offered last
time anyone looked. Separately, a model that is IN USE and no longer live is
reported as **BROKEN**, because a spec naming it will fail.

The first version diffed against `CAPABILITIES`, which holds the ten models this
project uses. That reported `gpt-5.2` as NEW — which it is not; it is simply not
one we characterised, and it would have reported as NEW on every run forever. The
first real run printed 186 lines of which about four mattered.

It also filters the catalogue to dispatchable text models. The endpoints return
everything the key can reach — embeddings, TTS, image, video, Whisper, GPT-3.5 —
and none of that is drift in a text-battery harness.

It is an alert, not an answer. A new model tells you `caps_for` will raise on it;
it says nothing about whether the platform moved under the models already
recorded — and that is the larger risk. `output_config.effort` replacing
`budget_tokens` was not a new-model event.

**Tier 2 — do the parameters still work, and still mean what we recorded.**
Three techniques on the same parameters.

*Rejection probes* send a bogus value and read the error, recovering the valid
set. Catches a value becoming illegal.

*Plausible probes* send a value real elsewhere but maybe not here. A bogus value
returns the API-WIDE SCHEMA, identical on every model of a provider; a plausible
one returns the PER-MODEL set, and those differ — `minimal` is in OpenAI's schema
and rejected by gpt-5.6-sol, gpt-5.5 and gpt-6-astra, each naming a different
valid set. Every enum in the table was recovered by the bogus technique, so every
one described the schema.

*Differential probes* send both arms and assert the response actually moves.
Catches a value becoming **inert** — and that is the half that matters, because
rejection is loud and inertness is silent. A parameter deprecated by being
accepted and ignored returns 200 on every call, parses cleanly, passes every
test, and produces an arm whose condition label is a lie.

It probes negatives as well as positives. A capability recorded `False` becoming
`True` is otherwise invisible, and that has already happened once.

**This is the only tier that validates the capability table against reality.**
Everything else asks whether the shape moved; this asks whether what we wrote
down is still true.

## What replaced tier 3

There was a third tier that compared fixture shapes on a schedule. It was built,
run, and **cut**. The reason is worth keeping, or someone rebuilds it.

Tiers 1 and 2 read **declarations** — a roster, an error message — which do not
vary between calls. Tier 3 read **behaviour**, which varies by design. Asking
"did the shape change" from one sample is unanswerable, because the shape is a
distribution rather than a fixed thing.

On its first real run it reported Gemini's truncation shape as drift. A confirming
call showed it unchanged: at a 5-token budget the model sometimes emits one text
token and sometimes none, and one sample cannot tell that from a change. Fixing
it properly needs N samples per fixture and a notion of "always / sometimes /
never" — 3x the cost, to detect a class of change that has happened once.

**A detector nobody trusts is worse than no detector.** So the three things it
was standing in for are done elsewhere, more cheaply and without the sampling
problem:

**`machine_psych/integrity.py`** — structural checks at parse time, on every
record rather than a weekly sample. `served_model` differing from the requested
model, `n_queries` null on a grounded call, a capability the table claims and the
response lacks. Warn-only, never raising: failing would lose the count, and how
often an issue occurs is the finding.

**The test suite** — assertions on specific structure, which cannot be wrong
about variance the way a shape diff can.

**`REFRESH.md`** — the fixture refresh, which found four API changes in one pass.
`printer.py` survives as the diagnostic for exactly that: `compare(shape_of(old),
shape_of(new))` answers "what changed in the shape" when both sides are already
in hand.

## The Printer

Still here, repurposed. It scrubs what is non-deterministic and compares
everything else exactly — **scrubbing rather than selecting, because a gap in a
scrub list produces noise and a gap in a compare list produces a miss.**

Counts are compared as categories. Three searches becoming four is not drift;
three becoming zero is.

## What nothing here catches

**A model updated in place under the same name.** `integrity.py` catches an
alias resolving to a DIFFERENT name, but a build replaced behind an unchanged one
is invisible — every check passes and a longitudinal comparison silently spans
two subjects. It is why `measured_on` exists and why a longitudinal series should
re-baseline rather than assume continuity.

**A new parameter nobody told us about.** APIs describe the parameters you send,
not the ones available. Changelogs are a human task.

**Pricing and rate limits.** Not correctness, but a battery that cost $30 last
month costing $200 is worth knowing. Worth a manual check when tier 1 reports a
new model, since that is when pricing usually moves.

**Search index changes.** The API is identical; the corpus changes completely.
Arguably not this system's job — but it is the change most likely to affect a
client finding, and it would look like a finding rather than an artifact.
