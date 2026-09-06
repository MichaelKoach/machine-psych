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
python drift/tier1.py                    # free. weekly, or on any announcement
python drift/tier2.py                    # ~20 calls/provider
python drift/tier3.py                    # compares fixture shapes
python drift/tier3.py --approve          # move the baseline. deliberate.
python drift/characterise.py openai/gpt-5.7    # ~6 calls, adds a model
```

Every tier exits `0` clean, `1` differences found, `2` **incomplete**. The third
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

**Tier 2 — do the parameters still work, and still mean what we recorded.** Two
techniques on the same parameters.

*Rejection probes* send a bogus value and read the error, recovering the valid
set. Catches a value becoming illegal.

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

**Tier 3 — did the response shape or behaviour change.** The expensive tier and
the only one that catches behaviour.

## How tier 3 avoids becoming noise

Re-collected responses differ constantly, because the subject is stochastic. A
differ that reports all of it gets skimmed within a fortnight, and then a real
change scrolls past unread.

So the Printer **scrubs** what is non-deterministic and compares everything else
exactly. Scrubbing rather than selecting, because **a gap in a scrub list
produces noise and a gap in a compare list produces a miss** — and noise is the
failure you notice.

Counts are compared as categories. Three searches becoming four is not drift;
three becoming zero is.

## The approval gate

The baseline is *approved*, not snapshotted. From the approval-testing
literature:

> people just update snapshots without really understanding what's going on.
> Therefore, despite having tests, bugs appear because there's no easy way to
> tell whether the change is legit.

So `tier3.py` **never writes the baseline as a side effect**. A run with
differences exits non-zero and leaves it untouched. Moving it takes `--approve`,
after a person has read the diff.

## The canary

Every tier-3 run injects one deliberate difference and requires it to appear. A
detector that breaks and reports nothing looks exactly like a detector reporting
no changes; without the canary those are indistinguishable.

## What nothing here catches

**A model updated in place under the same name.** No new model, no parameter
change, no shape change — every tier passes and every longitudinal comparison
silently spans two subjects. It is why `measured_on` exists and why a
longitudinal series should re-baseline rather than assume continuity.

**A new parameter nobody told us about.** APIs describe the parameters you send,
not the ones available. Changelogs are a human task.

**Pricing and rate limits.** Not correctness, but a battery that cost $30 last
month costing $200 is worth knowing. Worth a manual check when tier 1 reports a
new model, since that is when pricing usually moves.

**Search index changes.** The API is identical; the corpus changes completely.
Arguably not this system's job — but it is the change most likely to affect a
client finding, and it would look like a finding rather than an artifact.
