# Refreshing these fixtures

**Re-collect on a schedule, not when something looks wrong.** By the time
something looks wrong, a corpus has usually already been collected against it.

## Why

These are the only artifact in the project that can contradict the documentation.
The capability table is data someone typed; the status logic is prose in a
docstring. A synthetic fixture is built from both and agrees with both by
construction — only a recorded response can disagree.

It has. On 2026-08-28 a swap from reconstructed to recorded fixtures produced
twelve failures: four provider changes, one real code defect, and several tests
that turned out to be checking invented content.

**Four surfaces moved in about a week and none broke loudly:**

| provider | what changed | what it would have done |
|---|---|---|
| Anthropic | `thinking.type.enabled` → `output_config.effort` | every reasoning arm rejected |
| OpenAI | `usage.tool_usage` removed | `n_queries` silently `None` on every record |
| Gemini | truncates WITH partial text after all | a usable answer discarded |
| Gemini | `start_index` now always sent | nothing — the guard held |

## How, today

Run the collection cell (kept with the project notes; ~40 calls across three
providers), download `fixtures.zip`, and replace these directories. Then run the
suite and read the failures: a failure here is usually a finding, not a bug.

Keep the synthetic files — they are marked in their `_why` fields and record
states no provider has produced live:

    anthropic/overloaded.json
    openai/rate_limited.json
    openai/rejects_no_credit.json      (real, but a BILLING error at HTTP 429)
    gemini/unavailable_503.json
    gemini/unavailable_api_error.json

## The procedure

**It is built.** `drift/collect_fixtures.py` collects; `drift/REFRESH.md` is the
procedure, including how to diff the result. This section previously said the
tool was "planned, not built" and named `tests/refresh_fixtures.py`, a path that
never existed — stale in both directions at once, describing a phantom while the
real thing sat in another directory under another name.

```bash
python drift/collect_fixtures.py --out /tmp/fresh   # ~40 calls
```

Then diff the shapes with `drift/printer.py` and read the result before replacing
anything. `drift/REFRESH.md` has the whole sequence and the two things that
always appear in a diff and are not changes.

## Staleness

`measured_on` in the capability table says when a model was last checked. It
cannot say whether checking again would find something different. Only three
models carry the 2026-08-28 date; the rest inherit older family dates, and that
asymmetry is deliberate.
