# machine-psych — harness briefing

**What it is.** An instrument for running designed experiments on language models.
You specify prompts and conditions; it dispatches across Anthropic, OpenAI and
Gemini with repetitions and returns one corpus where results are comparable
across providers.

**The model is the research subject. The harness has no opinion about what you
study with it.** What you vary and what you measure is the study design. The
examples throughout this document are drawn from AEO work (Answer Engine
Optimisation — influencing what LLMs say about a company, the successor problem
to SEO) — what a model says
about a company, which sources shape it — because that is what it has mostly been
used for. **They are illustrations, not scope.** A question about reasoning
behaviour, prompt sensitivity, refusal patterns, or anything else measurable from
model output fits the same machinery.

**What this document is.** The vocabulary, the spec format, what can be measured,
what cannot, how to run it, and the traps that have already cost real work. It is
pasted into a conversation already in progress — keep whatever context you have
and add this.

**What it is not.** Complete. Nine capability fields rest on a single
observation (`mp.provisional(model)` lists them) and six such claims have already
turned out wrong when someone finally varied the input. Figures below marked
*(one battery)* are single observations, not established rates.

**So: check rather than assume.** `mp.caps_for(model)` is measured and dated;
this document is a summary of it and can lag. If something here contradicts the
code, the code is right. If a study design seems to need something not described
here, say so rather than inventing it — §7 lists the known gaps, and it is not
exhaustive either.

# PART A — OPERATING PROTOCOL

Part B is the reference. **This part is the order of operations**, because every
mistake made with this harness so far came not from missing information but from
having to work out what to do first.

## A1. Rules that silently invalidate an analysis

None of these raises an error. Each produces a corpus that looks fine and says
something false.

1. **Group on `model`, not `provider`, when comparing models.** Two Anthropic
   models share `provider="anthropic"`, so grouping on it averages the arms
   together.
2. **`search: True` is not `grounded`.** It permits search; the model decides
   whether to use it. Read `grounded` per record.
3. **`None` is not zero.** `None` means the provider *cannot* report it; `0` means
   it did not happen. Never fill a column to make it tidy.
4. **Side tables already carry identity columns.** Group `mp.citations(corpus)`
   on `model` directly. Merging identity back in creates `model_x` / `model_y`.
5. **Extract mechanically before reading closely.** Reading a few records and
   inferring a pattern is how three findings in this project died at n=7.
6. **Do not compare grounded against ungrounded records as if search caused the
   difference.** Within a `search: True` arm the model CHOSE which records to
   ground, so the two groups differ in whatever made it choose. Only the
   `search: False` versus `search: True` arms are assigned; compare those.
7. **Do not switch models partway through a programme.** A study run on
   different models is not comparable with one run before it.

## A2. Three states a model can be in

| state | means | how you get there |
|---|---|---|
| **provider-available** | the API lists it | `tier1.run()` shows it |
| **harness-qualified** | a complete `CAPABILITIES` entry exists | A4 below |
| **study-ready** | qualified, AND the capabilities THIS study uses have been seen to work | A4 |

**Available is not qualified.** `load_investigation` refuses any model without a
capability entry, however many providers list it — correctly, because an
unmeasured model produces arms whose conditions are guesses.

**Study-ready is scoped to the study, not to the model.** It does not require all
22 fields re-measured. Verify what the study depends on — the exact model id,
generation, the reasoning configuration you will use, search off, search
permitted, real grounding, source and citation extraction, a safe token budget,
correct corpus identity — and leave the rest marked `inherited` until a study
needs them. Qualification should be short and hard to misuse, not exhaustive.

## A3. Running a study, in order

1. **Choose models that are already qualified**, or qualify new ones first (A4).
   **Freeze the exact ids** — `gemini-3.8-flash`, not `gemini-flash-latest`. An
   alias resolves to whatever the provider ships next, and Google has shipped a
   Flash model every three weeks.
2. **Choose the reasoning configuration deliberately** (A6).
3. **Write the spec.** A5 lists what will reject it.
4. **Preflight:** `run = mp.load_investigation(spec)` — validates and prints the
   plan and cost. Sends nothing.
5. **Preregister:** `mp.save_investigation(spec)`.
6. **Run:** `mp.run_investigation(run, concurrency="auto", limits=LIMITS)`.
   Interrupted? Run it again with `resume=True`.
7. **Validate before interpreting** — counts per model × probe × rep, statuses,
   `served_model`, grounding rate, unique conversation ids. §4 has the list.
8. **Only then analyse**, starting with A1.

## A4. Qualifying a new model

For any model not already in `CAPABILITIES`:

1. **Confirm it is available:** `tier1.run([...])`.
   Then check it is not already qualified: `'provider/model' in mp.known_models()`.
2. **Measure what can be measured:** `python drift/characterise.py provider/model`.
   The block it prints is **PARTIAL** — 11 of 22 fields — and says so.
3. **Complete the rest.** Take `reasoning_levels` from the ENUMS it prints. For
   every other field, measure it or copy it from a sibling model **and mark it
   `"inherited:<sibling>"` in `evidence`** — `"inherited:claude-opus-5"`, naming
   the model it came from. An inherited value is a measurement of a
   different model — weaker than a single observation of this one, and
   `mp.provisional(model)` reports it.
4. **Register it for this session:**

   ```python
   import dataclasses
   from machine_psych.capabilities import caps_for, register_model
   sibling = caps_for('gemini/gemini-3.8-flash')
   new = dataclasses.replace(
       sibling, measured_on='2026-10-15',
       evidence={**sibling.evidence, 'reasoning_levels': 'single',
                 'tokens_per_query': 'inherited:gemini-3.8-flash'},
       notes='inherited from gemini-3.8-flash except as marked')
   register_model('gemini/gemini-3.9-flash', new)   # a hypothetical next release
   ```

   `register_model` runs the same checks as import and lasts until the kernel
   restarts. **To keep it, paste the `ModelCaps(...)` into `CAPABILITIES` in
   `machine_psych/capabilities.py`** and restart — a running kernel caches the
   old module.
5. **Smoke-test generation**, with `search: False` and then `search: True`.
6. **Smoke-test grounding — twice.** `search: True` on an ordinary prompt proves
   the configuration is *accepted*. It does not prove grounding works: the model
   may simply decline. Then send a prompt that plainly needs current information
   ("What did [company] announce this month?") and inspect a record that
   actually grounded — `mp.queries`, `mp.sources`, `mp.citations` all populated.
   Check `served_model` matches the id you asked for, and `status` is `ok`.
7. **Only then build the paid study.** Fields you did not verify stay marked
   `inherited`; that is expected, and `mp.provisional(model)` lists them.

## A5. What `load_investigation` refuses

A spec can be well-formed and still be refused. Each of these is deliberate.

| refused | because |
|---|---|
| a model absent from `CAPABILITIES` | an unmeasured model's arms are guesses |
| an intent the model cannot honour — e.g. `reasoning: "off"` on a model with no off level, or a level not in its `reasoning_levels` | set `on_unmet: "exclude"` on the study to drop those cells instead of failing |
| **`max_tokens` under 8192 on a combined-budget model** (Gemini) | the budget covers thinking AND output, and thinking has measured ~96% of it — the answer truncates and reads as a model failure |
| `temperature` / `top_p` / `top_k` on a model where sampling is inert | Gemini accepts them and discards them: `temperature: 0.0` gave six different answers in six runs |
| prompts inside a provider block | every provider must be asked the same questions |
| an `include` cell that does not pin every swept factor | its place in the design is undefined |
| an `exclude` that removes every cell, or names an unknown factor | the study would run nothing, or run a grid you did not intend |
| `on_unmet` other than `"error"` or `"exclude"` | a third option — run anyway — was rejected deliberately |

## A6. Choosing a reasoning configuration

| research goal | configuration |
|---|---|
| observe native, user-facing behaviour | **omit `reasoning`** — the model's default applies, usually adaptive |
| experiment on reasoning level | sweep explicit levels, per model, from `reasoning_levels` |
| hold reasoning as constant as possible | set an explicit per-model level — but the same word is **not** the same effort across providers |
| test genuinely no reasoning | only models with `reasoning_off=True`; Gemini's `low` is a gate, not off |

**Omitting `reasoning` is itself a configuration**, not "no reasoning." Effort
becomes a function of the prompt: zero thinking on easy records, thousands of
tokens on hard ones.

## A7. A current-model battery

The three models qualified on 2026-09-24, native reasoning, search crossed as a
factor, run at capacity. Verified to load: 96 records.

```python
spec = {
  "investigation_id": "demand_routing_battery",
  "metadata": {"name": "Demand routing across three providers",
               "rationale": ["native behaviour: reasoning omitted on purpose"]},
  "studies": [{
    "study_id": "routing",
    "rationale": ["does search availability change the recommendation"],
    "probes": [
      {"probe_id": "b2b_pipeline",
       "prompt_paths": [["We are a $300M B2B software company. ..."]]},
      {"probe_id": "japan_launch",
       "prompt_paths": [["We are a US consumer brand launching in Japan. ..."]]},
    ],
    "providers": {
      "anthropic/claude-opus-5-5": {"search": [False, True], "max_tokens": 8192,  "repetitions": 8},
      "openai/gpt-5.6-sol":        {"search": [False, True], "max_tokens": 16384, "repetitions": 8},
      "gemini/gemini-3.8-flash":   {"search": [False, True], "max_tokens": 16384, "repetitions": 8},
    }}]}

LIMITS = {"gemini": {"rpm": 1000, "input_tpm": 2_000_000,
                     "rpd": 10_000, "grounding_rpd": 1_500}}

run = mp.load_investigation(spec)            # preflight: validates, costs, sends nothing
mp.save_investigation(spec)                  # preregister
results, _ = mp.run_investigation(run, concurrency="auto", limits=LIMITS)
```

**Reasoning is omitted on every provider.** Gemini needs `max_tokens` of at least
8192 (A5). `search: [False, True]` makes search a condition — but the `True` arm
still contains records that did not ground, so compare on `grounded`, not on the
condition label.

---

# PART B — REFERENCE

## 0. Vocabulary

These words carry a specific meaning here that differs from their ordinary one.
**Two are used with two meanings in this document**, flagged below.

| term | here it means |
|---|---|
| **investigation** | the whole thing — one spec, one `investigation_id`, one corpus. The unit a person would call "the study" in ordinary speech. |
| **study** | ⚠ **two meanings.** As a spec key, a SUBSECTION of an investigation grouping probes that share providers. In prose ("study design"), the ordinary research sense. Read from context; when writing a spec, `studies` is always the key sense. |
| **probe** | one question being asked, possibly over several turns. A spec key. |
| **prompt_path** | one conversation: a list of turn strings. One string = single turn. |
| **battery** | the set of API calls a spec produces when run. Not a spec key — just what the calls are collectively called. |
| **condition** | the label identifying which arm a record belongs to, e.g. `reasoning=off`. **Provider-scoped**: the same label on two providers is not the same manipulation. |
| **arm** | one branch of a comparison — the records sharing a condition. |
| **cell** | ⚠ **two meanings.** One point in the grid (provider × probe × condition), which is what `on_unmet: "exclude"` drops. Also a Colab notebook cell. Context distinguishes them. |
| **sweep** | a list value in a spec, which expands into several conditions. |
| **repetition / rep** | the same call sent again, unchanged, to sample variance. Not a retry. |
| **turn** | one exchange within a multi-turn probe. `turn=0` is the first. |
| **run** | ⚠ **three uses.** A timestamped execution of an investigation (`list_runs`). The DataFrame `load_investigation` returns, conventionally named `run`. And the ordinary verb. |
| **record** | one API call's result: one row of the corpus plus its raw response on disk. |
| **corpus** | all records from one run, loaded as a DataFrame. |
| **side table** | citations, sources, queries, thoughts — expansions of the corpus joining on `record_id`. |
| **citation** | the model attributing part of its answer to a web page it retrieved. **Not an academic citation.** Carries a URL and, where the provider supplies it, the quoted text. |
| **source** | a URL the model retrieved. `cited=True` marks the ones it actually used; one provider also reports URLs it saw and did not use. |
| **query** | the search string the MODEL wrote, not the prompt you sent. Column name is `search`. |
| **intent** | a provider-neutral parameter name (`reasoning`, `search`) that the harness maps to each provider's own vocabulary. |
| **escape hatch** | a provider-named key in a spec whose contents are sent raw, bypassing the intent layer. |
| **grounded** | search actually ran on this record. Distinct from `search: True`, which only permits it. |
| **truncated** | a status: the answer was cut off by the token limit but **the partial answer is real and usable**. Not a failure. |
| **provider-available / harness-qualified / study-ready** | three states of a model — listed by the API; has a complete capability entry; qualified AND grounding seen to work. See A2. |
| **provisional / swept / structural / inherited** | how a capability was measured. `swept` = varied across cases. `structural` = read from response shape, so one observation suffices. `provisional` = ONE observation, treat as unverified. `inherited:<sibling>` = copied from the named sibling model, not measured on this one — e.g. `inherited:claude-opus-5`. |

---

## 1. The shape of an investigation

A **spec** is a dict. Studies contain probes; probes contain `prompt_paths`; each
path is a list of turns (one string = single turn, several = a conversation).
Providers are declared per study with their intents.

```python
spec = {
  "investigation_id": "liftoff_visibility",
  "metadata": {"name": "What LLMs say about Liftoff Cleaning",
               "rationale": ["why this study exists — free text, travels with the corpus"]},
  "studies": [{
    "study_id": "positioning",
    "probes": [
      {"probe_id": "unprompted",
       "rationale": ["does the brand appear when it is not named"],
       "prompt_paths": [["Who are the best commercial cleaning companies in Denver?"]]},
      {"probe_id": "branded",
       "prompt_paths": [["What is Liftoff Cleaning? Who are their customers?"]]},
      {"probe_id": "ladder",
       "prompt_paths": [["Who are the best commercial cleaners in Denver?",
                         "Which would you pick for a 40-person office, and why?"]]},
    ],
    "providers": {
      "anthropic/claude-sonnet-5": {"reasoning": "off", "search": True,
                                    "max_tokens": 8192, "repetitions": 8},
      "openai/gpt-5.6-sol":        {"reasoning": "low", "search": True,
                                    "max_tokens": 16384, "repetitions": 8},
      "gemini/gemini-3.7-flash":   {"reasoning": "low", "search": True,
                                    "max_tokens": 16384, "repetitions": 8},
    }}]}
```

The example above is an Answer Engine Optimisation study because that is the
common case here. **The same
structure serves any subject** — probes are prompts, providers are arms,
repetitions are samples. A study of refusal behaviour would swap the prompts for
edge-case requests and vary `system`; a study of reasoning would vary `reasoning`
across a difficulty ladder and read `thinking_tok`. Nothing about the spec format
assumes a company is being discussed.

**Intents** (the vocabulary — these map to each provider's own parameters):
`reasoning`, `search`, `max_tokens`, `verbosity`, `max_tool_calls`, `system`,
`temperature`, `top_p`, `top_k`.

**Meta keys** (shape the grid, never sent): `repetitions`, `note`, `include`,
`exclude`.

**A list value becomes a condition sweep.** `"reasoning": ["off", "high"]`
produces two arms. `"on_unmet"` on the study says what to do when a model cannot
honour an intent: `"error"` (default), `"exclude"` (drop the cell), `"run"`.

**A provider-named key is an escape hatch** that merges last and wins:
`"gemini": {"generation_config": {...}}`. It REPLACES the key rather than merging
into it.

---

## 2. Models, and what they actually take

| model | reasoning levels |
|---|---|
| `anthropic/claude-sonnet-5` | off, low, medium, high, xhigh, max |
| `openai/gpt-5.6-sol` | none, low, medium, high, xhigh, max |
| `openai/gpt-5.5` | none, low, medium, high, xhigh — **no `max`** |
| `gemini/gemini-3.7-flash` | low, medium, high — **no `minimal`** |

Also available: `claude-opus-5`, `claude-fable-5`, `gpt-5.6-luna`, `gpt-5.6-terra`,
`gemini-3.6-flash`, `gemini-3.1-pro-preview`.

**Levels are per-model, not per-provider, and the table above is a summary.**
`mp.caps_for(model)` carries the measured values with a `measured_on` date and an
`evidence` marker saying whether each field was swept across cases or seen once.
Three models were re-measured on 2026-09-05; the other seven inherit older family
dates and are **recorded rather than verified**. Check before designing an arm
around a capability.

**Traps that will otherwise produce a broken study:**

- **`condition` is provider-scoped.** A reasoning level on one provider is not the
  same manipulation as the same word on another. **Never group on `condition`
  across providers** — group on `config_resolved`.
- **Search is a PERMISSION, not a condition.** Measured 2026-09-09 across three
  providers and two prompts: each declined to search when the prompt did not need
  it, even with the tool attached. An arm labelled `search: True` therefore
  contains ungrounded records. **Read `grounded` per record** rather than trusting
  the arm label.
- **Gemini's `reasoning: low` is a GATE, not off.** Measured across a difficulty
  ladder: 0 thought tokens on a trivial prompt, 2,534 on a hard one. There is no
  reasoning-off arm on Gemini. Anthropic and OpenAI stayed at zero across the same
  ladder, so their off arms are real.
- **Omitting `reasoning` is itself a configuration, not "no reasoning."** The
  model then uses its default, which on these models is ADAPTIVE — effort becomes
  endogenous to the prompt. Measured on one battery with no reasoning key: zero
  thinking tokens on easy records, thousands on hard ones, 32K total for Sonnet
  against 57K for Opus. Omit deliberately when native behaviour is the subject;
  set it explicitly when reasoning is meant to be held constant.
- **Thinking tokens and readable reasoning are separate capabilities.**
  `thinking_tok` can be large while `thought_text` is null, because the provider
  bills the reasoning without exposing it. Check `readable_reasoning` in
  `caps_for`.
- **Repetition counts come from measured variance, not taste.** Six of six
  distinct answers has been measured on a one-sentence prompt. Any difference
  between arms must clear that floor. **8 reps is a reasonable default; 2 is not
  enough to claim anything.**

---

## 3. Cost

`mp.load_investigation(spec)` prints a per-provider estimate and **sends nothing**.
Always run it before the battery and read the number.

**The estimate's `may search` count is permission, not execution.** Search is a
model decision made per prompt, so a spec cannot predict it. The floor assumes
none of the records search; the ceiling assumes all of them do, heavily. Expect
the low end unless the prompts plainly need current information.

**A worked example, measured 2026-09-17.** Eighty records, two Anthropic models,
`search: True` on every one, eight single-turn business-advice prompts:

| | |
|---|---|
| records | 80 |
| records where search was permitted | 80 |
| **records that actually searched** | **5 (6.25%)** |
| input tokens | 328K |
| output tokens | 217K |
| **realised cost** | **$5.37, or $0.067/record** |
| wall clock | 54 minutes, sequential |

**How often a model grounds depends heavily on the PROVIDER, not only the
prompt.** The battery above used two Claude models, which mostly declined to
search — these prompts are answerable from parametric knowledge. A later battery
of the same kind of prompt, across all three providers:

| provider | grounded |
|---|---|
| `openai/gpt-5.6-sol` | **14 of 20** |
| `anthropic/claude-sonnet-5` | 2 of 20 |
| `gemini/gemini-3.7-flash` | 0 of 20 |

*(one battery each, 2026-09-24.)* So "grounding is rare" held for Claude and was
wrong for OpenAI, which searched on 70% of the same prompts. A prompt asking what
a company announced this year grounds nearly every time, on any provider.

The durable point: **grounded records dominate cost, and whether a record grounds
is not yours to set.** An ungrounded call is a fraction of a cent; a grounded one
ran 20–50K input tokens in an earlier battery, roughly $0.10–0.30. Read the
estimate, then expect the floor.

**Budget time, not just money.** A single call ran ~40 seconds in that battery —
Opus averaged 52s, Sonnet 29s. Sequentially, a 1,000-record battery is most of a
day. With `concurrency="auto"` (§4) calls run in parallel within each
repetition, and on a 12-record calibration it ran 4.3x faster than sequential
with identical output. The gain is larger on grounded calls, where waiting is
nearly all of the time.

---

## 4. Running it

```python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "genson"])
!rm -rf /content/mp
!git clone -q https://github.com/MichaelKoach/machine-psych.git /content/mp
sys.path.insert(0, "/content/mp")

from google.colab import userdata, drive
drive.mount('/content/drive')
import machine_psych as mp
from machine_psych import paths

# Secret names and the output path are per-person. These are the ones in use;
# substitute whatever the Colab secrets panel actually holds.
# Or set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY in the environment
# and skip these three lines entirely — useful for a script left running for
# days, where a pasted key is a key that gets committed.
paths.set_api_key("anthropic", userdata.get('Claude_AI_Research_API_Key'))
paths.set_api_key("openai",    userdata.get('ChatGPT_AI_Research_API_Key'))
paths.set_api_key("gemini",    userdata.get('Gemini-AI-Research-API-Key'))

# Anywhere on Drive. Records are written here as they arrive, so this must
# survive a runtime reset — a path under /content does not.
mp.set_base('/content/drive/MyDrive/<your folder>')
```

**If a key is missing, `userdata.get` returns `None` silently** and the first
dispatch fails with an auth error rather than a missing-key one. Check the
secrets panel before blaming the spec.

Then, in separate cells:

```python
mp.save_investigation(spec)                # optional: persist the spec itself
run = mp.load_investigation(spec)          # validates, prints the plan + cost. SENDS NOTHING.
results, _ = mp.run_investigation(run)     # the battery. writes to disk as it goes.
corpus = mp.load_corpus("liftoff_visibility")
```

**For anything longer than a few minutes, three parameters matter.**

```python
results, _ = mp.run_investigation(
    run,
    concurrency="auto",        # or {"anthropic": 8, "gemini": 2} to fix it
    resume=True,               # continue a run that stopped
    limits={"gemini": {"rpm": 1000, "rpd": 10_000, "grounding_rpd": 1_500}},
)
```

**`concurrency="auto"` is the one to use.** It paces against the limits each
provider reports on every response, keeping a token bucket per counter and
holding a call that would push past 80% of any of them. It starts at two in
flight and opens up once a provider states its limits. Preferred over a number,
because a number is a guess — this project tried to measure a ceiling and could
not: 8, 16 and 32 gave identical throughput because the battery held six
conversations per repetition and the extra workers were idle.

**What `auto` actually does, and where it cannot see.**

- **Paces per model FAMILY, not per provider.** Anthropic pools its limits by
  family: Opus models share one budget, Sonnet another. Tracked as one provider, a
  Sonnet-versus-Opus study would pace both to whichever was busier. OpenAI and
  Gemini stay one pool each, because their pooling is unconfirmed.
- **Learns what a call costs** rather than assuming. It does not reserve for a
  web search just because `search` is allowed — measured, most search-enabled
  calls never search.
- **Gemini reports no limits**, so `auto` cannot learn them — **declare them**
  from the console instead (below). Undeclared, a provider that answers without
  headers runs at `silent_ceiling` (default 64).
- **Hard cap of 64 in flight** per pool regardless of what a provider allows.

**Limits seen on this account, 2026-09-23** *(one account, one day — check your
own console)*: Anthropic reported **5,000 RPM and 5M input tokens/minute**, so it
is not the constraint; the 64 cap binds first. Gemini's console showed **1,000
RPM, 2M input tokens/minute, 10,000 requests/day**, and a **separate 1,500/day cap
on search grounding** that no pacing routes around.

**Declaring limits a provider will not report:**

```python
results, _ = mp.run_investigation(
    run, concurrency="auto",
    limits={"gemini": {"rpm": 1000, "input_tpm": 2_000_000,
                       "rpd": 10_000, "grounding_rpd": 1_500}},
)
```

Builds the same buckets a header would. Keys are **pools**, so an Anthropic
family is `"anthropic:sonnet"` rather than `"anthropic"`. Headers, where a
provider sends them, override a declaration on the first response.

**`rpd` and `grounding_rpd` are daily caps**, which no provider reports in
headers. Reaching one is a **clean stop**, not an error: records stay on disk
and `resume=True` continues after the reset. They reset at **midnight Pacific**,
not local midnight, and a same-day resume counts what today already spent — it
does not grant a second day's quota.

**The grounding cap is the one that bites.** Gemini 3 models allow 1,500
grounded requests a day, and a Tier 2 account reported the same figure — it does
not rise with tier. Only calls that actually grounded count against it.

**Queueing is reported, not handled.** A provider can queue requests rather than
reject them: measured on `gemini-3.7-flash`'s free tier, latency rose from 27s to
354s with queue depth and no 429 ever fired. If median latency on a pool climbs
past 3x its early baseline, the run prints `QUEUEING` once. Lower concurrency for
that provider — more in flight is buying waiting, not throughput.

**A fixed `concurrency` is per provider**, and a dict sets each one by name;
anything unnamed falls back to 1. Rate limits differ by an order of magnitude — Gemini's
free tier runs about 10 requests per minute against Anthropic's hundreds — so one
number for all three is either too slow for Anthropic or a 429 storm on Gemini.
An int applies the same cap to every provider. Default 1, which is sequential.

Conversations run in parallel WITHIN a repetition and the run waits at each
repetition boundary. **That is not a tuning choice.** Retrieval drift must spread
across conditions rather than confound with them, which requires that no arm
FINISHES before another STARTS.

Measured: ~40s per record sequential, so 8,000 records is about four days. Use
`auto` rather than picking a number. **If you fix one instead**, start at 4 per
provider and raise it while watching `attempts > 1` — a retry means the API
pushed back and the harness absorbed it, which is the ceiling announcing itself
one level early.

**`resume=True` continues a run that stopped** — it is NOT "resume if possible."
It requires an existing run and raises `FileNotFoundError` when there is none:

```python
mp.run_investigation(run, concurrency="auto")               # first attempt
mp.run_investigation(run, concurrency="auto", resume=True)  # only after that
```

Strict on purpose: a typo in the investigation id would otherwise start a fresh
battery silently and pay for everything again. It skips records already on disk
and writes into the same directory. `True` takes the most recent run of that
investigation; a timestamp string names one. A four-day battery will be
interrupted, and without this the next attempt re-sends everything.

**Interruptions are handled in three layers**, none of which needs configuring:
retries absorb blips of seconds; an outage latch holds every worker while the
network or one provider is unreachable, distinguishing the two by probing; and
`resume` covers anything longer, including power loss.

The latch waits up to 12 hours, set by `outage_ceiling=` in seconds. Past that
the run stops cleanly with everything on disk, and `resume=True` continues it.

**Retries honour the provider's own `retry-after`**, and on OpenAI a retry after a
lost response carries the same `Idempotency-Key`, so it returns the original
answer rather than billing twice. Anthropic and Gemini support for that is
unconfirmed, so it is not sent there.

**The package owns its own directory structure — do not invent one.** Under the
base it creates and uses exactly two folders:

```
<base>/Investigations/      specs, one JSON per investigation_id
<base>/Output Log/          one folder per run, one JSON per record
```

`mp.where()` returns those three paths. **Writing spec or record files anywhere
else breaks `load_corpus` and `list_investigations`**, which look only in these
locations.

**`save_investigation(spec)` is the only thing needed to persist a spec.** It
validates first, writes to `Investigations/` under the `investigation_id`, and
refuses to overwrite unless told to. There is no need to write the JSON by hand,
and no need for a second archive copy — the corpus already records the resolved
config on every record, so a run is reconstructable from its own output even if
the spec file is lost.

Saving is optional — `load_investigation` accepts the dict directly. But **saving
the exact spec before a paid battery is worth doing anyway**: it is the
preregistration, and it costs nothing.

**Validate after the run, before interpreting.** A check that fails here is cheap;
one that fails during analysis has already wasted the reading. Worth confirming:
total records against what was planned; records per **model** × probe × rep;
statuses; `served_model` against the model requested; grounding rate; and that
`conversation` is unique per experimental unit.

**Notes that matter.** Clone rather than `pip install` — `drift/` is not packaged.
**Restart the runtime after cloning**, or Python serves a cached module. Records
write to Drive as they arrive, so an interrupted run keeps what it collected.

---

## 5. The corpus

One row per record. Side tables join on `record_id`:

```python
mp.index(corpus)                  # compact overview
mp.citations(corpus)              # one row per citation: url, domain, quote, offset_unit
mp.sources(corpus)                # every URL seen; `cited` marks which were used
mp.queries(corpus)                # the search strings the model wrote — column is `search`
mp.thoughts(corpus)               # readable reasoning, where the provider exposes it
mp.read(corpus, record_id=0)      # print a record in full
mp.normalize(corpus)              # → (comparable view, report of what was set aside)
```

**Rate-limit headers are on every record** as `ratelimit_*` — remaining requests
and tokens as of that response. Tier 3, because Gemini sends none and the two
that do use incompatible shapes. Useful for finding real headroom before a large
battery; not comparable across providers.

**Useful columns:** `answer_text`, `answer_chars`, `status`, `provider`, `model`,
`probe`, `condition`, `rep`, `turn`, `conversation`, `n_queries`, `n_citations`,
`grounded`, `thinking_tok`, `latency`, `prompt_hash`.

**Side tables already carry the identity columns** — `record_id`, `provider`,
`model`, `probe`, `rep`, `condition` and the rest. Group on them directly:

```python
mp.citations(corpus).groupby("model").size()          # yes
labels = corpus[["record_id", "model"]]               # no — merging creates
cits.merge(labels, on="record_id").groupby("model")   #      model_x / model_y
```

**A record's identity is provider / MODEL / study / probe / path / rep / turn.**
`conversation` groups the turns of one exchange and includes the model, so two
models from one provider do not collide.

**`prompt_hash` joins the same prompt across providers** with no shared
identifier. That is how cross-provider comparison works.

**`provider` and `model` are different columns, and grouping on the wrong one
silently collapses a study.** `provider` is `anthropic`; `model` is
`anthropic/claude-sonnet-5`. A study comparing two models from the same family —
Sonnet against Opus, or two OpenAI models — has ONE value in `provider` for every
record, so `groupby("provider")` returns a single group and averages the arms
together.

**Group on `model` whenever the comparison is between models. Group on `provider`
only when the comparison is between providers.** The same applies to any
assertion about record counts: a `provider × probe` cell holds every model's
records, so it will contain more than you expect.

**Three rules that decide whether an analysis is right:**

1. **`None` means the provider CANNOT. `0` means it did not.** Never fill a
   column to make it tidy — a fill is a claim.
2. **`answer_chars`, not `out_tok_reported`.** Two providers count thinking inside
   the token figure and one does not, so token counts are not comparable.
3. **Compare citation QUOTES, not offsets.** Three mechanisms exist — quoted text,
   character offsets, byte offsets. `quote` is populated on all of them.

**Status has five values:** `ok`, `truncated`, `incomplete`, `unavailable`,
`error`. **`truncated` carries a real partial answer** — treat it as usable, not
as a failure.

---

## 6. Analysis

**The approach depends on the question, and the question depends on the project.**
What follows is what stays constant; do not assume a method before knowing what is
being asked.

**Corpora outgrow a context window fast.** An 80-record corpus exported to ~177K
tokens — about 2,200 per record. At 200 records that is ~440K, past most context
windows. **Mechanical extraction is an operational constraint, not a stylistic
preference**: aggregate across everything first, then send targeted subsets for
reading.

**Extract before you read.** Pull patterns mechanically across every record first,
then read to understand what the numbers mean. Reading a few records and inferring
the pattern is how this project produced three findings that died at n=7 — and in
each case the records read were simply the ones printed first, which is a sampling
decision disguised as a display choice.

**One record is a story; five is a finding.** Run-to-run variance is high. A
difference between arms must exceed it before it means anything.

**Modes available, matched to the question:**

- **Close reading** — `mp.read()`, or upload a small corpus. For nuance, small n.
- **Mechanical extraction** — `mp.mentions(corpus, ["BrandA", "BrandB"])` counts
  vocabulary across all records; `mp.verdict(corpus, pattern)` finds where a
  pattern appears and how far into the answer.
- **Statistical** — the corpus is a DataFrame. Group, count, compare. For large
  batteries, sample to understand the shape, then write code that runs over
  everything.
- **Classifiers and embeddings** — operate on `answer_text` directly. A derived
  column survives on the frame and the side tables keep working.

**Grounding is an OUTCOME, not a setting — analyse it as one.** On a Claude
battery 5 of 80 search-permitted records searched; on a three-provider battery
OpenAI searched on 14 of 20 and Gemini on none. The same prompt searched on 2 of 5
repetitions and declined on 3. Report grounding rate by model
and probe, and its variability across reps, before anything else.

**Everything retrieval-related must then be conditional on grounding.** Averaging
`n_queries` across all records mixes "searched and issued 2 queries" with "never
searched", and `None` there means the second. The same applies to citations and
sources.

**Side-table row counts are different units. Do not divide them.** One retrieved
source can support several citation spans, so `citations ÷ sources` is not a
citation rate:

| table | one row is |
|---|---|
| `sources` | one retrieved URL |
| `citations` | one citation span or quote occurrence |
| `queries` | one search string the model wrote |

For "what fraction of what it saw did it use", count distinct sources with
`cited == True` against distinct sources — not row counts across tables.

**At small n, read the grounded records individually first.** Aggregate totals
hide structure: in one battery a record retrieved ten sources and cited none of
them, which the totals buried.

**For citation work specifically:** group `sources` by `domain`, not by URL. Ask
how often a domain appears across reps before treating it as stable — the answer
is usually "less than you would think."

---

## 7. When a provider changes something

Model rosters move faster than a study does. Google shipped three Flash models in
six weeks; OpenAI released a whole `gpt-6-*` generation. **Two scripts answer
"what changed", and neither is part of the pip install** — `pyproject.toml`
packages `machine_psych*` only, deliberately, because these make live calls. They
run from a clone.

```python
import sys; sys.path.insert(0, '/content/mp')   # wherever the clone is
```

### Which models exist — free

```python
from drift import tier1
tier1.run(['anthropic', 'openai', 'gemini'])     # diff against the approved roster
tier1.approve(['anthropic', 'openai', 'gemini']) # record today's as the baseline
```

Three states, and the third is the one that matters. **NEW** — offered, not in
your roster. **GONE** — in your roster, no longer offered. **BROKEN** — a model
the capability table uses that the provider no longer serves, which is a battery
that fails at dispatch.

With no roster approved it lists everything and says `INCOMPLETE`, because there
is nothing to diff against. Approve once, then it is four lines.

### Whether the parameters still work — ~40 calls

```python
from drift import tier2
tier2.run(['anthropic'])
```

Probes each declared enum against the live API. This is how `thinking.type:
enabled` was found to need a companion field, and how `reasoning: minimal` turned
out to be in OpenAI's schema and rejected by every model.

### Measuring a new model — 5 calls

```python
from drift import characterise
outcome = characterise.characterise('anthropic/claude-opus-5-5')
print(characterise._render(outcome))
```

Or as the script it is: `python drift/characterise.py anthropic/claude-opus-5-5`.

**The block it prints is PARTIAL — 11 of 22 fields — and says so at the top.**
`reasoning_levels`, `tokens_per_query`, `answer_extraction` and eight others are
not measured here.

- **A new model:** the block is a starting point. Fill in the rest before using
  it in a study, or note explicitly that they were inherited from a sibling.
- **A model that already has an entry:** do NOT paste over it. Take
  `measured_on` and anything that genuinely changed. Pasting drops the eleven
  fields it does not produce and turns measured values into `None` — which reads
  as "the provider cannot" rather than "nobody looked".

### Do not switch models mid-programme

A study run on different models is not comparable with one run before it. Adding
`claude-opus-5-5` or `gpt-6-sol` is its own piece of work, not a change to a
battery about to run.

## 8. What this cannot do

State these rather than designing around them.

- **No page content.** The harness records which URLs were cited and what was
  quoted. It does not fetch the pages. *(Measured once, on 76 pages from one
  battery:* a plain HTTP fetch lost 45% of citations, and the losses were
  systematic — aggregators and review sites blocked, vendor sites did not. A
  headless browser recovered 3 of 10 blocked domains.*)*
- **No causal claims.** *"Publish this and the answer changes"* requires an
  intervention and time. Nothing here demonstrates it.
- **Retrieval is neither stable nor random.** *(One pilot: 3 prompts x 3 providers
  x 8 reps.)* A prompt returned a small stable core of 2–6 domains appearing in
  ≥70% of repetitions, plus a large variable tail; mean pairwise overlap between
  repetitions ran 0.32–0.88 depending on prompt type. Source sets barely overlapped
  BETWEEN providers — for category and comparison prompts, they shared nothing at
  all. **Branded prompts were the most stable, category the least.**
- **Nine capability fields rest on a single observation.** `mp.provisional(model)`
  lists them. Six single-observation claims have already turned out wrong, so
  treat those as provisional rather than fact.

---

## 9. Interpreting output

Not a style preference — a property of the instrument. **A battery costs money and
its failures are quiet.** A wrong condition label, a misread `None`, a column that
silently dropped — each produces a corpus that looks fine and says something
false. And the person reading a result is often not the person who designed the
test.

Two things follow:

**A cell should print what its numbers mean, not only the numbers.** A table the
reader has to reconstruct the logic from is a table that gets skimmed. One
plain-language line — *this is what the result implies* — is what makes a wrong
answer visible.

**A test should state what it checks and what each outcome would imply, before it
runs.** Otherwise the result arrives and the interpretation gets built around
whatever it happened to say, which is how a null result becomes a finding.

Beyond that, match whatever working style the conversation is already using.
