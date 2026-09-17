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

---

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
| **provisional / swept / structural** | how a capability was measured. `swept` = varied across cases. `structural` = read from response shape, so one observation suffices. `provisional` = ONE observation, treat as unverified. |

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

**Grounding is rare unless the prompt needs it.** These prompts described a
business situation and asked for a recommendation — answerable from parametric
knowledge, so the models mostly declined to search. A prompt asking what a company
announced this year grounds nearly every time.

The durable point: **grounded records dominate cost, and whether a record grounds
is not yours to set.** An ungrounded call is a fraction of a cent; a grounded one
ran 20–50K input tokens in an earlier battery, roughly $0.10–0.30. Read the
estimate, then expect the floor.

**Budget time, not just money.** Dispatch is sequential: ~40 seconds per record
in that battery, and Opus averaged 52s against Sonnet's 29s. A 1,000-record
battery is most of a day.

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
results, _ = mp.run_investigation(run)     # the battery. writes to Drive as it goes.
corpus = mp.load_corpus("liftoff_visibility")
```

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

**Grounding is an OUTCOME, not a setting — analyse it as one.** Measured on one
battery: 5 of 80 search-permitted records actually searched, and the same prompt
searched on 2 of 5 repetitions and declined on 3. Report grounding rate by model
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

## 7. What this cannot do

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

## 8. Interpreting output

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
