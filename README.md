# machine-psych

An experiment harness for studying how language models judge and recommend. You design a battery of prompts and decide which details to vary. The harness runs it against models from Anthropic, OpenAI and Google, many times over, and saves every raw response with everything the provider reported about how it was produced.

The model is the research subject. The harness is the lab.

## What it's for

A buyer asks an AI assistant which CRM a 60-person software company should shortlist. The model names three vendors and says why each fits. Change one detail (6,000 people instead of 60, no sales-ops team, a tighter budget) and the list can change.

That makes a recommendation a conditional judgment: a function from the buyer's situation to a shortlist. machine-psych is built to map that function. Which details move the answer? Which vendor drops out, and what does the model say about it on the way out? When does the model search the web before answering, and which pages end up in the answer? Measured across enough buyer profiles and categories, those judgments form a map of which buyers each model sends to which companies.

Buyers increasingly start their research this way. In a [2026 G2 survey](https://learn.g2.com/g2-2026-ai-search-insight-report) of more than 1,000 B2B software buyers, 51% said they start with an AI chatbot more often than with Google. G2 sells to software vendors, so read that as a direction rather than a census.

## The method

This is machine psychology: behavioral experiments with a language model as the subject. It is not silicon sampling, which treats a model as a stand-in for human respondents; here the model's own behavior is the object of study. And it is not benchmarking. There's no right answer being scored against.

The method borrows from the social sciences that study human judgment, and adapts them for a subject that can be asked the same question a thousand times.

### Framing a research question

Every investigation starts from a decision someone has to make, usually a buyer's or a marketer's, and turns it into a question about model behavior. "Will AI recommend us?" is too coarse to test. A testable version names the judgment, the conditions and the outcome: for which buyer situations does a model put this vendor on the shortlist, and what reason does it give?

A recommendation is treated as conditional. The finding worth having is rarely an average; it's the detail that flips the answer.

The outcome layers stay separate. Whether a model searches, what it retrieves, what it cites, whether it mentions a vendor, what it says about that vendor, and whether it recommends it to this buyer are six dependent variables, and a change in an earlier one is never taken as evidence for a later one. More citations are not more recommendations.

Unexplained variation is treated as a lead. When the same question gets different answers, the spread is data about the model and a prompt for the next design: which condition, if any, accounts for it?

And every question says whether it's about native behavior or a manipulation. Leaving reasoning at the model's default observes what a user would get. Setting it explicitly tests what reasoning changes. Those are different studies.

### Techniques borrowed from research on people

| Technique | With people | Adapted for models |
|---|---|---|
| Factorial vignettes, after Peter Rossi's factorial survey | Respondents judge short scenarios whose features vary systematically, so each feature's weight in the judgment can be estimated. | Buyer scenarios vary several details at once. A fractional-factorial screening design estimates the effect of each detail from a fraction of the full grid, and follow-up studies focus on the few that matter. |
| Repertory grid, after George Kelly | Show three items and ask which two are alike and how the third differs. The answers reveal the dimensions a person uses. | Triads of competing vendors reveal the dimensions a model uses to sort a category, before any question imposes ours. |
| Projective questions | Asking someone to describe another person's motives draws out views a direct question doesn't. | "Describe the typical buyer of this product" invites a model to commit to a profile. |
| Alternate wordings | Each survey item gets several wordings, so a result doesn't belong to one phrasing. | Each level of each manipulated detail has several paraphrases, balanced across the design. Models can be sensitive to small changes in wording, so this matters at least as much here. |
| Counterbalancing | Question order rotates so position effects don't line up with conditions. | Sentence order rotates across vignettes, and a check confirms that each detail's position is independent of its level. |
| Pretesting | Confirm each manipulation reads as intended before fielding. | Practitioners read the stimuli first, and a study isn't preregistered or run until they sign off. |
| Preregistration | Fix the hypotheses and analysis before collecting data. | The spec, with the rationale for each probe, and the stimulus manifest are saved before the first paid call, and every run records the code that produced it. |
| Codebooks and inter-rater reliability | Two coders apply a codebook independently, their agreement is reported, and disagreements are adjudicated. | Model coders apply the codebook, blinded to which model and condition produced each answer. Their agreement is measured as human coders' would be, every quote they cite is checked against the text, and a person audits a sample of disagreements. |
| Development and held-out samples | Build a coding scheme on one sample and test it on another. | Coding dimensions are developed on a development sample, frozen, then applied to sealed held-out batches drawn with a recorded seed. |

### Designing an investigation

1. Start from the decision, and write the question so it names the judgment, the conditions and the outcome.
2. Discover the dimensions. Open-ended and triad probes show how models describe the category in their own terms, and a development sample is coded to find the recurring dimensions.
3. Freeze them. The codebook is fixed before any held-out data is read.
4. Build the stimuli (vignettes, paraphrases, counterbalanced order) and pretest them with practitioners.
5. Choose the design. A screening design comes first, to find which details move the answer; focused experiments follow on the few that do.
6. Preregister the spec and stimulus manifest, then run a smoke test of a few calls before the full battery.
7. Run it, and validate the corpus (counts, served models, data quality) before reading a single answer.
8. Extract mechanically, code with reliability checks, and only then read closely.
9. Write each finding up as a field note: the question, the design, what changed and what didn't, the proposed mechanism and its strongest alternative, the implication, and the limits.

### What changes when the subject is a model

- A "participant" is a model under a fixed configuration, and the sample comes from repetition. Asking the same question many times gives the distribution of its answers. That is why nothing is ever cached: a cached answer is one observation counted twice.
- Each call starts fresh, with no memory of the last. Within-subject designs have no carryover or fatigue, and a multi-turn conversation is a scripted path rather than an open interview.
- The subject changes under you. Providers update models without notice, so every record carries the exact model that served it, and `drift/` shows what changed before you spend.
- The subject can consult the world mid-interview. Web search makes retrieval part of the response, so search is assigned as a condition, grounding is analyzed as an outcome, and the repetition barrier keeps drift in what the web returns from lining up with conditions.
- Coders can be models too. They need the scrutiny human coders get, plus two checks of their own: a model coder can quote text that isn't there, and can fail to answer without saying so. Quotes are verified, and a missing answer is recorded as missing, never as "no."
- "Believes" describes behavior. "The model believes this product suits small teams" means it says so, consistently, across wordings and repetitions. Nothing here claims access to what a model experiences.

## The shape of a study

```
investigation                     one preregistered spec, one folder of records
└── study                         a research question
    ├── probes                    the questions; each is one or more conversation paths,
    │                             and a path with several strings is a multi-turn conversation
    └── providers                 which models, and the settings for each
        ├── conditions            any setting given as a list becomes a factor to cross
        └── repetitions           the same cell, asked again
                                  └── record: one model response, saved whole
```

A study with three probes, three models, search off and on, and five repetitions is 3 × 3 × 2 × 5 = 90 records. One investigation can hold several studies.

## From spec to record

```
spec (a dict)
  │  load_investigation     validate · resolve intents per model · expand conditions
  │                         · estimate cost                         — sends nothing
  ▼
plan (one row per conversation)
  │  run_investigation      build provider-native requests · raw HTTP · pace to rate limits
  │                         · retry · ride out outages · resume       — writes as it goes
  ▼
records on disk             request sent + raw response + parsed fields + integrity flags
  │  load_corpus
  ▼
corpus (DataFrame)          one row per record, plus side tables: queries, sources,
                            citations, thoughts, response units
```

Nothing is normalized on the way in. Every record is written the moment its call returns, so a run that stops halfway keeps every response it received.

## Intents: one setting, three APIs

A spec says what you want in provider-neutral terms: `reasoning`, `search`, `max_tokens`, `verbosity`, `system`, `max_tool_calls`. The harness translates each intent into whatever the model's API needs. Here is `{"reasoning": "low", "search": True, "max_tokens": 16384}` sent to three models (prompt omitted):

```json
anthropic/claude-opus-5-5
{"model": "claude-opus-5-5", "max_tokens": 16384,
 "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"},
 "tools": [{"type": "web_search_20250305", "name": "web_search"}]}

openai/gpt-5.6-sol
{"model": "gpt-5.6-sol", "max_output_tokens": 16384, "store": false,
 "reasoning": {"effort": "low", "summary": "detailed"},
 "tools": [{"type": "web_search"}]}

gemini/gemini-3.8-flash
{"model": "gemini-3.8-flash", "store": false,
 "generation_config": {"max_output_tokens": 16384, "thinking_level": "low",
                       "thinking_summaries": "auto"},
 "tools": [{"type": "google_search"}]}
```

The translation runs off a capability table that records, per model, what was measured: which reasoning levels exist, whether thinking can be switched off, whether the token budget is shared between thinking and output, what the search tool returns, whether sampling parameters do anything. Each value carries its evidence (`swept` across conditions, a `single` observation, `structural` because it's read from response shape, or `inherited` from a sibling model and not yet re-measured), and `mp.provisional(model)` lists the weak ones.

Because the table is measured, the harness can refuse a study that wouldn't mean what it says:

```
UnmetIntentError: anthropic/claude-opus-5-5 cannot honour:
  reasoning='off': thinking cannot be disabled on this model, so there is no
  reasoning-off arm to run. Levels that do work: low, medium, high, xhigh, max
```

The study either fails at preflight or, with `on_unmet: "exclude"`, drops those cells and records that it did. It also refuses settings a provider silently accepts and ignores. One provider takes `temperature: 0` without complaint and gave six different answers in six runs; a sweep over that setting would produce arms that look different and are identical.

## Conditions and the repetition barrier

Any setting given as a list becomes a factor. `"reasoning": ["low", "high"], "search": [False, True]` produces four conditions, `low|False`, `low|True`, `high|False` and `high|True`, each run for every probe and repetition. `include` and `exclude` trim the grid when a full cross isn't wanted.

Runs are concurrent, with one constraint: **conversations run in parallel within a repetition, and the run waits at each repetition boundary.** Search results drift over minutes and hours. If one condition finished all its repetitions before another started, any drift in what the web returned would line up with the conditions and look like an effect. Holding every arm to the same repetition clock spreads that drift across conditions instead.

## What a record holds

Each record is one JSON file, written as the call returns:

- the exact request body sent, and the raw response body received;
- the parsed answer, plus how it was found (`structural` where the API marks the answer, `heuristic` where it has to be separated from process text);
- retrieval: the queries the model searched, the sources returned, and the citations in the answer, each a row in its own side table (one grounded answer in the test fixtures produced 5 queries, 40 sources and 27 citations);
- reasoning: thinking tokens, and readable thought text where the provider returns it;
- provenance: a content hash of the prompt and of the probe, and the settings as written and as resolved;
- integrity flags raised at parse time.

`_run.json` beside the records holds the spec, the probe hashes and the harness commit, so a corpus can be traced to the exact code that produced it, months later.

## Comparing across providers

Three APIs report different things in different units. The corpus sorts its columns into three tiers:

- Tier 1 means the same everywhere: identity, status, the answer, the prompt and its hash.
- Tier 2 is comparable with a stated caveat. Token counts are an example: one provider bills thinking inside output tokens, another reports it separately.
- Tier 3 exists only where a provider can report it, such as the full retrieval set or readable reasoning.

`mp.normalize(corpus)` returns Tiers 1 and 2 as one view, plus a report of everything set aside and why.

Two rules matter most when comparing. **`None` is not zero.** `None` means the provider cannot report the value; `0` means it reported none happened, and filling one with the other changes a finding. And **`grounded` is an outcome, not a setting.** `search: True` only permits search; the model decides whether to use it, and grounding rates differ a lot between models and prompts. Compare the search-off and search-on arms, which are assigned. Comparing grounded with ungrounded records inside the search-on arm compares records the model chose to search, which is a different question.

## Catching the failures that don't announce themselves

Measuring a model is easy to do badly in ways that look fine. Most of this harness exists because of a failure that once produced a clean-looking corpus.

| What can go wrong | What the harness does |
|---|---|
| The same question gets different answers on different runs. | Measures it. Every design repeats each cell, and no answer is ever cached. |
| A setting was silently ignored. | Refuses settings the capability table marks as inert for that model. |
| The model ran out of token budget while thinking, wrote a short answer in what was left, and the call came back `ok`. | Flags it as `budget_exhausted`, whether or not the response finished. |
| A provider served a different model than the one requested. | Flags `served_model_mismatch`. |
| The model searched, sources came back, and the answer cited none of them. | Flags `searched_but_cited_nothing`. |
| The answer had to be separated from process text, and some answer may have landed on the wrong side. | Flags `answer_split_heuristically` for text written between searches. |
| A provider changed its models or parameters. | `drift/` checks which models exist and whether settings still behave, before you spend. |
| Nobody can later say which code produced a result. | Records the commit on every run; `mp.require_commit(sha)` pins one. |

`mp.data_quality(corpus)` returns every flagged record (each integrity flag, and every status that isn't a usable answer) with the model and probe, as a table to save beside the results.

## Running studies that take hours

- `concurrency="auto"` keeps a token bucket per rate-limit pool, learned from the rate-limit headers each response carries. For providers that send none, you declare the limits from the provider console. Daily caps are tracked too, and reaching one stops the run cleanly.
- A dropped connection pauses the affected provider, or every provider if the network itself is down, and the run continues when it returns.
- `resume=True` reads the records already on disk, skips them, and finishes the rest in the same folder. OpenAI calls carry an idempotency key, so a retry after a lost response returns the original instead of paying twice.
- Progress shows as a bar, with a live count of records that need attention.

## Watching the providers

Models and parameters change without notice. Three scripts check what changed:

```bash
python drift/tier1.py --approve    # record today's model roster as the baseline, once
python drift/tier1.py              # which models are new, gone, or broken (free)
python drift/tier2.py              # do the parameters still mean what the table says
python drift/characterise.py anthropic/claude-opus-5-5   # measure a new model
```

`characterise` measures what can be measured in a few calls and prints a partial capability entry, labeled as partial, for the rest to be filled in by hand.

## Quick start

```bash
git clone https://github.com/MichaelKoach/machine-psych.git
cd machine-psych
pip install -e .
```

Clone rather than install from GitHub: `drift/` is not installed by pip, because those scripts make live calls that cost money.

```python
import machine_psych as mp
from machine_psych import paths

for provider in ("anthropic", "openai", "gemini"):
    paths.set_api_key(provider, "...")       # or set ANTHROPIC_API_KEY etc.
mp.set_base("research")                      # where specs and records are saved

sizes = {"small": "6-person", "mid": "60-person", "large": "6,000-person"}
spec = {
    "investigation_id": "crm_shortlist",
    "studies": [{
        "study_id": "company_size",
        "probes": [{"probe_id": size,
                    "prompt_paths": [[f"We're a {who} B2B software company. "
                                      "Which CRM should we shortlist, and why?"]]}
                   for size, who in sizes.items()],
        "providers": {
            "anthropic/claude-opus-5-5": {"search": [False, True], "max_tokens": 8192,  "repetitions": 5},
            "openai/gpt-5.6-sol":        {"search": [False, True], "max_tokens": 16384, "repetitions": 5},
            "gemini/gemini-3.8-flash":   {"search": [False, True], "max_tokens": 65536, "repetitions": 5},
        }}]}

run = mp.load_investigation(spec)            # validates, resolves, estimates cost; sends nothing
results, _ = mp.run_investigation(run, concurrency="auto")
corpus = mp.load_corpus("crm_shortlist")     # 90 records

mp.data_quality(corpus, summary=True)        # anything that needs attention, by model
mp.mentions(corpus, ["Salesforce", "HubSpot", "Pipedrive"])   # extract before you read
mp.citations(corpus)                         # one row per citation, with its quote
```

Reasoning is left unset here, so each model uses its own default. That observes the behavior a user would get, which is usually what a study of recommendations wants. Gemini's `max_tokens` is its maximum because its budget is shared between thinking and output, and a lower cap can bind a response that still comes back `ok`.

[HARNESS.md](HARNESS.md) is the operating manual. Its Part A covers the order of operations, the rules that silently invalidate an analysis, how to qualify a new model, and what `load_investigation` will refuse.

### What's in the repo

```
machine_psych/   the package
drift/           checks for what the providers changed (live calls, costs money;
                 not installed by pip, so clone to use it)
tests/           the test suite, plus recorded API responses from each provider
HARNESS.md       the operating manual
```

## Design commitments

**Raw HTTP, no SDK.** SDKs fill absent fields with `null`, and the difference between absent and null often carries information. One provider omits a citation offset when it's zero; another omits a token count when nothing was spent.

**Save everything, and normalize at analysis.** Raw responses go to disk unchanged, and a comparable view is computed from them afterward. Unified gateway layers were tried and dropped, because they throw away the structure this research reads.

**No caching, ever.** Every call is a measurement, and repetition is the instrument. Variation between identical calls is a property to measure, not noise to remove.

**Let the API refuse what it will refuse.** Every provider rejects unknown parameters and names the field, which beats any local check and stays current for free. The harness checks the opposite case: settings a provider accepts and ignores.

## What it doesn't claim

It measures what models say and do. Whether changing what a model says changes a company's pipeline is a separate question; the harness can support a study of that, with the right design and sales data, but it doesn't answer it alone.

## Who built it

machine-psych is built and maintained by Michael Koach at Construal Research, an independent research firm that studies how AI assistants interpret companies and decide which vendors to recommend to which buyers.

Questions, ideas, or a category you'd like to see mapped: michael@construalresearch.com
