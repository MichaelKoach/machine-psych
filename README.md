# machine-psych

A harness for running designed batteries of prompts against language models and
recording every response with its metadata. **The model is the research
subject.**

Methodologically this is **qualitative machine psychology**. Machine psychology —
psychology-inspired behavioural experiments run on LLMs — is an established field
and is almost entirely quantitative: inventories, scales, forced-choice tasks.
Applying *qualitative interview technique* to models is the gap this occupies.

Two things this is not. It is **not silicon sampling**, which uses an LLM as a
stand-in for human respondents; that literature's central objection — that
research on an LLM is research on a model of a population, not on the population
— is fatal to that use and an accurate description of this one. And it is **not
benchmarking**: there is no ground truth being scored against. The model's
behaviour is the data.

## What is in here

```
machine_psych/    the package — importable, does the work
drift/            operational scripts — live API calls, cost money, NOT installed
tests/            the suite, plus recorded API responses
```

**`drift/` is deliberately not part of the package.** `pyproject.toml` packages
`machine_psych*` only, so a `pip install` gets the library and not the tooling
that spends money. To run the drift checks, clone the repo rather than installing
it — `from drift import tier1` after a pip install will fail, and that has cost
time.

## Install

```python
!pip install -q git+https://github.com/MichaelKoach/machine-psych.git@COMMIT_SHA
```

**Pin the commit.** A version string is something a person typed; a commit SHA is
verifiable. A corpus recording the SHA it was collected under can be traced back
to the exact parsing code that produced it — which matters, because parsers have
bugs and those bugs get fixed after corpora are collected.

## Use

```python
import machine_psych as mp
mp.install_report()
```

## Design commitments

**Raw HTTP, no SDK.** Provider SDKs render absent optional fields as `null`, and
absent-versus-null repeatedly carries information — one provider omits a citation
offset when it is zero, another omits a token count when nothing was spent.

**No client-side validation of anything the API will catch.** Every provider
rejects unknown parameters and names the offending field. Those rejections are
better instruments than anything written here and they stay current for free.
*But validate what the API will silently accept* — one provider takes a sampling
parameter and ignores it, so a sweep would produce arms that look different and
are identical.

**Normalise at analysis, never at collection.** Raw responses go to disk
unchanged; a normalised view is a function over the corpus rather than a
constraint on it. Unified gateway layers were evaluated empirically and rejected:
they discard the structure this research reads.

**No response caching, ever.** Every call is a measurement, and repetitions are
the instrument. Run-to-run variance on identical input is a measured property,
not noise to optimise away.

## Watching what the providers change

Providers ship constantly; in one week during the build, four API surfaces moved
and none of them broke loudly. `drift/` is what looks:

```bash
python drift/tier1.py --approve   # establish the roster baseline, once
python drift/tier1.py             # which models exist. free
python drift/tier2.py             # do the parameters still mean what we recorded
```

Structural checks on every record run at parse time instead of on a schedule —
see `machine_psych/integrity.py`. A third scheduled tier that compared response
SHAPES was built and cut: it sampled a stochastic process once and could not tell
a changed model from a different output. **Check declarations on a schedule;
check behaviour at the point of use.**

`drift/REFRESH.md` is the fixture refresh procedure. It has found four API
changes in a single pass and is the only thing here that can contradict this
document.
