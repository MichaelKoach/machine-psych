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
