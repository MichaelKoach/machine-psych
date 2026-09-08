"""Analysis layer tests.

The important one is `test_normalize_loses_nothing`. The design commitment is
normalise at ANALYSIS, never at collection — and that commitment is empty unless
a Tier 3 finding is still reachable after normalising. If it is not, this is a
gateway library with extra steps and the whole argument for building it collapses.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib

import pandas as pd
import pytest

import machine_psych.runner as R
from machine_psych.analysis import (
    TIER1,
    TIER2,
    TIER3,
    format_stability,
    index,
    mentions,
    normalize,
    read,
    verdict,
)
from machine_psych.corpus import citations, load_corpus, sources, thoughts

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

SPEC = {
    "investigation_id": "a",
    "studies": [{
        "study_id": "s1",
        "probes": [{"probe_id": "triad",
                    "prompt_paths": [["Which two are most alike?"]]}],
        "providers": {
            "anthropic/claude-sonnet-5": {"reasoning": "off", "search": True,
                                          "repetitions": 2},
            "openai/gpt-5.6-sol": {"reasoning": "high", "search": True,
                                   "repetitions": 2},
            "gemini/gemini-3.7-flash": {"reasoning": "medium", "max_tokens": 16384,
                                        "search": True, "repetitions": 2},
        },
    }],
}


def _r(provider, name="grounded_ok"):
    return json.loads((FIXTURES / provider / f"{name}.json").read_text())["response"]


def _fake(cfg):
    if cfg.get("generation_config"):
        return _r("gemini")
    if cfg.get("store") is False and "input" in cfg:
        return _r("openai", "reasoning_summary")
    return _r("anthropic")


@pytest.fixture
def corpus(tmp_path):
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(SPEC)
        R.run_investigation(run, dispatch=_fake, verbose=False, export=False)
        return load_corpus("a")


# ═══════════════════════════════════════════════════════════════════════════════
# normalize — the test of the premise
# ═══════════════════════════════════════════════════════════════════════════════

def test_normalize_loses_nothing(corpus):
    """THE test. Tier 3 must still be reachable after normalising.

    The commitment is normalise at ANALYSIS, never at collection: raw responses go
    to disk unchanged and a narrow view is a FUNCTION over them rather than a
    CONSTRAINT on them. Gateway libraries do the opposite and the structure is
    permanently gone.

    If this fails, the whole argument for building this instead of using one of
    them collapses.
    """
    view, _report = normalize(corpus)
    assert "grounded" not in view.columns
    assert "thought_text" not in view.columns

    # and yet
    assert len(thoughts(corpus)) > 0
    assert len(sources(corpus, retrieved_only=True)) > 0
    assert len(citations(corpus)) > 0


def test_normalize_returns_a_report_rather_than_printing(corpus):
    """A data function with a side effect is invisible when called inside another
    function and unusable when the caller wants the list programmatically."""
    view, report = normalize(corpus)
    assert isinstance(view, pd.DataFrame)
    assert report.dropped and isinstance(report.dropped, dict)


def test_report_names_which_providers_have_each_dropped_column(corpus):
    """"Dropped" is not "unavailable" — it says where to go and get it."""
    _, report = normalize(corpus)
    assert report.by_provider["n_sources_retrieved"] == [
        "anthropic/claude-sonnet-5"]
    assert set(report.by_provider["thought_text"]) == {
        "gemini/gemini-3.7-flash", "openai/gpt-5.6-sol"}


def test_report_carries_caveats_on_tier2(corpus):
    """A Tier 2 column is comparable only with its caveat attached.

    `out_tok_reported` is the sharp case: two providers count thinking inside it
    and one keeps it separate, so comparing it directly measures accounting
    convention rather than output.
    """
    _, report = normalize(corpus)
    assert "NOT comparable" in report.caveats["out_tok_reported"]
    assert "answer_chars" in report.caveats["out_tok_reported"]


def test_view_keeps_tier1_and_tier2_only(corpus):
    view, _ = normalize(corpus)
    assert set(view.columns) <= set(TIER1) | set(TIER2)
    assert not set(view.columns) & set(TIER3)


def test_tier_lists_do_not_overlap():
    """A column in two tiers would be both comparable and not."""
    assert not set(TIER1) & set(TIER2)
    assert not set(TIER1) & set(TIER3)
    assert not set(TIER2) & set(TIER3)


# ═══════════════════════════════════════════════════════════════════════════════
# index and read
# ═══════════════════════════════════════════════════════════════════════════════

def test_index_shows_answer_chars_not_out_tok(corpus):
    """Putting a non-comparable number in the default view is how it ends up in a
    comparison."""
    cols = set(index(corpus).columns)
    assert "answer_chars" in cols
    assert "out_tok_reported" not in cols


def test_read_flags_an_inferred_answer(corpus):
    """One provider infers the answer from block position; the others are told.

    Analysis should know which kind of claim it is reading, and a caveat that
    only lives in a docstring is a caveat nobody sees.
    """
    anth = corpus[corpus.provider == "anthropic"].iloc[0]
    text = read(corpus, record_id=int(anth.record_id), return_text=True)
    assert "INFERRED" in text

    other = corpus[corpus.provider == "openai"].iloc[0]
    assert "INFERRED" not in read(corpus, record_id=int(other.record_id),
                                  return_text=True)


def test_read_survives_a_missing_answer(corpus):
    """NaN is a float and is truthy, so `or ""` does not catch it and slicing
    raises — the bug that shipped in two predecessor modules."""
    broken = corpus.copy()
    broken.loc[broken.index[0], "answer_text"] = float("nan")
    broken.attrs.update(corpus.attrs)
    text = read(broken, chars=50, return_text=True)
    assert "no answer" in text


def test_read_shows_domains_not_redirect_urls(corpus):
    """One provider's URLs are redirects whose host is always the same.

    An earlier version reached for the private side-table helper and bypassed the
    enrichment the public accessor adds — a private helper used for convenience
    is how two views of one table drift apart.
    """
    gem = corpus[corpus.provider == "gemini"].iloc[0]
    text = read(corpus, record_id=int(gem.record_id), return_text=True)
    assert "vertexaisearch" not in text


def test_read_does_not_claim_retrieval_where_it_is_unknown(corpus):
    """`retrieved` is None where a provider does not report what it saw, so the
    count must dropna rather than fillna — otherwise the line says it saw
    nothing."""
    oa = corpus[corpus.provider == "openai"].iloc[0]
    text = read(corpus, record_id=int(oa.record_id), return_text=True)
    assert "retrieved" not in text

    anth = corpus[corpus.provider == "anthropic"].iloc[0]
    assert "retrieved" in read(corpus, record_id=int(anth.record_id),
                               return_text=True)


def test_filters_reject_unknown_columns(corpus):
    with pytest.raises(KeyError, match="not a column"):
        index(corpus, nonexistent="x")


# ═══════════════════════════════════════════════════════════════════════════════
# mentions
# ═══════════════════════════════════════════════════════════════════════════════

def test_mentions_defaults_to_whole_word():
    """A real failure: a brand name matched inside longer words and the inflated
    count looked plausible enough to survive review."""
    df = pd.DataFrame([{"record_id": 0, "provider": "p", "probe": "x",
                        "condition": "c", "rep": 0, "turn": 0,
                        "answer_text": "We discussed Discuss.io and discussion."}])
    whole = mentions(df, ["Discuss"])
    partial = mentions(df, ["Discuss"], whole_word=False)
    assert int(whole["count"].sum()) < int(partial["count"].sum())


def test_mentions_extracts_across_every_record(corpus):
    """Extract before you read. Every qualitative claim this project made from a
    single read has died at a larger n."""
    df = mentions(corpus, ["HubSpot", "Pipedrive"])
    assert len(df)
    assert set(df.columns) >= {"record_id", "provider", "term", "count"}


def test_mentions_of_an_empty_vocabulary_is_empty(corpus):
    assert len(mentions(corpus, [])) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# verdict
# ═══════════════════════════════════════════════════════════════════════════════

def test_verdict_in_heading_is_false_never_none(corpus):
    """REGRESSION 9. `~df.in_heading` raises on None, and a filter that raises is
    worse than one that is merely wrong."""
    v = verdict(corpus, r"most alike|leads on|starts at")
    assert v.in_heading.isin([True, False]).all()
    assert not v.in_heading.isna().any()


def test_verdict_reports_position_as_a_fraction(corpus):
    v = verdict(corpus, r"HubSpot|Pipedrive")
    found = v[v.found]
    assert len(found)
    assert ((found.position >= 0) & (found.position <= 1)).all()


def test_verdict_pattern_is_caller_supplied(corpus):
    """Verdict phrasing is probe-specific; a built-in pattern would fit the probe
    it was written for and silently miss every other."""
    assert not verdict(corpus, r"zzz-no-such-phrase").found.any()


# ═══════════════════════════════════════════════════════════════════════════════
# format_stability
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_headers_is_nan_not_perfect_stability(corpus):
    """Backwards would be worse than missing.

    This is a proxy for "will regex extraction work", and answers with no
    structure at all are the case where it definitely will not. Scoring that 1.0
    inverts the signal the function exists to give.
    """
    fs = format_stability(corpus)
    headerless = fs[fs.mean_headers == 0]
    assert len(headerless)
    assert headerless.jaccard.isna().all()


def test_identical_headers_score_one():
    df = pd.DataFrame([
        {"record_id": i, "provider": "p", "probe": "x", "condition": "c",
         "rep": i, "turn": 0, "answer_text": "# A\ntext\n## B\nmore"}
        for i in range(3)])
    fs = format_stability(df)
    assert fs.iloc[0].jaccard == 1.0


def test_disjoint_headers_score_zero():
    df = pd.DataFrame([
        {"record_id": 0, "provider": "p", "probe": "x", "condition": "c",
         "rep": 0, "turn": 0, "answer_text": "# Alpha\ntext"},
        {"record_id": 1, "provider": "p", "probe": "x", "condition": "c",
         "rep": 1, "turn": 0, "answer_text": "# Beta\ntext"}])
    assert format_stability(df).iloc[0].jaccard == 0.0
