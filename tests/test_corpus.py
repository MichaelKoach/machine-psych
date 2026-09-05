"""Corpus loading and side tables.

Built on a real three-provider run, because the questions this module answers only
become questions when a corpus holds more than one provider — an empty citations
table is unambiguous with one provider and ambiguous with three.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib

import pandas as pd
import pytest

import machine_psych.runner as R
from machine_psych.corpus import (capability_note, citations, load_corpus,
                                  queries, sources, thoughts, units)

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

SPEC = {
    "investigation_id": "c",
    "studies": [{
        "study_id": "s1",
        "probes": [{"probe_id": "triad", "prompt_paths": [["Which two?"]]}],
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


def _response(name, provider):
    return json.loads((FIXTURES / provider / f"{name}.json").read_text())["response"]


def _fake(cfg):
    if cfg.get("generation_config"):
        return _response("grounded_ok", "gemini")
    if cfg.get("store") is False and "input" in cfg:
        return _response("grounded_ok", "openai")
    return _response("grounded_ok", "anthropic")


@pytest.fixture
def corpus(tmp_path):
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(SPEC)
        R.run_investigation(run, dispatch=_fake, verbose=False, export=False)
        return load_corpus("c")


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

def test_one_frame_holds_every_provider(corpus):
    assert len(corpus) == 6
    assert set(corpus.provider) == {"anthropic", "openai", "gemini"}


def test_structural_nulls_raise_rather_than_fill(corpus, tmp_path):
    """`turn` is written by the runner on every record.

    A null means the file is corrupt, not that the value is unknown — and
    defaulting it to 0 would silently make a turn-3 record look like turn 0.
    """
    run_dir = pathlib.Path(corpus.attrs["run_dir"])
    path = sorted(run_dir.glob("[0-9]*.json"))[0]
    record = json.loads(path.read_text())
    record["turn"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="corrupt"):
        with contextlib.redirect_stdout(io.StringIO()):
            load_corpus("c")


def test_side_tables_are_rederived_from_the_raw_response(corpus):
    """Not read from whatever the runner wrote.

    If the two ever disagree, trusting the stored value propagates the stale one
    — and a parser improving after a corpus was collected is the normal case,
    not an edge one.
    """
    assert len(citations(corpus)) > 0
    assert set(citations(corpus).provider) == {"anthropic", "openai", "gemini"}


# ═══════════════════════════════════════════════════════════════════════════════
# The filtering bug
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_side_tables_follow_the_filter(corpus, provider):
    """Silent wrong data, and the kind that looks plausible.

    The cache is keyed by investigation/run and that key SURVIVES a filter, so an
    earlier version returned every provider's citations against a one-provider
    frame. Every row was real, the count was wrong, and nothing would have
    flagged it.
    """
    sub = corpus[corpus.provider == provider]
    cited = citations(sub)
    assert set(cited.provider) <= {provider}
    assert len(cited) < len(citations(corpus))


def test_an_empty_filter_yields_empty_side_tables(corpus):
    empty = corpus[corpus.provider == "nonexistent"]
    assert len(citations(empty)) == 0
    assert len(sources(empty)) == 0


def test_a_rebuilt_frame_raises_rather_than_guessing(corpus):
    """`attrs` does not survive `concat`, which is why the tables are not stored
    there in the first place. The error says to reload rather than returning
    something plausible."""
    rebuilt = pd.DataFrame(corpus.to_dict("records"))
    with pytest.raises(KeyError, match="Reload"):
        citations(rebuilt)


# ═══════════════════════════════════════════════════════════════════════════════
# Three mechanisms, one table
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_three_citation_mechanisms_in_one_frame(corpus):
    cited = citations(corpus)
    assert set(cited.offset_unit) == {"quoted", "char", "byte"}


def test_quote_is_populated_regardless_of_mechanism(corpus):
    """Offsets are provenance; the quote is what an analysis compares.

    Extracted where only offsets exist, read where it is supplied — and a loader
    applying the wrong offset convention produces spans that are silently wrong
    and drift further into the answer.
    """
    assert citations(corpus)["quote"].notna().all()


def test_domain_comes_from_title_where_urls_are_redirects(corpus):
    """One provider's URLs are redirects whose netloc is always the same host, so
    parsing them would give one meaningless value for every source."""
    cited = citations(corpus)
    gem = cited[cited.provider == "gemini"]
    assert len(gem)
    assert not gem.domain.str.contains("vertexaisearch").any()
    assert gem.domain.nunique() > 1


# ═══════════════════════════════════════════════════════════════════════════════
# Absence vs unavailability
# ═══════════════════════════════════════════════════════════════════════════════

def test_cited_and_retrieved_are_separate_booleans(corpus):
    """`retrieved` is None where a provider does not report what it saw.

    "Does not report" is a different claim from "saw nothing", and conflating
    them makes the quieter providers look more selective than the one that
    publishes its whole retrieval set.
    """
    srcs = sources(corpus)
    anth = srcs[srcs.provider == "anthropic"]
    others = srcs[srcs.provider != "anthropic"]
    assert anth.retrieved.notna().all()
    assert others.retrieved.isna().all()


def test_retrieved_only_excludes_unknown_not_just_false(corpus):
    """None must not match a `retrieved == True` filter."""
    only = sources(corpus, retrieved_only=True)
    assert set(only.provider) == {"anthropic"}


def test_capability_note_explains_an_empty_table(corpus):
    """The thing a caller needs to read an empty side table correctly.

    An empty citations frame for a provider means it cited nothing OR that it
    cannot cite — one is a finding about the model, the other about the API.
    """
    note = capability_note(corpus, "retrieval_set")
    assert note["anthropic/claude-sonnet-5"] is True
    assert note["openai/gpt-5.6-sol"] is False

    reasoning = capability_note(corpus, "readable_reasoning")
    assert reasoning["anthropic/claude-sonnet-5"] is False
    assert reasoning["gemini/gemini-3.7-flash"] is True


def test_grounded_is_meaningful_on_one_provider_only(corpus):
    """Only where providing the tool is a permission rather than a condition."""
    assert corpus[corpus.provider == "gemini"].grounded.notna().all()
    assert corpus[corpus.provider != "gemini"].grounded.isna().all()


def test_tier3_columns_are_never_filled(corpus):
    """A fill is a claim.

    `fillna(0)` on a token sum says a failed record contributed nothing — true.
    `fillna(False)` on `grounded` says a provider declined to do something it
    cannot do — false, and it destroys the distinction the whole design turns on.
    """
    for col in ("grounded", "n_sources_retrieved", "thought_text"):
        assert corpus[col].isna().any(), f"{col} has no nulls — was it filled?"


# ═══════════════════════════════════════════════════════════════════════════════
# Identity and grouping
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("table", [citations, sources, queries, units])
def test_side_tables_carry_identity(corpus, table):
    """So they group without a merge, which is the whole reason for duplicating
    these columns onto every row."""
    df = table(corpus)
    assert len(df)
    for col in ("record_id", "provider", "probe", "rep", "condition"):
        assert col in df.columns, f"{table.__name__} lacks {col}"


def test_grouping_works_without_a_merge(corpus):
    grouped = citations(corpus).groupby(["provider", "offset_unit"]).size()
    assert len(grouped) == 3


def test_record_id_filter(corpus):
    assert set(citations(corpus, record_id=0).record_id) <= {0}


def test_prompt_hash_joins_across_providers(corpus):
    """The mechanism the cross-provider design rests on.

    The same prompt sent to three providers produces three records that join on
    content, with no shared identifier and no coordination.
    """
    assert corpus.prompt_hash.nunique() == 1
    assert corpus.provider.nunique() == 3
