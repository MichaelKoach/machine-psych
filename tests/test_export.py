"""Export tests.

The format exists because of a hard constraint — the near-raw one did not fit a
context window — so several of these assert on SIZE and on the honesty of what
the export claims about itself, not only on structure.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib

import pytest

import machine_psych.runner as R
from machine_psych.export import HOW_TO_READ, export_corpus

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

SPEC = {
    "investigation_id": "e",
    "metadata": {"name": "export"},
    "studies": [{
        "study_id": "s1",
        "probes": [{"probe_id": "triad", "rationale": ["why this exists"],
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
def exported(tmp_path):
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(SPEC)
        res, _ = R.run_investigation(run, dispatch=_fake, verbose=False,
                                     export=False)
        dest = export_corpus("e", quiet=True)
    return json.loads(dest.read_text()), pathlib.Path(res.attrs["run_dir"])


# ═══════════════════════════════════════════════════════════════════════════════
# Structure
# ═══════════════════════════════════════════════════════════════════════════════

def test_derived_tables_not_raw_responses(exported):
    payload, _ = exported
    assert set(payload) >= {"how_to_read_this", "provenance", "spec",
                            "capabilities", "records", "citations", "sources",
                            "queries", "thoughts"}
    assert all("_raw_response" not in r for r in payload["records"])


def test_encrypted_payloads_are_gone(exported):
    """Signatures and widget markup are most of the raw bulk and no analysis
    reads them."""
    payload, _ = exported
    blob = json.dumps(payload["records"] + payload["citations"] +
                      payload["sources"] + payload["queries"])
    assert "encrypted_content" not in blob
    assert "search_suggestions" not in blob


def test_capabilities_are_in_the_export(exported):
    """The section most likely to be dropped as unnecessary, and it is not.

    An analysing conversation reading a None in `thought_text` cannot otherwise
    tell whether the provider CANNOT expose reasoning or simply DID NOT — one is
    a fact about an API and the other a finding about a model, and they lead to
    opposite conclusions.
    """
    payload, _ = exported
    caps = payload["capabilities"]
    assert caps["anthropic/claude-sonnet-5"]["readable_reasoning"] is False
    assert caps["gemini/gemini-3.7-flash"]["readable_reasoning"] is True
    assert caps["anthropic/claude-sonnet-5"]["retrieval_set"] is True
    assert {c["citation_offsets"] for c in caps.values()} == {
        "quoted", "char", "byte"}


def test_provenance_is_three_mechanisms(exported):
    payload, _ = exported
    prov = payload["provenance"]
    assert "harness" in prov and "probe_hashes" in prov
    assert prov["investigation"] == "e"


def test_spec_travels_with_the_corpus(exported):
    """Including the rationale fields, which say what each probe was FOR — an
    analysis that cannot see the intent will infer one."""
    payload, _ = exported
    probe = payload["spec"]["studies"][0]["probes"][0]
    assert probe["rationale"] == ["why this exists"]


# ═══════════════════════════════════════════════════════════════════════════════
# NaN handling — the one place a fill is correct
# ═══════════════════════════════════════════════════════════════════════════════

def test_nan_becomes_null_not_nan(exported):
    """JSON has no NaN, and `null` is the honest rendering of an absent value.

    This is emphatically NOT filling a Tier 3 column with False: null preserves
    "unknown or inapplicable" where False would claim the provider declined.
    """
    payload, _ = exported
    assert "NaN" not in json.dumps(payload)


def test_tier3_nulls_survive_the_export(exported):
    payload, _ = exported
    by_provider = {r["provider"]: r for r in payload["records"]}
    assert by_provider["gemini"]["grounded"] is True
    assert by_provider["openai"]["grounded"] is None
    assert by_provider["openai"]["n_sources_retrieved"] is None
    assert by_provider["anthropic"]["n_sources_retrieved"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# how_to_read_this
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("required", [
    "NOT comparable",          # out_tok
    "offset_unit",             # three citation mechanisms
    "PROVIDER-SCOPED",         # condition
    "PERMISSION",              # conditional grounding
    "Extract before you read",
    "truncated",
    "100x",                    # billed vs processed
])
def test_how_to_read_covers_the_traps(required):
    """Each of these is a trap that has actually been fallen into.

    The document is the only thing travelling with the corpus, so anything not
    said here has to be rediscovered by whoever reads it.
    """
    assert required in " ".join(HOW_TO_READ)


def test_how_to_read_warns_about_offset_units():
    text = " ".join(HOW_TO_READ)
    assert "byte" in text and "silently wrong" in text


# ═══════════════════════════════════════════════════════════════════════════════
# Size — the constraint that motivated the format
# ═══════════════════════════════════════════════════════════════════════════════

def test_export_is_smaller_than_raw(exported):
    payload, run_dir = exported
    raw = sum(p.stat().st_size for p in run_dir.glob("[0-9]*.json"))
    exported_size = len(json.dumps(payload))
    assert exported_size < raw


def test_include_raw_for_attaches_only_named_records(tmp_path):
    """The escape hatch: a conversation needing to see exactly what came back for
    one record, without re-exporting everything in a format that does not fit."""
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(SPEC)
        R.run_investigation(run, dispatch=_fake, verbose=False, export=False)
        dest = export_corpus("e", include_raw_for=[0], quiet=True)
    payload = json.loads(dest.read_text())
    attached = [r["record_id"] for r in payload["records"] if "_raw_response" in r]
    assert attached == [0]


def test_truncation_is_recorded_never_silent(tmp_path):
    """An analysis mistaking a cut answer for a short one would read an export
    SETTING as a finding about verbosity.

    `answer_chars` deliberately keeps the TRUE length so length comparisons stay
    valid even on a truncated export.
    """
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(SPEC)
        R.run_investigation(run, dispatch=_fake, verbose=False, export=False)
        dest = export_corpus("e", max_answer_chars=100, quiet=True)
    payload = json.loads(dest.read_text())
    cut = [r for r in payload["records"] if r.get("answer_truncated_for_export")]
    assert cut
    for r in cut:
        assert len(r["answer_text"]) == 100
        assert r["answer_chars"] == r["answer_truncated_for_export"]
        assert r["answer_chars"] > 100


def test_size_report_measures_rather_than_assumes(tmp_path, capsys):
    """An earlier version asserted "the answers are the bulk" and offered
    truncation, which saved 5% on the first corpus measured — the side tables
    dominated.

    The breakdown is printed so the decision is made against THIS corpus rather
    than against a guess about corpora in general.
    """
    R.set_base(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        run = R.load_investigation(SPEC)
        R.run_investigation(run, dispatch=_fake, verbose=False, export=False)
    export_corpus("e")
    out = capsys.readouterr().out
    assert "tokens/record" in out
    assert any(section in out for section in ("sources", "records", "citations"))


# ═══════════════════════════════════════════════════════════════════════════════
# Generality — from the 2026-09-05 fourth-provider audit
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_guide_does_not_count_providers():
    """`how_to_read_this` travels WITH the corpus to an analysing conversation.

    It said "Three mechanisms", "`quote` is populated for all three", "None on
    two providers" — static prose that became FALSE the moment a fourth provider
    appeared whose citations are a bare URL list with no quote at all.

    A guide that ships false claims to the reader is worse than no guide, and
    this one is the only orientation an analysing conversation gets.
    """
    import re
    text = " ".join(HOW_TO_READ)
    for phrase in (r"all three", r"Three mechanisms", r"two providers",
                   r"one provider only", r"on one provider and not another"):
        assert not re.search(phrase, text), (
            f"the guide asserts a provider COUNT ({phrase!r}) — it must point at "
            f"`capabilities` and `citation_mechanisms` instead")

    assert "citation_mechanisms" in text
    assert "capabilities" in text


def test_citation_mechanisms_are_derived_from_the_records(exported):
    """Including whether each carries a quote.

    A provider returning a numbered source list has no offsets AND no quote — it
    attributes the whole answer to the SET rather than a span to a source. An
    analysis assuming a quote exists would silently drop it, so the export says
    so per mechanism.
    """
    payload, _ = exported
    mechanisms = payload["citation_mechanisms"]
    assert mechanisms, "no mechanisms recorded"
    for info in mechanisms.values():
        assert set(info) >= {"citations", "with_quote", "providers", "note"}
        if info["with_quote"] == info["citations"]:
            assert "compare these" in info["note"]
        elif info["with_quote"] == 0:
            assert "only the URL" in info["note"]


def test_statuses_seen_is_read_not_asserted(exported):
    payload, _ = exported
    assert payload["statuses_seen"] == sorted(set(payload["statuses_seen"]))
    assert all(s in ("ok", "truncated", "incomplete", "unavailable", "error")
               for s in payload["statuses_seen"])
