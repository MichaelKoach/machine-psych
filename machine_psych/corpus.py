"""Loading a run back off disk.

`load_corpus` returns one DataFrame, one row per record, with `provider` as a
column. The side tables — citations, sources, queries, thoughts, units — hang off
it and carry the identity columns, so they group without a merge.

Two things here are easy to get wrong and were, in the modules this replaces:

**Side tables do not live in `corpus.attrs`.** That is the obvious place and it
fails: `attrs` does not survive `concat`, filtering, or most DataFrame
operations, so `sources(corpus[corpus.rep == 0])` raised. They live in a
module-level dict keyed by `investigation/run`, with only the KEY in `attrs` —
which survives because it is read, never propagated.

**Absence and unavailability are different, and a side table must say which.**
`citations(corpus)` returning no rows for a provider can mean it cited nothing or
that it cannot cite. The capability table decides, and `capability_note` on the
returned frame says so rather than leaving a caller to infer it from an empty
result.
"""

from __future__ import annotations

import json
import pathlib
from urllib.parse import urlparse

import pandas as pd

from . import paths
from .capabilities import caps_for
from .providers import get_provider

__all__ = ["load_corpus", "load_record", "citations", "sources", "queries",
           "thoughts", "units", "capability_note"]

# Bounded so a session loading many corpora does not accumulate indefinitely.
# **A soft guard, not a tuned figure.** Eight is more corpora than anyone has open
IDENTITY = ["record_id", "investigation", "provider", "model", "study", "probe",
            "path", "rep", "turn", "conversation", "condition"]


def load_record(path) -> dict:
    """One record file, raw and unparsed — including the full response body.

    The escape hatch for a question the side tables cannot answer. Everything
    else in this module reads through a provider parser; this does not.
    """
    return json.loads(pathlib.Path(path).read_text())


def load_corpus(investigation: str, run: str | None = None,
                provider: str | None = None, base=None) -> pd.DataFrame:
    """Load a run into a DataFrame. Defaults to the most recent run.

    Defaulting is a convenience with a real hazard — an analysis silently reading
    a different run than intended — so the run that was loaded is always printed.

    `provider` filters to one arm. The whole run is one directory with per-provider
    subdirectories, so a partial arm is visibly its own incomplete set rather than
    holes in a shared table.
    """
    # `paths.RECORDS_DIR` read at call time, not imported as a name — set_base
    # rebinds it, and an imported name would keep the value it had at load.
    records_dir = pathlib.Path(base) if base else paths.RECORDS_DIR
    # These three errors LIST THE ALTERNATIVES, which is the standard the rest of
    # this package holds itself to — `caps_for` names the known models, the
    # analysis filters name the available columns. An error that says only what
    # is absent leaves the caller to guess, and a typo'd investigation name is
    # the commonest way to arrive here.
    inv_dir = records_dir / investigation
    if not inv_dir.exists():
        available = sorted(p.name for p in records_dir.iterdir()
                           if p.is_dir()) if records_dir.exists() else []
        raise FileNotFoundError(
            f"no investigation {investigation!r} in {records_dir}\n"
            f"  Available: {available or '(none — nothing has been run yet)'}")

    runs = [p for p in inv_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(
            f"{investigation!r} exists but has no runs under {inv_dir}. "
            f"The spec was saved; the battery was never run.")

    if run and not (inv_dir / run).exists():
        raise FileNotFoundError(
            f"no run {run!r} under {investigation!r}\n"
            f"  Available: {sorted(p.name for p in runs)}")
    run_dir = inv_dir / run if run else max(runs, key=lambda p: p.name)

    files = sorted(run_dir.glob("[0-9]*.json"))
    if not files:
        raise FileNotFoundError(
            f"no records in {run_dir}. The directory exists, so the run started "
            f"and wrote nothing — check whether it was interrupted on the first "
            f"call, or whether persist=False was set.")

    rows: list[dict] = []
    cites: list[dict] = []
    srcs: list[dict] = []
    qs: list[dict] = []
    ths: list[dict] = []
    uns: list[dict] = []

    for rec_id, path in enumerate(files):
        d = load_record(path)
        if provider and d.get("provider") != provider:
            continue
        row = {k: v for k, v in d.items()
               if k not in ("config", "response")}
        row["record_id"] = rec_id
        row["file"] = str(path)

        response = d.get("response") or {}
        prov = d.get("provider")
        if prov and response:
            # Side tables are re-derived from the raw response rather than read
            # from whatever the runner wrote. If the two ever disagree, trusting
            # the stored value silently propagates the stale one — and a parser
            # improving after a corpus was collected is the normal case, not an
            # edge one.
            parsed = get_provider(prov).parse(response)
            ident = {k: row.get(k) for k in IDENTITY if k != "record_id"}
            ident["record_id"] = rec_id
            for name, target in (("citations", cites), ("sources", srcs),
                                 ("queries", qs), ("thoughts", ths),
                                 ("units", uns)):
                for entry in getattr(parsed, name):
                    target.append({**entry, **ident})
        rows.append(row)

    corpus = pd.DataFrame(rows)
    _coerce(corpus)

    # Side tables go straight into `attrs`.
    #
    # An earlier version kept them in a module-level LRU cache and stored only a
    # KEY here, on the stated grounds that "the tables themselves would be lost
    # by the first filter". That is false, and the comment contradicted itself —
    # the key and the tables live in the same dict, so either both survive or
    # neither does. Measured: `attrs` carries DataFrames through boolean
    # filtering, column selection, head, copy, sort_values, groupby and concat.
    #
    # Removing the cache removed the LRU eviction, the key indirection, and a
    # "reload with load_corpus()" error path — all of them guarding against
    # something that does not happen.
    corpus.attrs["side_tables"] = {
        "citations": _frame(cites), "sources": _frame(srcs),
        "queries": _frame(qs), "thoughts": _frame(ths), "units": _frame(uns),
    }
    corpus.attrs["models"] = (sorted(corpus.model.dropna().unique())
                              if len(corpus) else [])
    corpus.attrs["investigation"] = investigation
    corpus.attrs["run"] = run_dir.name
    corpus.attrs["run_dir"] = str(run_dir)

    _describe(corpus, investigation, run_dir)
    return corpus


def _domains(df: pd.DataFrame) -> list:
    """Real domain per row: the title where the URL is a redirect, else the netloc.

    One provider's search URLs are redirects whose netloc is always the same
    host, so parsing them would give one meaningless value for every source —
    its `title` carries the real domain instead.

    **`is_redirect` must be present and boolean.** It was absent from two
    providers' `sources` rows, so `df.get(col, default)` returned a column of
    NaN — and NaN is TRUTHY, so every row took the redirect branch and `domain`
    silently held page titles on providers whose URLs are perfectly ordinary.
    The bug survived a full live run and was only caught by reading the output.

    So this raises on a missing column rather than defaulting. A default that
    produces the wrong branch is worse than one that produces an error.
    """
    if "is_redirect" not in df.columns:
        raise KeyError(
            "`is_redirect` is missing from this side table. Every provider must "
            "declare it — a default would silently route rows down the redirect "
            "branch and fill `domain` with page titles.")
    out = []
    for title, url, redirect in zip(df["title"], df["url"], df["is_redirect"]):
        if bool(redirect):
            out.append(title)
        elif url:
            out.append(urlparse(url).netloc.replace("www.", "") or None)
        else:
            out.append(title)
    return out


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df):
        cols = [c for c in IDENTITY if c in df.columns]
        rest = [c for c in df.columns if c not in cols]
        df = df[cols + rest]
    return df


def _coerce(corpus: pd.DataFrame) -> None:
    """Types, and NOT fills.

    `turn`, `rep` and `attempts` are structural and always written by the runner,
    so a missing one is corruption rather than a case to handle — they are cast,
    never defaulted.

    **Tier 3 columns are never filled.** A fill is a claim: `fillna(0)` on a token
    sum says a failed record contributed nothing, which is true; `fillna(False)`
    on `grounded` says a provider declined to do something it cannot do, which is
    not. The predecessor loader did exactly that, harmlessly within one provider
    and wrongly the moment a corpus held more than one.
    """
    if not len(corpus):
        return
    for col in ("turn", "rep", "path", "attempts", "n_turns"):
        if col in corpus:
            if corpus[col].isna().any():
                raise ValueError(
                    f"{col!r} is null on {int(corpus[col].isna().sum())} record(s). "
                    f"It is written by the runner on every record, so a null means "
                    f"the file is corrupt rather than that the value is unknown.")
            corpus[col] = corpus[col].astype(int)


def _describe(corpus: pd.DataFrame, investigation: str, run_dir) -> None:
    print(f"{investigation} / {run_dir.name}")
    if not len(corpus):
        print("  (no records)")
        return

    # `truncated` is a SUCCESS — the answer is real and cut off, which is the
    # whole reason the fifth state exists. Counting it as "not ok" made a study
    # that worked perfectly read as total failure on the first live run: four
    # truncated records with 582–1,169 characters of usable text, reported as
    # `ok: 0`. A headline that misrepresents the data is worse than no headline.
    usable = corpus[corpus.status.isin(["ok", "truncated"])]
    failed = corpus[~corpus.status.isin(["ok", "truncated"])]
    n_trunc = int((corpus.status == "truncated").sum())
    print(f"  {len(corpus)} records"
          + (f", {len(usable)} with answers" if n_trunc else "")
          + (f" ({n_trunc} truncated)" if n_trunc else "")
          + (f", {len(failed)} failed" if len(failed) else "")
          + f" · {corpus.probe.nunique()} probes · {corpus.condition.nunique()} conditions")

    # Per provider, never one total — three token conventions make a combined
    # figure a false comparison.
    for prov, g in corpus.groupby("provider"):
        note = []
        if "n_queries" in g and g.n_queries.notna().any():
            note.append(f"{int(g.n_queries.fillna(0).sum())} queries")
        if "n_citations" in g and g.n_citations.fillna(0).sum():
            note.append(f"{int(g.n_citations.fillna(0).sum())} citations")
        if "grounded" in g and g.grounded.notna().any():
            # Count only where the column MEANS something. `fillna(False)` here
            # would read as "these records declined to ground" when they mean
            # "this provider has no such concept" — and a summary line is exactly
            # where that misreading would go unchallenged, because nobody audits
            # a print statement.
            known = g.grounded.dropna()
            note.append(f"{int(known.sum())}/{len(known)} grounded")
        retried = int((g.attempts > 1).sum()) if "attempts" in g else 0
        if retried:
            note.append(f"{retried} retried")
        print(f"    {prov:<12} {len(g):>4} records"
              + ("   " + " · ".join(note) if note else ""))

    statuses = corpus.status.value_counts().to_dict()
    if set(statuses) - {"ok"}:
        print(f"  statuses: {statuses}")
    # Loud by design. Recording an issue and never surfacing it is the silent
    # failure the integrity checks exist to prevent — a count that nobody sees
    # is the same as no count.
    from .integrity import describe_integrity
    report = describe_integrity(corpus)
    if report:
        print(report)


# ═══════════════════════════════════════════════════════════════════════════════
# Side tables
# ═══════════════════════════════════════════════════════════════════════════════

def _side(corpus: pd.DataFrame, name: str) -> pd.DataFrame:
    """A side table, FILTERED to the records still in `corpus`.

    The filtering is the point. The cache is keyed by investigation/run and the
    key survives a filter, so `citations(corpus[corpus.provider == "gemini"])`
    would otherwise return every provider's citations against a one-provider
    frame — silent wrong data, and the kind that looks plausible because the rows
    are all real.

    A record_id present in the cache and absent from the frame has been filtered
    out deliberately; the side table follows.
    """
    tables = corpus.attrs.get("side_tables")
    if tables is None:
        raise KeyError(
            "side tables are not available for this frame — it was built rather "
            "than loaded. Reload with load_corpus(); filtering a loaded corpus "
            "is fine, constructing a DataFrame by hand is not.")
    df = tables[name]
    if not len(df) or "record_id" not in corpus.columns:
        return df
    return df[df.record_id.isin(set(corpus.record_id))]


def capability_note(corpus: pd.DataFrame, capability: str) -> dict[str, bool]:
    """Which providers in this corpus CAN do something.

    The thing a caller needs to read an empty side table correctly. `citations()`
    returning no rows for a provider means it cited nothing OR that it cannot
    cite, and those are different findings — one is about the model, the other
    about the API.
    """
    models = corpus.attrs.get("models", [])
    return {m: bool(getattr(caps_for(m), capability)) for m in models}


def citations(corpus: pd.DataFrame, record_id: int | None = None) -> pd.DataFrame:
    """One row per citation, across every provider.

    **Compare quotes, not offsets.** Three mechanisms exist — quoted text,
    character offsets, byte offsets — and `offset_unit` records which produced
    each row. The offsets are provenance; `quote` is the comparable object and is
    populated for all three, extracted where only offsets exist.

    A loader applying the wrong convention produces spans that are silently wrong
    and drift further into the answer: measured wrong on 70 of 107 citations on
    the byte-offset provider.
    """
    df = _side(corpus, "citations")
    if len(df) and "domain" not in df.columns:
        # `title` carries the real domain on the provider whose URLs are
        # redirects — its netloc is always Google's, so parsing the URL would
        # give one meaningless value for every source.
        df = df.copy()
        df["domain"] = _domains(df)
    return df if record_id is None else df[df.record_id == record_id]


def sources(corpus: pd.DataFrame, record_id: int | None = None,
            retrieved_only: bool = False) -> pd.DataFrame:
    """One row per URL, with `cited` and `retrieved` as SEPARATE booleans.

    Only one provider exposes the retrieval set — every URL it saw, not only
    those it cited — so `retrieved` is None elsewhere rather than False. "This
    provider does not report what it saw" is a different claim from "it saw
    nothing", and conflating them makes the quieter providers look more selective.

    That distinction is what makes "what separates a cited source from an uncited
    one" answerable at all, and it is answerable on one provider.
    """
    df = _side(corpus, "sources")
    if len(df) and "domain" not in df.columns:
        df = df.copy()
        df["domain"] = _domains(df)
    if retrieved_only and len(df):
        df = df[df.retrieved == True]  # noqa: E712 — None must not match
    return df if record_id is None else df[df.record_id == record_id]


def queries(corpus: pd.DataFrame, record_id: int | None = None) -> pd.DataFrame:
    """One row per search query issued, with the call that produced it.

    Every provider links a result to its call, so "which query produced this
    source" is answerable everywhere rather than only where the linkage was first
    noticed.

    Note this is the QUERY list, not the search COUNT — those differ on two
    providers in opposite directions, and `n_queries` on the corpus is the
    authoritative count.
    """
    df = _side(corpus, "queries")
    return df if record_id is None else df[df.record_id == record_id]


def thoughts(corpus: pd.DataFrame, record_id: int | None = None) -> pd.DataFrame:
    """One row per readable reasoning step.

    Two of three providers now populate this; it was one when the column was
    designed. An empty frame for a provider means either that it cannot expose
    reasoning or that the request did not ask — `capability_note(corpus,
    "readable_reasoning")` distinguishes them, and one provider requires the
    summary to be requested per call.
    """
    df = _side(corpus, "thoughts")
    return df if record_id is None else df[df.record_id == record_id]


def units(corpus: pd.DataFrame, record_id: int | None = None) -> pd.DataFrame:
    """Provider-native response elements: blocks, items, or steps.

    Named neutrally on purpose. Three names for one concept would be the naming
    rule — a column that means different things must not share a name — violated
    inside our own API, and `type` already carries the provider-specific shape.
    """
    df = _side(corpus, "units")
    return df if record_id is None else df[df.record_id == record_id]
