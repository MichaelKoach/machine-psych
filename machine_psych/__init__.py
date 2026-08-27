"""machine-psych — qualitative behavioural research on language models.

The model is the research subject. The question is how it sorts a category, what
it recommends, what it associates with a brand, and what moves any of that.

Installed from git and pinned by commit, so a corpus can record verifiable
provenance rather than a version string someone typed.
"""

__version__ = "0.1.0"

from importlib import metadata

DIST = "machine-psych"


def provenance():
    """What is actually installed, for the corpus to record.

    Returns the distribution version and, when installed from a git URL, the
    commit it came from. A corpus that stores this can be traced to the exact
    parsing code that produced it — which matters because parsers have bugs, and
    those bugs get fixed after corpora are already on disk.
    """
    out = {"version": __version__, "commit": None, "source": None}
    try:
        direct = metadata.distribution(DIST).read_text("direct_url.json")
        if direct:
            import json
            d = json.loads(direct)
            out["source"] = d.get("url")
            out["commit"] = (d.get("vcs_info") or {}).get("commit_id")
    except Exception:
        pass
    return out


def install_report():
    """One line, for the top of a notebook."""
    p = provenance()
    commit = (p["commit"] or "unknown")[:8]
    print(f"{DIST} {p['version']}  ·  commit {commit}")
    if p["commit"] is None:
        print("  NOT installed from git — provenance cannot be recorded.")
        print(f"  Use: pip install git+https://github.com/MichaelKoach/{DIST}.git@SHA")
    return p
