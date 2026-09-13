"""Where things live, and what version produced them.

A layer-0 module: it imports nothing from the package, so everything else can
import it without a cycle.

It exists because of one. `runner.py` had `from . import provenance` at module
level, and `corpus.py` reached back into `runner` for the records directory. Both
worked, and the first worked **by accident of ordering** — `provenance()` happened
to be defined in `__init__.py` above the submodule imports, so moving that
function thirty lines down would have stopped the package importing at all.

Configuration state and provenance are the two things every layer needs and
neither belongs to any of them, so they live here.
"""

from __future__ import annotations

import os
import pathlib
from importlib import metadata

__all__ = [
    "BASE",
    "DIST",
    "INVESTIGATIONS_DIR",
    "RECORDS_DIR",
    "VERSION",
    "api_key",
    "install_report",
    "provenance",
    "set_api_key",
    "set_base",
]

DIST = "machine-psych"


def _installed_version() -> str:
    """The installed version, or the source one when running from a checkout."""
    try:
        return metadata.version(DIST)
    except Exception:
        return "0.1.0"


VERSION = _installed_version()

# A NEUTRAL default. This was one person's Google Drive folder until 2026-09-09,
# which meant anyone else importing the package wrote to a path that did not exist
# for them — and failed on the first record write rather than at import, which is
# a confusing place to discover it.
#
# `MACHINE_PSYCH_BASE` lets an environment set it without a code change. Falling
# back to the working directory means the package is usable immediately from a
# checkout, and `set_base()` is how anyone points it somewhere durable.
#
# In Colab the fallback is EPHEMERAL — /content vanishes on disconnect — so
# `set_base()` to a Drive path is effectively required there. `install_report()`
# prints where it is currently pointed.
BASE = pathlib.Path(os.environ.get("MACHINE_PSYCH_BASE", pathlib.Path.cwd()))
INVESTIGATIONS_DIR = BASE / "Investigations"
RECORDS_DIR = BASE / "Output Log"

_API_KEYS: dict[str, str] = {}


def set_base(path) -> pathlib.Path:
    """Point the package at a different root.

    Rebinds the module globals rather than returning a config object, so callers
    that imported `RECORDS_DIR` directly would keep the OLD value — which is why
    everything downstream reads `paths.RECORDS_DIR` at call time rather than
    importing the name.
    """
    global BASE, INVESTIGATIONS_DIR, RECORDS_DIR
    BASE = pathlib.Path(path)
    INVESTIGATIONS_DIR = BASE / "Investigations"
    RECORDS_DIR = BASE / "Output Log"
    for directory in (INVESTIGATIONS_DIR, RECORDS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    return BASE


def set_api_key(provider: str, key: str) -> None:
    """Keys are held in memory only and never written to a corpus or a spec."""
    _API_KEYS[provider] = key


def api_key(provider: str) -> str | None:
    return _API_KEYS.get(provider)


def provenance() -> dict:
    """What is actually installed, for the corpus to record.

    Returns the distribution version and, when installed from a git URL, the
    commit it came from. A corpus that stores this can be traced to the exact
    parsing code that produced it — which matters because parsers have bugs, and
    those bugs get fixed after corpora are already on disk. Three corpora
    collected earlier in this project were produced by code that has since been
    corrected, and there is no way to reconstruct what the collecting version did.
    """
    out = {"version": VERSION, "commit": None, "source": None}
    try:
        direct = metadata.distribution(DIST).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        # Running from a checkout rather than an install. Not an error, and the
        # report says `commit unknown`, which is the honest answer.
        return out
    if not direct:
        return out
    try:
        import json
        info = json.loads(direct)
    except json.JSONDecodeError:
        # Malformed metadata is worth not crashing over, but it is also not
        # something to hide — a bare `except Exception` here swallowed every
        # possible failure including bugs in this function.
        return out
    out["source"] = info.get("url")
    out["commit"] = (info.get("vcs_info") or {}).get("commit_id")
    return out


def install_report() -> dict:
    """One line, for the top of a notebook."""
    p = provenance()
    commit = (p["commit"] or "unknown")[:8]
    print(f"{DIST} {p['version']}  ·  commit {commit}")
    if p["commit"] is None:
        print("  NOT installed from git — provenance cannot be recorded.")
        print(f"  Use: pip install git+https://github.com/MichaelKoach/{DIST}.git@SHA")
    return p
