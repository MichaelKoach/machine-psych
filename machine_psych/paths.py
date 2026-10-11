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
    "require_commit",
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


# The conventional names, so a machine that already has them set for another
# tool needs no extra configuration.
_ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY",
             "openai": "OPENAI_API_KEY",
             "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def api_key(provider: str) -> str | None:
    """An explicit `set_api_key` wins; otherwise the environment.

    A battery left running for days is started by a script, not by a notebook,
    and a key pasted into that script is a key that gets committed. Reading the
    environment means the script carries no secret and the same file works on any
    machine.

    In-memory first so a notebook can still override per session, which is how
    every run before this worked.
    """
    if provider in _API_KEYS:
        return _API_KEYS[provider]
    names = _ENV_KEYS.get(provider, ())
    for name in (names,) if isinstance(names, str) else names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _git_checkout_commit(root: pathlib.Path) -> str | None:
    """The commit a git checkout is at, read from its files — no `git` binary.

    Handles a branch (HEAD names a ref, stored loose or in `packed-refs`), a
    detached HEAD (HEAD holds the hash itself — what `git checkout <sha>` leaves
    when a notebook pins a commit), and a `.git` FILE pointing elsewhere, as in a
    worktree.
    """
    git = root / ".git"
    try:
        if git.is_file():
            target = git.read_text(encoding="utf-8").strip()
            if not target.startswith("gitdir:"):
                return None
            git = (root / target.split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None                    # detached: the hash itself
    ref = head.split(":", 1)[1].strip()
    loose = git / ref
    try:
        if loose.exists():
            return loose.read_text(encoding="utf-8").strip() or None
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return None


def _git_dirty(root: pathlib.Path) -> bool | None:
    """Whether the checkout has local edits — None if that cannot be asked."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Ignore files the harness itself does not ship: caches and build output
    # appear in any working copy and say nothing about the code that ran.
    changed = [line for line in out.stdout.splitlines()
               if line.strip() and not any(x in line for x in (
                   "__pycache__", ".egg-info", ".pytest_cache", ".ruff_cache"))]
    return bool(changed)


def _detect_provenance() -> dict:
    """Which code is running. Called ONCE, when the package is imported.

    **Captured at import, not at run time, on purpose.** A notebook that re-clones
    the repo without restarting keeps running the OLD code from memory while the
    NEW commit sits on disk. Reading the disk at run time would record a commit
    that did not run. Reading it at import records the one that did.

    Two sources, in order:
    1. pip metadata, when installed from a git URL (`pip install git+...@sha`).
    2. the git checkout the package was loaded from — a clone on `sys.path`, an
       editable install, a VM, a laptop. **This was missing.** Every notebook
       loads the harness from a clone, so every run recorded `commit: None` while
       the exact commit sat in the clone's `.git` folder.
    """
    out = {"version": VERSION, "commit": None, "source": None, "dirty": None}
    try:
        direct = metadata.distribution(DIST).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct = None
    if direct:
        import json
        try:
            info = json.loads(direct)
        except json.JSONDecodeError:
            info = {}
        commit = (info.get("vcs_info") or {}).get("commit_id")
        if commit:
            out.update(commit=commit, source=info.get("url"))
            return out

    root = pathlib.Path(__file__).resolve().parent.parent
    commit = _git_checkout_commit(root)
    if commit:
        out.update(commit=commit, source=f"git checkout at {root}",
                   dirty=_git_dirty(root))
    return out


_PROVENANCE = _detect_provenance()


def provenance() -> dict:
    """What code is running, for every run to record.

    Captured when the package was imported — see `_detect_provenance`. A corpus
    that stores this can be traced to the exact code that produced it, which
    matters because the harness changes: parsers get fixed and checks get added
    after corpora are already on disk.

    `dirty=True` means the checkout had local edits, so the commit alone does not
    fully describe the code that ran.
    """
    return dict(_PROVENANCE)


def require_commit(expected: str | None) -> str:
    """Stop unless the running harness is the commit expected. Returns it.

    For a notebook that pins its harness: after `git checkout <sha>` in setup,
    `mp.require_commit(HARNESS_COMMIT)` confirms that commit is the code actually
    LOADED — not just the one on disk. The commonest way they differ is
    re-running setup without restarting the runtime, which re-clones the files
    but keeps the old code in memory.

    `expected=None` checks only that the commit is known. A prefix of at least
    seven characters is accepted.
    """
    running = _PROVENANCE["commit"]
    if running is None:
        raise RuntimeError(
            "the running harness commit is unknown — it was not loaded from a git "
            "checkout or a git install, so a run cannot record which code produced it")
    if expected is not None:
        expected = expected.strip()
        if len(expected) < 7:
            raise ValueError(f"commit {expected!r} is too short to identify — use 7+ characters")
        if not running.startswith(expected):
            raise RuntimeError(
                f"the harness running is {running[:12]}, not {expected[:12]}. If you "
                f"just checked out {expected[:12]}, restart the runtime: the old code "
                f"is still loaded in memory")
    if _PROVENANCE.get("dirty"):
        print(f"  note: harness {running[:12]} has local edits — the commit alone "
              f"does not describe the code that ran")
    return running


def install_report() -> dict:
    """One line, for the top of a notebook."""
    p = provenance()
    commit = (p["commit"] or "unknown")[:8]
    print(f"{DIST} {p['version']}  ·  commit {commit}")
    if p["commit"] is None:
        print("  commit unknown — not loaded from a git checkout or a git install,")
        print("  so runs cannot record which code produced them.")
    elif p.get("dirty"):
        print("  with local edits — the commit alone does not describe the code")
    return p
