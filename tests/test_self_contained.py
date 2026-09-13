"""The repo must be self-contained, and every reference in it must resolve.

**Written broad-to-narrow, deliberately.** Six earlier audit passes all started
from a suspicion and grepped for it — precise, and they kept missing things
nobody thought to look for. Three separate defects were found only by someone
reading output and asking why, not by any check.

The specific failure: `capabilities.py` told users to read `MERGE_SPEC Appendix
J`, a document that has never been in this repo. It was a broken pointer from the
day it was written, and it REPLACED an earlier broken pointer to
`tests/characterise.py` — one dangling reference swapped for another, recorded as
a fix. A test comparing against that document also skipped in every checkout for
weeks, silently, because it looked one level above the repo root.

None of that was findable by grepping for what I suspected. It required
enumerating EVERY reference of each class and checking all of them.

So these tests enumerate first and filter second. That ordering matters: an
earlier version of this scan excluded `__pycache__`, `.git`, `build` and
`.egg-info` from a list it wrote from memory, and missed `.pytest_cache` — the
same narrowness, in the tool built to catch it.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

def _source_files() -> list[pathlib.Path]:
    """The files git tracks — ASK, do not enumerate.

    This kept a hand-written list of artifact directories and it was wrong three
    times: it missed `.pytest_cache`, then `.ruff_cache`, each time because the
    list was written from memory rather than derived. The tool built to catch
    narrow searching kept searching narrowly.

    `git ls-files` is the authoritative answer to "what is in this repo", it
    already knows about `.gitignore`, and it cannot go stale when a new tool
    invents a new cache directory.

    Falls back to a walk when git is unavailable — a source tarball has no `.git`
    — and the fallback is the only place a list is needed.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO, check=True,
                             capture_output=True, text=True, timeout=30).stdout
        tracked = [REPO / line for line in out.split("\n") if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    ignored = {"__pycache__", ".git", "build", ".egg-info"}
    return [p for p in REPO.rglob("*")
            if p.is_file()
            and not any(part in ignored or part.startswith(".")
                        or part.endswith(".egg-info") for part in p.parts)]


# ═══════════════════════════════════════════════════════════════════════════════
# BROAD — every reference of every class
# ═══════════════════════════════════════════════════════════════════════════════

# Files the code CREATES at runtime. Not references to something expected to
# exist, so not dangling — but listed explicitly, because a blanket "ignore
# anything that looks generated" would hide a real miss.
RUNTIME_ARTIFACTS = {
    "_corpus.json", "_run.json", "_spec.json",
    "roster.json", "current.json", "baseline.json", "absent.json",
    "direct_url.json", "pyproject.toml", "setup.py",
}

# A repo must be able to DISCUSS its own history — "X was removed", "Y named a
# path that never existed" — without those sentences reading as live pointers.
#
# The rule is general rather than a list of names: a reference is historical if
# its line says so. An earlier version kept a hardcoded set, and this test then
# failed on its OWN docstrings, which name the very references they explain. A
# list would have grown by two; the rule covers the class.
# Two words, not a phrase list. Enumerating English phrasings is the same
# narrowness this test exists to catch — an earlier version had "never existed"
# and "did not exist" and then failed on "was never in here". A past-tense or
# negating word anywhere near the reference is enough, because the cost of a
# false negative here is one missed prose mention and the cost of a false
# positive is a test nobody can satisfy.
HISTORICAL_MARKERS = re.compile(
    r"\b(?:never|not|no longer|was|were|used to|previously|removed|deprecated|"
    r"historical|outlived|stale|phantom|superseded|replaced)\b", re.IGNORECASE)


def test_every_file_reference_resolves_inside_the_repo():
    """No pointer may leave the repo.

    `MERGE_SPEC.md` was never in here, and three code comments plus a runtime
    error message named it. Anyone hitting `caps_for` on an unknown model was
    told to read a document they do not have and cannot get.
    """
    names = {p.name for p in _source_files()}
    relative = {str(p.relative_to(REPO)) for p in _source_files()}

    pattern = re.compile(
        r"\b([\w./-]+\.(?:py|md|json|toml|txt|yml|yaml|cfg|ipynb))\b")
    dangling: dict[str, list[str]] = {}

    for f in _source_files():
        if f.suffix == ".json":          # fixtures are data, not prose
            continue
        lines = f.read_text(errors="replace").split("\n")
        for i, line in enumerate(lines, 1):
            # A SENTENCE, not a line. Prose wraps, and "named `x.py`, a path that
            # never existed" puts the marker on the following line — which an
            # earlier version of this check read as a live pointer, failing on
            # its own docstrings.
            context = " ".join(lines[max(0, i - 6):i + 4])
            for m in pattern.finditer(line):
                ref = m.group(1)
                base = ref.split("/")[-1]
                if base in RUNTIME_ARTIFACTS:
                    continue
                if HISTORICAL_MARKERS.search(context):
                    continue
                if base in names or ref in relative:
                    continue
                if any(r.endswith(ref) for r in relative):
                    continue
                dangling.setdefault(ref, []).append(
                    f"{f.relative_to(REPO)}:{i}")

    assert not dangling, (
        "references that do not resolve inside the repo:\n  "
        + "\n  ".join(f"{k} — {v[:3]}" for k, v in sorted(dangling.items())))


def test_every_import_resolves():
    modules = set()
    for f in _source_files():
        if f.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    missing = [m for m in sorted(modules) if importlib.util.find_spec(m) is None]
    assert not missing, f"unresolvable imports: {missing}"


def test_every_command_in_the_docs_names_a_file_that_exists():
    """A README telling you to run something that is not there is worse than one
    that says nothing — you follow it before you doubt it."""
    bad = []
    for doc in (p for p in _source_files() if p.suffix == ".md"):
        for i, line in enumerate(doc.read_text().split("\n"), 1):
            m = re.match(r"^\s*!?python3?\s+([\w/]+\.py)", line.strip())
            if m and not (REPO / m.group(1)).exists():
                bad.append(f"{doc.relative_to(REPO)}:{i}  {line.strip()[:60]}")
    assert not bad, "commands naming missing files:\n  " + "\n  ".join(bad)


def test_no_test_silently_skips_on_a_missing_external_file():
    """A test that skips when something outside the repo is absent skips ALWAYS,
    in every checkout, and reports itself as a skip rather than a gap.

    One did, for weeks, looking one level above the repo root.
    """
    offenders = []
    for f in (p for p in _source_files() if p.name.startswith("test_")):
        source = f.read_text()
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "skip"):
                continue
            window = source.split("\n")[max(0, node.lineno - 6):node.lineno]
            if any("parents[" in w and ".parent" in w for w in window):
                offenders.append(f"{f.relative_to(REPO)}:{node.lineno}")
    assert not offenders, (
        f"tests skipping on a path outside the repo: {offenders}")


# ═══════════════════════════════════════════════════════════════════════════════
# NARROWER — claims the repo makes about itself
# ═══════════════════════════════════════════════════════════════════════════════

def test_error_messages_name_only_things_that_exist():
    """`caps_for` named `tests/characterise.py`, which did not exist. The fix
    pointed it at `MERGE_SPEC Appendix J`, which also did not exist here.

    One dangling pointer replaced by another, and recorded as a fix. This is the
    check that would have caught either.
    """
    names = {p.name for p in _source_files()}
    relative = {str(p.relative_to(REPO)) for p in _source_files()}
    bad = []
    for f in (p for p in _source_files() if p.suffix == ".py"):
        source = f.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Raise) or not node.exc:
                continue
            text = ast.unparse(node.exc)
            for m in re.finditer(r"([\w/]+\.(?:py|md))", text):
                ref = m.group(1)
                if ref.split("/")[-1] in names or ref in relative:
                    continue
                bad.append(f"{f.relative_to(REPO)}:{node.lineno} names {ref!r}")
    assert not bad, "error messages naming missing files:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("doc,claim", [
    ("README.md", "drift/"),
    ("SETUP.md", "drift"),
])
def test_the_docs_describe_the_directories_that_exist(doc, claim):
    assert claim in (REPO / doc).read_text()


def test_no_doc_claims_something_is_unbuilt_when_it_is_built():
    """`_REFRESH.md` said the refresh tool was "planned, not built" and named
    `tests/refresh_fixtures.py` — a path that never existed, describing something
    that DID exist as `drift/collect_fixtures.py`.

    Stale in both directions at once.
    """
    text = (REPO / "tests" / "fixtures" / "_REFRESH.md").read_text()
    built = "drift/collect_fixtures.py"
    assert built in text, "the refresh doc does not name the tool that exists"
    assert (REPO / built).exists()


def test_no_personal_paths_or_names_in_the_repo():
    """The repo must be usable by someone who is not its author.

    `paths.BASE` defaulted to one person's Google Drive folder until 2026-09-09.
    Anyone else importing the package wrote to a path that did not exist for them
    and failed on the first RECORD WRITE rather than at import — which is a
    confusing place to discover a configuration problem.
    """
    import re

    OFFENDERS = {
        "a personal Drive path": r"MyDrive",
        "a named project folder": r"Adaptation Systems|AEO GEO Research",
        "an absolute home path": r"/Users/|/home/[a-z]",
    }
    bad = []
    for f in _source_files():
        if f.suffix == ".json" or f.name.startswith("test_"):
            continue          # fixtures hold real recorded data; tests may assert on it
        text = f.read_text(errors="replace")
        for i, line in enumerate(text.split("\n"), 1):
            for name, pat in OFFENDERS.items():
                if re.search(pat, line):
                    bad.append(f"{f.relative_to(REPO)}:{i} — {name}")
    assert not bad, "paths specific to one person:\n  " + "\n  ".join(bad)


def test_the_base_defaults_to_something_neutral():
    """And can be set from the environment without a code change."""
    from machine_psych import paths

    assert "MyDrive" not in str(paths.BASE)
    source = pathlib.Path(paths.__file__).read_text()
    assert "MACHINE_PSYCH_BASE" in source, (
        "no environment override — the only way to set the base is a code change")


def test_setup_documents_keys_and_output_location():
    """Installing gets you the package. Neither keys nor the output path has a
    default that works for someone else, so both must be documented."""
    setup = (REPO / "SETUP.md").read_text()
    assert "set_api_key" in setup, "SETUP never mentions API keys"
    assert "set_base" in setup, "SETUP never says where output goes"
    assert "wiped on disconnect" in setup or "wiped" in setup, (
        "SETUP does not warn that the Colab default is ephemeral")
