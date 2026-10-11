"""Which code produced which result.

Every notebook in this project loads the harness from a git CLONE on `sys.path`.
Provenance looked only at pip's install metadata, which a clone does not have — so
every run recorded `commit: None` while the exact commit sat in the clone's `.git`
folder. The methods paper needs to say which code produced which corpus.
"""

from __future__ import annotations

import pytest

import machine_psych as mp
from machine_psych import paths

SHA = "c9636279f980a1b2c3d4e5f60718293a4b5c6d7e"


def _git(tmp_path, head, refs=None, packed=None):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n", encoding="utf-8")
    for name, sha in (refs or {}).items():
        (git / "refs" / "heads" / name).write_text(sha + "\n", encoding="utf-8")
    if packed:
        (git / "packed-refs").write_text(packed, encoding="utf-8")
    return tmp_path


def test_a_branch_checkout_yields_its_commit(tmp_path):
    root = _git(tmp_path, "ref: refs/heads/main", refs={"main": SHA})
    assert paths._git_checkout_commit(root) == SHA


def test_a_pinned_checkout_yields_its_commit(tmp_path):
    """`git checkout <sha>` — how a notebook pins — leaves a DETACHED head that
    holds the hash itself rather than naming a branch."""
    root = _git(tmp_path, SHA)
    assert paths._git_checkout_commit(root) == SHA


def test_packed_refs_are_read(tmp_path):
    """`git gc` moves branch refs into one `packed-refs` file and deletes the loose
    ones; a reader that only checked the loose file would lose the commit."""
    root = _git(tmp_path, "ref: refs/heads/main",
                packed=f"# pack-refs with: peeled\n{SHA} refs/heads/main\n")
    assert paths._git_checkout_commit(root) == SHA


def test_a_worktree_gitdir_file_is_followed(tmp_path):
    real = _git(tmp_path / "real", "ref: refs/heads/main", refs={"main": SHA})
    work = tmp_path / "work"
    work.mkdir()
    (work / ".git").write_text(f"gitdir: {real / '.git'}\n", encoding="utf-8")
    assert paths._git_checkout_commit(work) == SHA


def test_no_checkout_means_unknown_not_an_error(tmp_path):
    assert paths._git_checkout_commit(tmp_path) is None


def test_provenance_is_captured_at_import_not_at_call(monkeypatch):
    """**Re-running setup without a restart** re-clones the files but keeps the OLD
    code in memory. Reading the disk at run time would record a commit that did
    not run. So detection runs once, at import, and calls return that capture."""
    def boom():
        raise AssertionError("provenance was re-detected at call time")
    monkeypatch.setattr(paths, "_detect_provenance", boom)
    assert paths.provenance() == paths._PROVENANCE
    assert paths.provenance() is not paths._PROVENANCE, "callers must get a copy"


def test_require_commit_accepts_the_running_commit(monkeypatch):
    monkeypatch.setattr(paths, "_PROVENANCE", {**paths._PROVENANCE, "commit": SHA, "dirty": False})
    assert mp.require_commit(SHA[:10]) == SHA


def test_require_commit_stops_a_mismatch_and_says_to_restart(monkeypatch):
    monkeypatch.setattr(paths, "_PROVENANCE", {**paths._PROVENANCE, "commit": SHA, "dirty": False})
    with pytest.raises(RuntimeError, match="restart the runtime"):
        mp.require_commit("0000000deadbeef")


def test_require_commit_stops_when_the_commit_is_unknown(monkeypatch):
    """A run that cannot say which code produced it should not start silently."""
    monkeypatch.setattr(paths, "_PROVENANCE", {**paths._PROVENANCE, "commit": None})
    with pytest.raises(RuntimeError, match="unknown"):
        mp.require_commit(None)


def test_require_commit_refuses_an_ambiguous_prefix(monkeypatch):
    monkeypatch.setattr(paths, "_PROVENANCE", {**paths._PROVENANCE, "commit": SHA})
    with pytest.raises(ValueError, match="too short"):
        mp.require_commit("c96")


# ═══════════════════════════════════════════════════════════════════════════════
# data_quality — one table every notebook saves the same way
# ═══════════════════════════════════════════════════════════════════════════════

def _frame():
    import pandas as pd
    return pd.DataFrame([
        {"record_id": "a", "model": "gemini/gemini-3.8-flash", "probe": "p1", "status": "ok",
         "integrity": [{"severity": "warning", "code": "budget_exhausted", "detail": "used 64,000"}]},
        {"record_id": "b", "model": "anthropic/claude-opus-5-5", "probe": "p2", "status": "ok",
         "integrity": [{"severity": "info", "code": "answer_split_heuristically", "detail": "x"}]},
        {"record_id": "c", "model": "openai/gpt-5.6-sol", "probe": "p3", "status": "truncated",
         "error": None, "integrity": None},
        {"record_id": "d", "model": "openai/gpt-5.6-sol", "probe": "p4", "status": "ok",
         "integrity": None},
    ])


def test_data_quality_lists_both_new_flags_and_bad_statuses():
    dq = mp.data_quality(_frame())
    assert set(dq.flag) == {"budget_exhausted", "answer_split_heuristically", "status:truncated"}
    assert "d" not in set(dq.record_id), "a clean record was listed"


def test_data_quality_gives_the_retry_list_with_identity():
    """The budget-exhausted retry list comes straight from the table, with enough
    identity to rebuild the calls."""
    retry = mp.data_quality(_frame()).query("flag == 'budget_exhausted'")
    assert retry.record_id.tolist() == ["a"] and retry.probe.tolist() == ["p1"]


def test_data_quality_summary_counts_by_model():
    s = mp.data_quality(_frame(), summary=True)
    assert {"model", "flag", "severity", "records"} <= set(s.columns)
    assert s.records.sum() == 3


def test_a_clean_corpus_gives_an_empty_frame_with_columns():
    clean = _frame().iloc[[3]]
    dq = mp.data_quality(clean)
    assert dq.empty and {"flag", "severity", "detail"} <= set(dq.columns)
