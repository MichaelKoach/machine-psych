"""Loading an investigation and running it.

Two functions. `load_investigation` turns a spec into the exact list of calls it
will make and SENDS NOTHING, so a design mistake costs a read rather than a
battery. `run_investigation` executes that list.

The runner is the highest-difficulty code here, and **the order of operations is
the specification**. Sixteen points in the sequence are places where getting it
wrong produces a bug this project has already had — each is marked in the code
with what it breaks. They are not stylistic.

Two decisions shape the whole thing:

**Execution is round-robin over repetitions and interleaved across providers**, so
retrieval drift spreads across conditions rather than confounding with them. The
web changes during a run; a battery that finishes all of condition A before
starting condition B has made time a hidden factor.

**A failed turn ends its conversation, never the run.** One provider's outage must
not cost the other two their arms, and a conversation that broke at turn 1 still
has a usable turn 0.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from datetime import datetime, timezone

import pandas as pd

from . import paths
from .capabilities import caps_for, provider_of
from .integrity import check_record

# Re-exported so `runner.set_base(...)` keeps working for callers, while the
# STATE lives in `paths`. A facade over one source of truth, not a copy of it —
# an earlier version held the paths here and `corpus` reached back into this
# module to read them, which made a low-level module depend on a high-level one.
from .paths import set_api_key, set_base
from .providers import get_provider
from .spec import (
    InvestigationError,
    expand_conditions,
    join_text,
    probe_hash,
    prompt_hash,
    resolve,
    validate_investigation,
)

__all__ = [
    "estimate",
    "list_investigations",
    "list_runs",
    "load_investigation",
    "run_investigation",
    "save_investigation",
    "set_api_key",
    "set_base",
]

# Paths and keys live in `paths` — a layer-0 module — and are read AT CALL TIME
# rather than imported as names. `set_base` rebinds module globals, so a name
# imported at module load would keep the old value forever.


# ═══════════════════════════════════════════════════════════════════════════════
# Files
# ═══════════════════════════════════════════════════════════════════════════════

def save_investigation(spec: dict, name: str | None = None,
                       overwrite: bool = False) -> pathlib.Path:
    """Validate and write a spec. Refuses to overwrite without being told to.

    Validating on write rather than only on load means a malformed spec never
    reaches disk, where it would look like a working investigation until someone
    tried to run it.
    """
    validate_investigation(spec)
    paths.INVESTIGATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = paths.INVESTIGATIONS_DIR / f"{name or spec['investigation_id']}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists — pass overwrite=True")
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"wrote {path}")
    return path


def list_investigations() -> list[str]:
    """Spec names available under the current base."""
    if not paths.INVESTIGATIONS_DIR.exists():
        return []
    return sorted(p.stem for p in paths.INVESTIGATIONS_DIR.glob("*.json"))


def list_runs(investigation: str | None = None) -> list[str]:
    """Run timestamps, newest last. Worth checking before loading a corpus,
    because `load_corpus` defaults to the most recent and a silently different
    run is the kind of mistake that produces a real-looking wrong answer."""
    root = paths.RECORDS_DIR if investigation is None else paths.RECORDS_DIR / investigation
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


# ═══════════════════════════════════════════════════════════════════════════════
# Cost
# ═══════════════════════════════════════════════════════════════════════════════

def estimate(rows: list[dict]) -> dict:
    """Rough per-provider cost. Distinguishes a $200 run from a $3 one.

    **Per provider, never one total.** Three accounting conventions — two count
    thinking inside output, one keeps it separate; one bills retrieved content as
    input and one does not — so a single number would imply a comparability that
    does not exist.

    The shape is `n_queries × tokens_per_query`, not a flat per-call figure.
    Earlier estimators used a constant and were badly wrong: a 16-search call
    measured 186K input against a 50K estimate. `tokens_per_query` is roughly a
    provider constant; the search COUNT is what varies, and it varies by question
    shape — recognition probes ran 2 searches, neighbour probes 8 to 11.

    That count cannot be predicted from a spec, so the range is wide on purpose.
    A narrow estimate here would be a guess wearing a number's clothes.
    """
    out: dict[str, dict] = {}
    for row in rows:
        model = row["model"]
        caps = caps_for(model)
        provider = provider_of(model)
        entry = out.setdefault(provider, {
            "records": 0, "searched": 0, "est_in_low": 0, "est_in_high": 0})

        searched = bool(row["params"].get("search"))
        n_turns = len(row["turns"])
        entry["records"] += n_turns
        entry["searched"] += n_turns if searched else 0

        if searched:
            # 2 searches on a recognition-shaped probe, 11 on a neighbour-shaped
            # one — both measured, and the spec cannot tell which this is.
            entry["est_in_low"] += n_turns * 2 * caps.tokens_per_query
            entry["est_in_high"] += n_turns * 11 * caps.tokens_per_query
        else:
            entry["est_in_low"] += n_turns * 200
            entry["est_in_high"] += n_turns * 2_000
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_investigation(path_or_spec) -> pd.DataFrame:
    """Spec -> one row per conversation. Sends nothing.

    Every call the run will make is decided here, so a mistake in the design
    costs a read rather than a battery.

    Rows come back in **round-robin order over repetitions, interleaved across
    providers**. The web changes during a run: a battery that finishes all of
    condition A before starting condition B has made time a hidden factor, and
    the same argument applies across providers.
    """
    if isinstance(path_or_spec, dict):
        spec, sha = path_or_spec, None
    else:
        p = pathlib.Path(path_or_spec)
        if not p.exists():
            p = paths.INVESTIGATIONS_DIR / f"{path_or_spec}.json"
        raw = p.read_bytes()
        spec = json.loads(raw)
        sha = hashlib.sha256(raw).hexdigest()
    validate_investigation(spec)

    rows: list[dict] = []
    summary: list[tuple] = []

    for study in spec["studies"]:
        on_unmet = study.get("on_unmet", "error")
        study_rows: list[dict] = []

        for model, block in study["providers"].items():
            conditions = expand_conditions(block)
            reps = block.get("repetitions", 1)
            levels_varying = [c["label"] for c in conditions]

            for rep in range(reps):
                for probe in study["probes"]:
                    system = join_text(probe.get("system_prompt"))
                    for p_i, path in enumerate(probe["prompt_paths"]):
                        for cond in conditions:
                            passed = {**cond["values"], "model": model}
                            if system:
                                passed["system"] = system
                            try:
                                resolved, cfg_passed, unmet = resolve(
                                    model, passed, on_unmet=on_unmet)
                            except InvestigationError:
                                raise
                            if unmet and on_unmet == "exclude":
                                continue

                            turns = [join_text(t) for t in path]
                            study_rows.append({
                                "study": study["study_id"],
                                "probe": probe["probe_id"],
                                "probe_hash": probe_hash(probe),
                                "model": model,
                                "provider": provider_of(model),
                                "path": p_i,
                                "rep": rep,
                                "condition": cond["label"],
                                "turns": turns,
                                "params": resolved,
                                "config_passed": cfg_passed,
                                "intent_unmet": [i for i, _ in unmet],
                            })
        rows.extend(study_rows)
        summary.append((study, study_rows, levels_varying))

    if not rows:
        raise InvestigationError(
            "no calls survive expansion — every cell was excluded or unmet")

    # (2) ROUND-ROBIN over repetitions, INTERLEAVED across providers. Sorting by
    #     rep first and provider second means rep 0 of every provider runs before
    #     rep 1 of any, so retrieval drift spreads across the design rather than
    #     landing on whichever arm ran last.
    rows.sort(key=lambda r: (r["rep"], r["provider"], r["study"], r["probe"],
                             r["path"], r["condition"]))

    run = pd.DataFrame(rows)
    run.attrs.update(investigation_id=spec["investigation_id"], spec=spec,
                     sha256=sha)

    _print_plan(spec, summary, rows)
    return run


def _print_plan(spec: dict, summary: list, rows: list[dict]) -> None:
    print(spec["investigation_id"])
    for study, study_rows, _ in summary:
        print(f"\n  {study['study_id']}")
        by_model: dict[str, int] = {}
        for r in study_rows:
            by_model[r["model"]] = by_model.get(r["model"], 0) + len(r["turns"])
        for model, n in sorted(by_model.items()):
            block = study["providers"][model]
            conds = len(expand_conditions(block))
            note = block.get("note")
            print(f"    {model:<32}{n:>5} calls   "
                  f"{conds} cond x {block.get('repetitions', 1)} rep")
            if note:
                print(f"      note: {note}")

    n_records = sum(len(r["turns"]) for r in rows)
    print(f"\n  {len(rows)} conversations, {n_records} records")

    est = estimate(rows)
    print("\n  estimated input tokens, per provider — never one total, because "
          "three\n  accounting conventions make a single number a false "
          "comparison:")
    for provider, e in sorted(est.items()):
        print(f"    {provider:<12} {e['records']:>4} records "
              f"({e['searched']} searched)   "
              f"{e['est_in_low']/1000:>6,.0f}K - {e['est_in_high']/1000:,.0f}K in")
    if any(e["searched"] for e in est.values()):
        print("  the range is wide because search COUNT varies by question shape "
              "— 2\n  on a recognition probe, 11 on a neighbour probe, both "
              "measured — and\n  a spec cannot say which this is.")


# ═══════════════════════════════════════════════════════════════════════════════
# Running
# ═══════════════════════════════════════════════════════════════════════════════

def _write_record(path: pathlib.Path, record: dict) -> None:
    """Write one record, without the scratch keys."""
    payload = {k: v for k, v in record.items()
               if not k.startswith("_") and k != "path_on_disk"}
    payload["config"] = record.get("_config")
    payload["response"] = record.get("_body")
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False,
                               default=str))


def run_investigation(run: pd.DataFrame, persist: bool = True,
                      export: bool = True, verbose: bool = True,
                      retries: int = 5, backoff: int = 4,
                      dispatch=None) -> tuple[pd.DataFrame, list]:
    """Execute a loaded investigation. Returns (results, raw).

    `dispatch` overrides the provider's own, for tests. Everything else follows
    §6b; the numbered comments mark orderings, not preferences.
    """
    inv = run.attrs.get("investigation_id", "unnamed")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = paths.RECORDS_DIR / inv / stamp
    if persist:
        n = 2
        while run_dir.exists():
            run_dir = paths.RECORDS_DIR / inv / f"{stamp}_{n}"
            n += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "_spec.json").write_text(
            json.dumps(run.attrs.get("spec", {}), indent=2, ensure_ascii=False))
        (run_dir / "_run.json").write_text(json.dumps({
            "investigation_id": inv,
            "started_at": stamp,
            "spec_sha256": run.attrs.get("sha256"),
            # Three provenance mechanisms answering three questions: what code
            # ran, whether the instrument changed, and what each intent resolved
            # to. The third is per-record.
            "harness": paths.provenance(),
            "probe_hashes": {r.probe: r.probe_hash
                             for r in run.itertuples(index=False)},
            "providers": sorted(run.provider.unique()),
        }, indent=2, default=str))

    providers = {p: get_provider(p, api_key=paths.api_key(p))
                 for p in run.provider.unique()}

    n_records = int(sum(len(r.turns) for r in run.itertuples(index=False)))
    if verbose:
        where = str(run_dir) if persist else "(not persisted)"
        print(f"{inv} · {len(run)} conversations, {n_records} records · {where}\n")

    rows: list[dict] = []
    raw: list = []
    failures: list[tuple] = []
    # (1) DECLARED OUTSIDE THE LOOP. The interrupt handler reads it, and a
    #     NameError there loses everything collected.
    conv_rows: list[dict] = []
    interrupted: BaseException | None = None
    seq = 0
    t_start = time.perf_counter()

    try:
        for r in run.itertuples(index=False):
            provider = providers[r.provider]
            send = dispatch or provider.dispatch
            conversation = f"{r.provider}/{r.probe}/p{r.path}/r{r.rep}"
            input_items: list = []
            conv_rows = []
            broke_at = None

            for turn_i, turn_text in enumerate(r.turns):
                # (3) PROVIDER-SHAPED. messages / input list / step_list — and
                #     the wrong one is a 400 on every turn after the first.
                input_items = [*input_items, provider.user_turn(turn_text)]
                config = provider.build(input_items, r.params)

                # (4) RETRY LIVES HERE, not in dispatch, so `latency` measures
                #     ONE call and `attempts` is visible in the record.
                body = None
                status = error = None
                attempt = 0
                latency = 0.0
                for attempt in range(1, retries + 1):
                    t0 = time.perf_counter()
                    try:
                        body, exc = send(config), None
                    # (5) Exception, NOT BaseException — an interrupt must reach
                    #     the outer handler rather than being caught here.
                    except Exception as e:
                        body, exc = None, e
                    latency = time.perf_counter() - t0

                    if exc is not None:
                        # (6) A TIMEOUT IS TRANSIENT. An earlier draft broke on
                        #     any exception, losing records a retry recovers.
                        status = provider.classify_exception(exc)
                        error = repr(exc)
                    else:
                        status, error = provider.status(body)

                    # (7) ONLY `unavailable` retries. A 400 five times wastes a
                    #     minute per record and never succeeds.
                    if status != "unavailable":
                        break
                    if attempt < retries:
                        time.sleep(backoff * attempt)

                parsed = provider.parse(body or {})
                prompt = turn_text
                # Structural checks AT PARSE TIME, warn-only. A weekly check
                # samples once; this sees every call. Never raises — failing
                # would lose the count, and how OFTEN an issue occurs is the
                # finding.
                issues = check_record(parsed, status, r.params, caps_for(r.model))
                record = {
                    "integrity": [
                        {"severity": i.severity, "code": i.code, "detail": i.detail}
                        for i in issues] or None,
                    "investigation": inv,
                    "provider": r.provider, "model": r.model,
                    "served_model": parsed.served_model,
                    "study": r.study, "probe": r.probe, "probe_hash": r.probe_hash,
                    "path": r.path, "rep": r.rep, "turn": turn_i,
                    "conversation": conversation, "n_turns": len(r.turns),
                    "condition": r.condition,
                    "status": status,
                    # (7b) NOT YET KNOWABLE. Written null here and filled after
                    #      the conversation ends.
                    "conversation_status": None,
                    "error": error, "attempts": attempt,
                    "latency": round(latency, 2),
                    "prompt": prompt, "prompt_hash": prompt_hash(prompt),
                    "answer_text": parsed.answer_text,
                    "answer_chars": parsed.answer_chars,
                    "answer_extraction": parsed.answer_extraction,
                    "in_tok_billed": parsed.in_tok_billed,
                    "in_tok_processed": parsed.in_tok_processed,
                    "out_tok_reported": parsed.out_tok_reported,
                    "thinking_tok": parsed.thinking_tok,
                    "n_queries": parsed.n_queries,
                    "n_citations": len(parsed.citations),
                    "thought_text": parsed.thought_text,
                    "n_thought_steps": parsed.n_thought_steps,
                    "n_sources_retrieved": parsed.n_sources_retrieved,
                    "n_answer_blocks": parsed.n_answer_blocks,
                    "n_process_blocks": parsed.n_process_blocks,
                    "grounded": parsed.grounded,
                    "config_passed": r.config_passed,
                    "config_resolved": r.params,
                    "intent_unmet": r.intent_unmet,
                    "_body": body, "_config": config,
                    "path_on_disk": None,
                }

                if persist:
                    name = (f"{seq:04d}_{r.provider}_{r.study}_{r.probe}"
                            f"_p{r.path}_r{r.rep}_t{turn_i}.json")
                    safe = "".join(c if c.isalnum() or c in "-_." else "_"
                                   for c in name)
                    fp = run_dir / safe
                    record["path_on_disk"] = str(fp)
                    # (8) FIRST WRITE. An interrupt keeps it.
                    _write_record(fp, record)

                conv_rows.append(record)
                raw.append(body)
                seq += 1

                if verbose:
                    _print_line(seq, n_records, r, turn_i, status, attempt,
                                latency, parsed)

                # (9) A FAILED TURN ENDS THIS CONVERSATION, NEVER THE RUN.
                #     `truncated` CONTINUES: the answer is real — measured at
                #     4,338 characters on one provider — and a truncated turn 0
                #     can still be passed back.
                if status not in ("ok", "truncated"):
                    broke_at = turn_i
                    failures.append((seq - 1, conversation, turn_i, error))
                    break

                # (10) AFTER the ok check. All steps verbatim where signatures
                #      are required.
                input_items = input_items + provider.passback(body or {})

            # (11) `failed` means it died on turn 0; `incomplete` means later.
            #      A conversation of truncated turns is still `ok` here —
            #      per-record status carries truncation, this carries whether the
            #      conversation finished.
            conv_status = ("ok" if broke_at is None
                           else "failed" if broke_at == 0
                           else "incomplete")
            for rec in conv_rows:
                rec["conversation_status"] = conv_status
                if persist and rec.get("path_on_disk"):
                    # (12) SECOND WRITE. Skipping this leaves null on disk while
                    #      the in-memory frame is correct — the bug that survived
                    #      a whole session of testing because nobody reloaded.
                    _write_record(pathlib.Path(rec["path_on_disk"]), rec)
            rows.extend(conv_rows)
            conv_rows = []

    # (13) BaseException, not Exception — a notebook stop is KeyboardInterrupt.
    except BaseException as e:
        interrupted = e
        for rec in conv_rows:
            if rec.get("conversation_status") is None:
                rec["conversation_status"] = "incomplete"
                if persist and rec.get("path_on_disk"):
                    _write_record(pathlib.Path(rec["path_on_disk"]), rec)
        rows.extend(rec for rec in conv_rows if rec not in rows)

        # (13b) SAVE, THEN RE-RAISE ANYTHING THAT IS NOT AN INTERRUPT.
        #
        # Catching BaseException is right for the partial-save, and wrong as a
        # place to stop. A TypeError in the record loop was indistinguishable
        # from a notebook stop: the run returned an empty DataFrame, printed no
        # error, and reported itself as interrupted. That cost a debugging round
        # when a genuine bug was introduced above.
        #
        # The records are already on disk by this point, so re-raising loses
        # nothing and surfaces the traceback that names the actual fault.
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise

    # (14) STRIP SCRATCH KEYS before the DataFrame, or they leak into it and
    #      then into the export.
    for rec in rows:
        rec.pop("_body", None)
        rec.pop("_config", None)

    results = pd.DataFrame(rows)
    results.attrs["run_dir"] = str(run_dir) if persist else None
    results.attrs["investigation_id"] = inv
    results.attrs["interrupted"] = repr(interrupted) if interrupted else None

    if verbose and len(results):
        _print_summary(results, failures, interrupted, n_records,
                       time.perf_counter() - t_start, run_dir if persist else None)

    if persist and export and len(results):
        # (15) Never let an export failure discard a completed run.
        try:
            from .export import export_corpus
            export_corpus(inv, run_dir.name, quiet=not verbose)
        except Exception as e:
            print(f"  export failed, records are on disk: {e!r}")

    return results, raw


def _print_line(seq, n_records, r, turn_i, status, attempt, latency, parsed):
    tag = f" x{attempt}" if attempt > 1 else ""
    code = {"ok": "200", "truncated": "TRUNC", "unavailable": "503",
            "incomplete": "INC", "error": "ERR"}.get(status, status)
    bits = []
    if parsed.in_tok_processed:
        bits.append(f"{parsed.in_tok_processed/1000:.1f}K→"
                    f"{(parsed.out_tok_reported or 0)/1000:.1f}K")
    if parsed.thinking_tok:
        bits.append(f"{parsed.thinking_tok} think")
    if parsed.n_queries:
        bits.append(f"{parsed.n_queries} srch")
    if parsed.citations:
        bits.append(f"{len(parsed.citations)} cite")
    turn_tag = f" t{turn_i}" if len(r.turns) > 1 else ""
    print(f"  [{seq:>3}/{n_records}] {r.provider[:3]} {r.probe[:26]:<26}"
          f" r{r.rep}{turn_tag}  {code}{tag}  {latency:6.1f}s  {'  '.join(bits)}")


def _print_summary(results, failures, interrupted, n_records, elapsed, run_dir):
    n_conv = results.conversation.nunique()
    conv_ok = int(results.groupby("conversation").conversation_status
                  .first().eq("ok").sum())
    print(f"\n  {n_conv} conversations · {conv_ok} ok · {len(failures)} failed"
          f" · {elapsed/60:.0f}m {elapsed%60:.0f}s")

    # Per provider, never one total. Three accounting conventions.
    for provider, g in results.groupby("provider"):
        retried = int((g.attempts > 1).sum())
        print(f"    {provider:<12} {len(g):>4} records  "
              f"{g.in_tok_processed.fillna(0).sum()/1000:>7,.0f}K in  "
              f"{g.out_tok_reported.fillna(0).sum()/1000:>6,.0f}K out"
              + (f"  {retried} retried ({int(g.attempts.sum())} calls)"
                 if retried else ""))

    if failures:
        print("\n  failed:")
        for i, conv, turn_i, err in failures[:10]:
            print(f"    [{i}] {conv} t{turn_i}\n         {err}")
        if len(failures) > 10:
            print(f"    … {len(failures) - 10} more")

    if interrupted is not None:
        print(f"\n  INTERRUPTED after {len(results)} of {n_records} records: "
              f"{type(interrupted).__name__}")
    if run_dir:
        print(f"\n  {run_dir}")
