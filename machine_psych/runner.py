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
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from . import paths
from .capabilities import caps_for, provider_of

# Re-exported so `runner.set_base(...)` keeps working for callers, while the
# STATE lives in `paths`. A facade over one source of truth, not a copy of it —
# an earlier version held the paths here and `corpus` reached back into this
# module to read them, which made a low-level module depend on a high-level one.
from .governor import DailyCapReached, Governor, pool_key
from .integrity import check_record
from .outage import OutageGaveUp, OutageLatch
from .paths import set_api_key, set_base
from .providers import get_provider
from .ratelimit import RateLimits
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
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
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
            "records": 0, "search_permitted": 0,
            "est_in_low": 0, "est_in_high": 0})

        permitted = bool(row["params"].get("search"))
        n_turns = len(row["turns"])
        entry["records"] += n_turns
        entry["search_permitted"] += n_turns if permitted else 0

        # ── the LOW bound assumes search PERMITTED is not search USED ────────
        #
        # This read "80 records (80 searched)" and estimated 1.86M-10.2M input
        # tokens for a battery that used 328K — 18% of the stated floor. It
        # treated every search-enabled record as grounding.
        #
        # Measured on that battery: **5 of 80 records actually searched.** All
        # three providers decline when the prompt does not need retrieval, which
        # this project established and this function did not apply.
        #
        # So the range now spans the real outcome: the floor assumes NOTHING
        # searches, the ceiling assumes everything does at the heavy end. That is
        # a wide range, and honestly so — grounding is a model decision, not a
        # setting, and a spec cannot predict it. The floor is now reachable,
        # which the old one was not.
        if permitted:
            entry["est_in_low"] += n_turns * 2_000            # none of them search
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
              f"({e['search_permitted']} may search)   "
              f"{e['est_in_low']/1000:>6,.0f}K - {e['est_in_high']/1000:,.0f}K in")
    if any(e["search_permitted"] for e in est.values()):
        # "(80 searched)" said PERMITTED and read as EXECUTED. On a real battery
        # 5 of 80 searched, and the estimate overstated input by 6x at its floor.
        # "Expect the low end" was advice drawn from a Claude-only battery (5 of
        # 80 searched). On the same kind of prompt OpenAI searched on 14 of 20 —
        # the grounding rate is a property of the PROVIDER as much as the prompt,
        # so the advice has to say which is which.
        print("  `may search` counts records where search is PERMITTED, not where "
              "it will\n  RUN — that is the model's decision, and it varies by "
              "PROVIDER. On one\n  battery of business-advice prompts: OpenAI "
              "searched 14 of 20, Claude 2,\n  Gemini 0. The floor assumes none "
              "search; the ceiling assumes all do.\n  Expect OpenAI nearer the "
              "ceiling, and every provider there if the prompts\n  need current "
              "information.")


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
                               default=str), encoding="utf-8")


def ledger_key(provider, model, study, probe, path, rep, turn, condition) -> str:
    """The identity of one logical API call, stable across restarts.

    **This is an idempotency key in the ordinary engineering sense: it names the
    ACTION, not the network attempt.** The standard failure it prevents is a
    request that reached the provider, was billed, generated an answer, and lost
    its response on the way back — retried blindly, that pays twice for one
    record.

    Derived from content rather than generated, because a key that changes
    between attempts is not a key. A random UUID per retry is the documented way
    people lose idempotency and get charged repeatedly.

    `model` is in here and `provider` is not enough: two models from one provider
    are two different actions, and leaving it out is the same defect that made
    eighty records share forty conversation ids.
    """
    return "|".join(str(x) for x in
                    (provider, model, study, probe, path, rep, turn,
                     condition or "base"))


def completed_keys(run_dir: pathlib.Path) -> set[str]:
    """Which logical calls a run directory already holds.

    **The ledger is the records themselves.** An in-memory set dies with the
    process, and a 72-hour battery that loses power at hour 60 would re-send
    everything — paying twice for work already on disk. Reading the directory
    makes the ledger survive anything the machine survives.

    A record without `conversation_status` was in flight when the run stopped and
    is NOT counted complete: it may have been billed, but there is no answer in
    it, so it has to be redone.
    """
    done: set[str] = set()
    if not run_dir.exists():
        return done
    for f in sorted(run_dir.glob("[0-9]*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue          # a torn write from a power cut; redo it
        if rec.get("conversation_status") is None:
            continue
        done.add(ledger_key(rec.get("provider"), rec.get("model"),
                            rec.get("study"), rec.get("probe"),
                            rec.get("path"), rec.get("rep"),
                            rec.get("turn"), rec.get("condition")))
    return done


def _run_conversation(r, *, provider, send, seq_base, n_records, retries, backoff,
                      persist, run_dir, inv, verbose, failures, raw_out,
                      conv_rows, print_lock=None, latch=None,
                      governor=None):
    """Run one conversation — every turn of one row of the plan — and return its
    records.

    **This is the unit of concurrency.** Turns inside a conversation are
    sequential by necessity: turn 1 passes back turn 0's response. Conversations
    are independent of one another, so they are what can safely run in parallel.

    Lifted out of `run_investigation` without changing what it does, so the
    numbered orderings it carries still hold. Two things changed to make it
    callable from several threads:

    **`seq_base` replaces a shared counter.** Record filenames were numbered from
    a mutable `seq` incremented per turn. Under concurrency that makes a filename
    depend on which thread finished first, so the same battery rerun would name
    the same record differently. The number now comes from the row's position in
    the plan.

    **`failures`, `raw_out` and `conv_rows` are passed IN.** `list.append` is
    atomic under the GIL so the lists are safe, but their ORDER is not
    deterministic — the caller sorts by sequence before building the frame.

    `conv_rows` in particular must be the caller's list, not a local one. The
    interrupt handler marks whatever is in flight as `incomplete` and writes it,
    and it can only do that if it holds a reference. Making it local here broke
    that, which the interrupt test caught immediately.
    """
    # MODEL, not provider. Two models from one provider produced
    # IDENTICAL conversation ids — 80 records collapsed into 40
    # conversations, and the run summary reported "40 conversations"
    # against 80 records. Any analysis using `conversation` as a unique
    # key silently merged the two arms it was comparing.
    #
    # The condition is included too: a sweep runs the same probe at
    # several settings, and those are separate conversations.
    _model = r.model.split("/", 1)[-1]
    conversation = (f"{r.provider}/{_model}/{r.probe}"
                    f"/p{r.path}/r{r.rep}"
                    + (f"/{r.condition}" if r.condition else ""))
    input_items: list = []
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
        # Initialised OUTSIDE the retry loop, like `body` and `status`, because
        # the record is built after the loop ends and a dispatch that raised on
        # its first attempt would otherwise leave this unbound.
        limits = RateLimits(source=r.provider)
        for attempt in range(1, retries + 1):
            # (4c) HOLD IF THE WORLD IS DOWN. Returns instantly when it is not.
            #      Placed before the clock so an outage does not inflate
            #      `latency`, which measures one call rather than one wait.
            if latch is not None:
                latch.wait(r.provider)

            # (4d) PACE AGAINST THE PROVIDER'S STATED LIMIT. Blocks only when
            #      this call would push past the margin; returns instantly
            #      otherwise. Estimated before sending because a header arrives
            #      one round trip too late to prevent anything.
            if governor is not None:
                # Keyed on the POOL, not the provider: Anthropic pools by model
                # family, so Sonnet and Opus draw on separate budgets and
                # tracking them as one throttles both to whichever is busier.
                #
                # The estimate is LEARNED from what calls on this pool actually
                # cost. An earlier version used `tokens_per_query x 2` whenever
                # `search` was allowed, which assumed a search-enabled call would
                # search — the assumption this project disproved, and one that
                # over-reserved 12x on the 94% of calls that never search.
                _pool = pool_key(r.provider, r.model)
                governor.count_day(_pool)
                governor.acquire(
                    _pool,
                    governor.estimate_in(_pool, bool(r.params.get("search"))),
                    r.params.get("max_tokens", 4096))

            t0 = time.perf_counter()
            try:
                # (4e) RELEASE IN A `finally`, paired with the acquire above.
                #      Releasing only on the paths that LEAVE the retry loop
                #      leaked a slot per retry: a run with five retries narrowed
                #      itself permanently, and the narrowing grew with every
                #      transient failure — worst exactly when throughput matters
                #      most.
                # (4b) THE FULL PATH WHERE THERE IS ONE, so rate-limit headers
                #      survive. `send` is the provider's own bound `dispatch`
                #      in a real run and a plain function in a test, and only
                #      the former can report headers — a stub keeps the
                #      all-None `RateLimits` set above.
                bound = getattr(send, "__self__", None)
                if bound is not None and hasattr(bound, "dispatch_full"):
                    # The same key on every attempt of this turn, so a retry
                    # after a LOST RESPONSE returns the original answer rather
                    # than regenerating and billing again. Derived from the
                    # record's identity, never generated, because a key that
                    # changes between attempts is not a key.
                    _http, body, limits = bound.dispatch_full(
                        config,
                        idempotency_key=ledger_key(
                            r.provider, r.model, r.study, r.probe, r.path,
                            r.rep, turn_i, r.condition))
                else:
                    body = send(config)
                exc = None
            # (5) Exception, NOT BaseException — an interrupt must reach
            #     the outer handler rather than being caught here.
            except Exception as e:
                body, exc = None, e
            finally:
                latency = time.perf_counter() - t0
                if governor is not None:
                    governor.release(pool_key(r.provider, r.model))

            if exc is not None:
                # (6) A TIMEOUT IS TRANSIENT. An earlier draft broke on
                #     any exception, losing records a retry recovers.
                status = provider.classify_exception(exc)
                error = repr(exc)
                if latch is not None:
                    # Only CONNECTION errors are evidence about the network. An
                    # HTTP error means the network carried the request fine.
                    import requests as _rq
                    latch.record_failure(
                        r.provider, conversation,
                        isinstance(exc, (_rq.exceptions.ConnectionError,
                                         _rq.exceptions.Timeout)))
            else:
                status, error = provider.status(body)
                if governor is not None:
                    governor.observe(pool_key(r.provider, r.model), limits)
                if latch is not None:
                    # A call got through. Without clearing, three failures spread
                    # across a whole battery would eventually latch a healthy run.
                    latch.record_success(r.provider)

            # (7) ONLY `unavailable` retries. A 400 five times wastes a
            #     minute per record and never succeeds.
            if status != "unavailable":
                break
            if attempt < retries:
                # (7a) THE PROVIDER'S OWN NUMBER FIRST. Every one returns
                #      `retry-after` saying exactly how long to wait; a fixed
                #      schedule ignores it. That matters more under concurrency,
                #      where every worker otherwise wakes at the same instant
                #      and re-trips the same limit.
                #
                #      Jitter regardless, for the same reason.
                wait = limits.retry_after or backoff * attempt
                time.sleep(wait * (1 + random.random() * 0.25))

        parsed = provider.parse(body or {})
        prompt = turn_text
        # Structural checks AT PARSE TIME, warn-only. A weekly check
        # samples once; this sees every call. Never raises — failing
        # would lose the count, and how OFTEN an issue occurs is the
        # finding.
        issues = check_record(parsed, status, r.params, caps_for(r.model))
        record = {
            # Remaining capacity as of THIS response. All-None where the provider
            # sends no headers — `None` means "did not say", which is not the
            # same as zero remaining.
            **limits.as_record(),
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

        if governor is not None:
            _pool = pool_key(r.provider, r.model)
            # What the call ACTUALLY cost, so the estimate converges instead of
            # resting on an assumption about whether search would happen.
            governor.record_cost(_pool, parsed.in_tok_processed or parsed.in_tok_billed)
            warning = governor.queue.record(_pool, latency)
            if warning and print_lock is not None:
                with print_lock:
                    print(f"\n  QUEUEING — {warning}\n")
            if parsed.grounded:
                governor.count_grounded(_pool)

        if persist:
            # MODEL in the name. Two models from one provider produced files
            # distinguished only by their sequence prefix, so a human reading the
            # directory could not tell the arms apart — and a resume keyed on the
            # name would have merged them.
            _m = str(r.model).split("/", 1)[-1]
            name = (f"{seq_base + turn_i:04d}_{r.provider}_{_m}_{r.study}"
                    f"_{r.probe}_p{r.path}_r{r.rep}_t{turn_i}.json")
            safe = "".join(c if c.isalnum() or c in "-_." else "_"
                           for c in name)
            fp = run_dir / safe
            record["path_on_disk"] = str(fp)
            # (8) FIRST WRITE. An interrupt keeps it.
            _write_record(fp, record)

        conv_rows.append(record)
        raw_out.append(body)

        if verbose:
            # One line at a time. Sixteen workers writing to stdout interleave
            # mid-line and the progress log becomes unreadable.
            if print_lock is not None:
                with print_lock:
                    _print_line(seq_base + turn_i + 1, n_records, r, turn_i,
                                status, attempt, latency, parsed)
            else:
                _print_line(seq_base + turn_i + 1, n_records, r, turn_i, status,
                            attempt, latency, parsed)

        # (9) A FAILED TURN ENDS THIS CONVERSATION, NEVER THE RUN.
        #     `truncated` CONTINUES: the answer is real — measured at
        #     4,338 characters on one provider — and a truncated turn 0
        #     can still be passed back.
        if status not in ("ok", "truncated"):
            broke_at = turn_i
            failures.append((seq_base + turn_i, conversation, turn_i, error))
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
    return conv_rows



def run_investigation(run: pd.DataFrame, persist: bool = True,
                      export: bool = True, verbose: bool = True,
                      retries: int = 5, backoff: int = 4,
                      concurrency: int | dict[str, int] | str = 1,
                      resume: str | bool | None = None,
                      outage_ceiling: float = 12 * 3600, latch=None,
                      silent_ceiling: int = 64, limits: dict | None = None,
                      dispatch=None) -> tuple[pd.DataFrame, list]:
    """Execute a loaded investigation. Returns (results, raw).

    `dispatch` overrides the provider's own, for tests. Everything else follows
    §6b; the numbered comments mark orderings, not preferences.

    **`concurrency` is per provider, not global**, and a dict sets it per
    provider by name: `{"anthropic": 8, "gemini": 2}`. Anything unnamed falls
    back to 1. An int applies the same cap to every provider, so
    `concurrency=4` across three providers is twelve calls in flight and four
    against each account.

    Per provider because rate limits are, and because they differ by an order of
    magnitude: Gemini's free tier runs about 10 requests per minute against
    Anthropic's hundreds. One number for all three is therefore either too slow
    for Anthropic or a 429 storm on Gemini.

    **`concurrency="auto"` paces against the limits each provider reports**,
    instead of a number chosen by hand. Every response states remaining requests
    and tokens; the governor keeps a token bucket per counter and holds a call
    that would push past 80% of any of them. It starts at two in flight and opens
    up once a provider has stated its limits.

    Preferred over a fixed number, because a fixed number is a guess. This
    project tried to measure one and could not: batteries at 8, 16 and 32
    returned identical throughput because the battery held six conversations per
    repetition, so the extra workers were idle. That was the design ceiling, not
    a rate limit.

    Default 1 — sequential, the behaviour every earlier battery had.

    Conversations run in parallel WITHIN a repetition and the run waits at each
    repetition boundary. That is what keeps the ordering guarantee: retrieval
    drift must spread across conditions rather than confound with them, which
    requires that no arm FINISHES before another STARTS. Running every arm of
    rep 0 in one window satisfies that; running all of arm A then all of arm B
    does not — and so does not become acceptable just because it is faster.

    Measured sequentially: ~40s per record, so 8,000 records is about four days.

    **`resume` continues a run that stopped.** `True` picks the most recent run
    of this investigation; a timestamp string names one. Records already on disk
    are skipped, and everything else is dispatched into the SAME directory.

    A four-day battery will be interrupted — power, wifi, a reboot, a laptop
    lid. Without this, the next attempt starts a new directory and re-sends
    everything, which pays a second time for records already collected. **The
    records ARE the ledger**: an in-memory set dies with the process, and a file
    on disk does not.

    A record still missing `conversation_status` was in flight when the run
    stopped and is redone. It may have been billed with no answer returned, which
    is the ambiguous case no client can resolve — so the cost is accepted rather
    than a hole left in the corpus.
    """
    inv = run.attrs.get("investigation_id", "unnamed")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = paths.RECORDS_DIR / inv / stamp
    already: set[str] = set()

    if resume:
        # Continue INTO the existing directory rather than beside it. A resume
        # that writes somewhere new leaves the corpus split across two places
        # and `load_corpus` reads one of them.
        prior = sorted((paths.RECORDS_DIR / inv).glob("*"),
                       key=lambda d: d.name) if (paths.RECORDS_DIR / inv).exists() else []
        prior = [d for d in prior if d.is_dir()]
        if resume is not True:
            prior = [d for d in prior if d.name == str(resume)]
        if not prior:
            raise FileNotFoundError(
                f"no run to resume for {inv!r}"
                + (f" named {resume!r}" if resume is not True else "")
                + f". Runs present: {[d.name for d in prior] or 'none'}")
        run_dir = prior[-1]
        already = completed_keys(run_dir)
        print(f"  resuming {run_dir.name} — {len(already)} records already on "
              f"disk, not re-sent")

    if persist:
        if not resume:
            n = 2
            while run_dir.exists():
                run_dir = paths.RECORDS_DIR / inv / f"{stamp}_{n}"
                n += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "_spec.json").write_text(
            json.dumps(run.attrs.get("spec", {}), indent=2, ensure_ascii=False), encoding="utf-8")
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
        }, indent=2, default=str), encoding="utf-8")

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
    t_start = time.perf_counter()

    # (1b) SEQUENCE NUMBERS ARE PRE-ASSIGNED, never counted during the run.
    #      A shared counter makes a record's filename depend on which thread
    #      finished first, so the same battery rerun would name the same record
    #      differently. Position in the plan is stable; arrival order is not.
    plan = list(run.itertuples(index=False))
    bases, _n = {}, 0
    for i, r in enumerate(plan):
        bases[i] = _n
        _n += len(r.turns)

    # (1b2) SKIP WHAT IS ALREADY ON DISK, but keep the sequence numbers computed
    #       over the WHOLE plan. Renumbering after the skip would give a resumed
    #       record a different filename from the one it would have had, and the
    #       directory would no longer read in plan order.
    if already:
        keep = []
        for i, r in enumerate(plan):
            keys = [ledger_key(r.provider, r.model, r.study, r.probe, r.path,
                               r.rep, t, r.condition) for t in range(len(r.turns))]
            if all(k in already for k in keys):
                continue          # every turn of this conversation is recorded
            keep.append((i, r))
        skipped = len(plan) - len(keep)
        plan_idx = keep
        print(f"  {skipped} conversations already complete, "
              f"{len(keep)} to run")
    else:
        plan_idx = list(enumerate(plan))

    # (1c) IN FLIGHT, so the interrupt handler can still see partial work.
    #      Every worker appends to its own list; the handler walks all of them.
    #      `conv_rows` alone was enough while this was sequential and is not now.
    in_flight: list[list[dict]] = []
    print_lock = threading.Lock()

    # (1f) ONE LATCH PER RUN, shared by every worker. Before this, an outage
    #      longer than the ~40s retry budget failed the conversation, then the
    #      next, then the next — as fast as the network could refuse them.
    #      `outage_ceiling` and `latch` are parameters because the latch SLEEPS
    #      in real time — a test that triggers an outage would otherwise wait
    #      hours, and a caller with a flakier connection may want a longer wall.
    # `silent_ceiling` is the concurrency for a provider that answers but reports
    # no limits — Gemini, today. Without it such a provider never leaves cold
    # start and runs two at a time for the whole battery, and every repetition
    # waits for it.
    # `limits` declares what a console shows, for providers that do not report
    # it: {"gemini": {"rpm": 1000, "input_tpm": 2_000_000, "rpd": 10_000,
    #                 "grounding_rpd": 1_500}}. Keyed by POOL, so an Anthropic
    #  family is "anthropic:sonnet" rather than "anthropic".
    governor = (Governor(silent_ceiling=silent_ceiling, declared=limits)
                if concurrency == "auto" else None)
    if governor is not None and resume:
        # Today's calls already spent today's quota. A fresh governor would
        # count from zero and allow a second full day of it.
        governor.seed_today(run_dir)

    if latch is None:
        latch = OutageLatch(
            providers={r.provider for r in plan},
            ceiling=outage_ceiling,
            probe=lambda name: providers[name].reachable(
                next(row.model for row in plan if row.provider == name)))

    def _gated(i, r, gates):
        """Hold this provider's slot for the whole conversation.

        Acquired around the conversation rather than around each call, because a
        multi-turn conversation that released between turns could have more of
        one provider in flight than the cap allows.
        """
        with gates[r.provider]:
            return _one(i, r)

    def _one(i, r):
        mine: list[dict] = []
        in_flight.append(mine)
        _run_conversation(
            r, provider=providers[r.provider],
            send=dispatch or providers[r.provider].dispatch,
            seq_base=bases[i], n_records=n_records, retries=retries,
            backoff=backoff, persist=persist, run_dir=run_dir, inv=inv,
            verbose=verbose, failures=failures, raw_out=raw,
            conv_rows=mine, print_lock=print_lock, latch=latch,
            governor=governor)
        return mine

    try:
        # A dict always takes the concurrent path, even if every value is 1:
        # the semaphores are what enforce the caps, and the sequential branch has
        # none. `max()` over an empty dict would also raise, hence the default.
        _auto = concurrency == "auto"
        _max_conc = (max(concurrency.values(), default=1)
                     if isinstance(concurrency, dict)
                     else 1 if _auto else concurrency)
        if _max_conc <= 1 and not isinstance(concurrency, dict) and not _auto:
            for i, r in plan_idx:
                conv_rows = []
                in_flight.append(conv_rows)
                _run_conversation(
                    r, provider=providers[r.provider],
                    send=dispatch or providers[r.provider].dispatch,
                    seq_base=bases[i], n_records=n_records, retries=retries,
                    backoff=backoff, persist=persist, run_dir=run_dir, inv=inv,
                    verbose=verbose, failures=failures, raw_out=raw,
                    conv_rows=conv_rows, print_lock=print_lock, latch=latch,
                    governor=governor)
                rows.extend(conv_rows)
                in_flight.remove(conv_rows)
                conv_rows = []
        else:
            # (1d) A BARRIER AT EVERY REPETITION. Parallel inside one rep,
            #      sequential between them. Without the barrier the first
            #      conditions to finish would see a different web from the last,
            #      which is the confound the ordering exists to prevent.
            #
            #      Workers are capped PER PROVIDER because rate limits are.
            by_rep: dict = {}
            for i, r in plan_idx:
                by_rep.setdefault(r.rep, []).append((i, r))

            # (1g) A SEMAPHORE PER PROVIDER, because a shared pool does not cap
            #      anything per provider. Sized at `concurrency * n_providers`,
            #      every worker in it could be running the same provider — so
            #      a rep holding twenty Gemini conversations and four Anthropic
            #      would put all its workers on Gemini, which is exactly the
            #      account with the tightest limit. The comment here claimed a
            #      per-provider cap that the code did not enforce.
            # Under "auto" the governor paces, so the semaphore is only a sanity
            # ceiling rather than the control — sized at the governor's own cap.
            caps = {p: (governor._max if _auto else
                        concurrency.get(p, 1) if isinstance(concurrency, dict)
                        else concurrency)
                    for p in {r.provider for _, r in plan_idx}}
            gates = {p: threading.Semaphore(max(1, n)) for p, n in caps.items()}

            for rep in sorted(by_rep):
                group = by_rep[rep]
                # Enough threads that every provider can reach its own cap at
                # once; the semaphores, not the pool size, do the limiting.
                workers = max(1, sum(caps.get(p, 1)
                                     for p in {r.provider for _, r in group}))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_gated, i, r, gates) for i, r in group]
                    try:
                        for f in as_completed(futures):
                            done = f.result()
                            rows.extend(done)
                            if done in in_flight:
                                in_flight.remove(done)
                    except BaseException:
                        # (1e) STOP SUBMITTING, then let the outer handler write
                        #      whatever is still in flight. Without this the pool
                        #      drains every queued conversation before the
                        #      interrupt is honoured.
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise

    # (13) BaseException, not Exception — a notebook stop is KeyboardInterrupt.
    except BaseException as e:
        interrupted = e
        # A COPY, deliberately. Workers may still be appending when the handler
        # runs, and iterating a list that is being mutated raises "list changed
        # size during iteration" — which would lose exactly the partial records
        # this handler exists to save. Ruff's PERF101 flags the cast as
        # unnecessary; it is not, and the rule is disabled for this line.
        for pending in list(in_flight):  # noqa: PERF101
            for rec in pending:
                if rec.get("conversation_status") is None:
                    rec["conversation_status"] = "incomplete"
                    if persist and rec.get("path_on_disk"):
                        _write_record(pathlib.Path(rec["path_on_disk"]), rec)
            rows.extend(rec for rec in pending if rec not in rows)

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
        # OutageGaveUp joins them. It is not a bug: it means the network stayed
        # down past the ceiling, which is a STOP rather than a fault. Re-raising
        # it would discard the records collected in memory and print a traceback
        # for a condition the run handled correctly. The message already points
        # at `resume=True`.
        # DailyCapReached joins them for the same reason: a quota spent is a
        # handoff to `resume=True`, not a bug. Grouping it with bugs would discard
        # everything collected in memory and print a traceback for a condition
        # the run handled correctly — the mistake first made with OutageGaveUp.
        if not isinstance(e, (KeyboardInterrupt, SystemExit, OutageGaveUp,
                              DailyCapReached)):
            raise

    # (13b) A RESUMED RUN RETURNS THE WHOLE CORPUS, not only what this attempt
    #       collected. Returning the increment would make every analysis after a
    #       resume silently partial, which is worse than the interruption.
    if already:
        for f in sorted(run_dir.glob("[0-9]*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("conversation_status") is None:
                continue
            k = ledger_key(rec.get("provider"), rec.get("model"), rec.get("study"),
                           rec.get("probe"), rec.get("path"), rec.get("rep"),
                           rec.get("turn"), rec.get("condition"))
            if k in already:
                rec["path_on_disk"] = str(f)
                rows.append(rec)

    # (13c) SORT BY PLAN POSITION. Under concurrency rows arrive in COMPLETION
    #       order, which varies run to run — so a corpus would not be
    #       reproducible and record_id would not match the filename on disk.
    #       `path_on_disk` carries the pre-assigned sequence; records without a
    #       path (persist=False) keep their arrival order, which is all that is
    #       available.
    rows.sort(key=lambda rec: (pathlib.Path(rec["path_on_disk"]).name
                               if rec.get("path_on_disk") else ""))

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
