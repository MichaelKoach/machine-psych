"""Investigation specs: validate, hash, expand, resolve.

A spec says what to ask and under what conditions. This module turns one into a
list of calls without making any, so a mistake costs a round trip and not a
battery.

Three decisions shape it, each recorded where it applies:

**Prompts are shared; conditions are per provider.** You never write a condition a
provider cannot honour, because you write what each provider actually does. The
cost is repeating `reasoning: "off"` three times, and that repetition is exactly
where you notice one provider cannot do it.

**The vocabulary is the union of provider capabilities**, not the intersection —
`verbosity` is expressible in a shared spec rather than buried in a provider
block, because a study design should be readable in one place.

**Unmet intents raise.** Not warn-and-downgrade, which is what an evaluation
framework does because a downgraded arm still scores. We run designed
comparisons, where a silently downgraded arm is a condition that looks like a
manipulation and is not.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
from typing import Any

from .capabilities import ModelCaps, caps_for, provider_of

__all__ = [
    "InvestigationError", "UnmetIntentError",
    "join_text", "probe_hash", "prompt_hash", "validate_investigation",
    "expand_conditions", "resolve", "load_investigation_dict",
    "INTENTS", "META_KEYS",
]


class InvestigationError(ValueError):
    """A structural problem with a spec: ragged probes, duplicate ids, empty
    turns, prompt sets that differ across providers."""


class UnmetIntentError(InvestigationError):
    """A study asks a provider for something it cannot do.

    Raising is the default and warn-and-downgrade is deliberately not an option.
    An evaluation framework can downgrade `effort: "max"` to `"high"` and log it,
    because a downgraded run still produces a scoreable answer. A downgraded ARM
    is a condition that looks like a manipulation and is not, and it dies at n=7
    looking like a real null result.
    """


# Everything in a provider block that is not a condition to sweep.
META_KEYS = {"repetitions", "exclude", "include", "note"}

# The union vocabulary. A provider that cannot honour one raises rather than
# silently ignoring it; `ModelCaps.supports` decides, taking the VALUE — because
# `reasoning` is supported on every provider and `reasoning: "off"` is not.
INTENTS = {
    "model", "max_tokens", "reasoning", "verbosity", "search",
    "system", "max_tool_calls",
    # Recognised and universally dead. Listed so they route through the unmet
    # path — which explains WHY per provider — rather than through the
    # unknown-intent warning, which would call them typos. They are not typos;
    # they are parameters that used to work.
    "temperature", "top_p", "top_k",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Text and hashing
# ═══════════════════════════════════════════════════════════════════════════════

def join_text(value: str | list[str] | None) -> str | None:
    """A string, or a list of lines joined with newlines.

    JSON strings cannot contain literal newlines, so a long prompt is written as
    a list. This is the only place that convention is interpreted.
    """
    if value is None:
        return None
    return value if isinstance(value, str) else "\n".join(value)


def probe_hash(probe: dict) -> str:
    """Content hash over a probe's prompts and system prompt.

    Computed, never declared. lm-eval-harness carries a manual VERSION on every
    task because their tasks are CODE and a human must judge whether a change is
    semantically meaningful. Ours are PROMPTS: a hash detects the change instead
    of trusting someone to remember, and it cannot be forgotten.

    Excludes `probe_id`, `rationale` and `note` — renaming a probe or improving
    its rationale does not change the instrument, and treating it as though it
    did would make the hash useless for the question it exists to answer.
    """
    payload = {
        "system_prompt": join_text(probe.get("system_prompt")),
        "prompt_paths": [[join_text(t) for t in path]
                         for path in probe.get("prompt_paths", [])],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def prompt_hash(text: str | None) -> str:
    """Content hash of one prompt, for joining records across providers.

    This is the mechanism the whole cross-provider design rests on: three modules
    computing the same hash independently found nine shared prompts across three
    corpora with no coordination and no shared identifier.
    """
    return hashlib.sha256((text or "").encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# Validation — structure only. Parameter legality is the API's job.
# ═══════════════════════════════════════════════════════════════════════════════

def validate_investigation(spec: dict) -> bool:
    """Raise on anything structurally wrong; return True otherwise.

    Deliberately does NOT check parameter values. Every provider rejects unknown
    parameters and names the offending field, and those rejections stay current
    for free. What is checked here is what the API cannot see: shape, uniqueness,
    and whether the arms will actually compare.
    """
    if not spec.get("investigation_id"):
        raise InvestigationError("missing investigation_id")
    if not spec.get("studies"):
        raise InvestigationError("no studies")

    seen_studies: set[str] = set()
    for study in spec["studies"]:
        sid = study.get("study_id")
        if not sid:
            raise InvestigationError("a study has no study_id")
        if sid in seen_studies:
            raise InvestigationError(f"duplicate study_id {sid!r}")
        seen_studies.add(sid)

        _validate_probes(study, sid)
        _validate_providers(study, sid)

    return True


def _validate_probes(study: dict, sid: str) -> None:
    probes = study.get("probes")
    if not probes:
        raise InvestigationError(f"study {sid!r} has no probes")

    seen: set[str] = set()
    for probe in probes:
        pid = probe.get("probe_id")
        if not pid:
            raise InvestigationError(f"a probe in {sid!r} has no probe_id")
        if pid in seen:
            raise InvestigationError(f"duplicate probe_id {pid!r} in {sid!r}")
        seen.add(pid)

        paths = probe.get("prompt_paths")
        if not paths:
            raise InvestigationError(f"probe {pid!r} has no prompt_paths")

        # Rectangular: every path in a probe has the same turn count. A probe
        # whose paths differ in length produces records that are not comparable
        # turn-for-turn, which is the entire point of grouping them as one probe.
        counts = {len(p) for p in paths}
        if len(counts) > 1:
            detail = ", ".join(f"path {i} has {len(p)}" for i, p in enumerate(paths))
            raise InvestigationError(
                f"probe {pid!r} is not rectangular — {detail}. Paths in one probe "
                f"must have the same turn count; use separate probes otherwise.")
        if counts.pop() < 1:
            raise InvestigationError(f"probe {pid!r} has an empty path")

        for p_i, path in enumerate(paths):
            for t_i, turn in enumerate(path):
                if not join_text(turn):
                    raise InvestigationError(
                        f"probe {pid!r} path {p_i} turn {t_i} is empty")


def _validate_providers(study: dict, sid: str) -> None:
    providers = study.get("providers")
    if not providers:
        raise InvestigationError(
            f"study {sid!r} has no `providers` block. Conditions are written per "
            f"provider so you never ask for something a provider cannot do.")

    for key, block in providers.items():
        if "/" not in key:
            raise InvestigationError(
                f"provider key {key!r} in {sid!r} is not 'provider/model'. The "
                f"provider must be explicit rather than inferred from a model "
                f"name that happens to look like one vendor's.")
        caps_for(key)          # raises UnknownModelError with the add procedure

        if not isinstance(block, dict):
            raise InvestigationError(f"{key!r} in {sid!r} is not a config block")

        labels = [inc.get("label") for inc in block.get("include", [])]
        if len(labels) != len(set(labels)):
            raise InvestigationError(f"duplicate include labels under {key!r} in {sid!r}")

    # Prompt sets must be identical across providers within a study: a study whose
    # providers ask different questions produces a corpus containing arms that do
    # not compare, and NOTHING DOWNSTREAM WOULD NOTICE — every join is on
    # prompt_hash, so mismatched prompts simply fail to join and read as missing
    # data rather than as a design error.
    #
    # Today the invariant holds by construction: prompts live on probes, which are
    # study-level, so every provider gets the same set. An earlier version of this
    # function "enforced" it by hashing the probes and checking the result was
    # non-empty — a check that could not fail, since the probes are already
    # validated non-empty upstream. Enforcement theatre is worse than no check:
    # it reads as a guarantee and provides none.
    #
    # What is actually guarded is the format change that would break the
    # invariant — a provider block carrying its own probes or prompts.
    for key, block in providers.items():
        intruders = {"probes", "prompt_paths", "system_prompt"} & set(block)
        if intruders:
            raise InvestigationError(
                f"provider block {key!r} in {sid!r} declares {sorted(intruders)}. "
                f"Prompts are study-level so every provider is asked the same "
                f"questions; per-provider prompts would produce arms that cannot "
                f"be compared, and the join on prompt_hash would silently report "
                f"them as missing rather than as mismatched.")


# ═══════════════════════════════════════════════════════════════════════════════
# Condition expansion — per provider block
# ═══════════════════════════════════════════════════════════════════════════════

def _levels(block: dict) -> dict[str, dict[str, Any]]:
    """Each factor -> {label: value}.

    A scalar is one level labelled by its own value. A list is swept and
    self-labelling. A dict is swept with explicit labels — which is the one that
    trips people: `{"allowed_domains": ["x.com"]}` reads as ONE CONDITION LABELLED
    "allowed_domains", not as a held complex value. To hold a dict constant, name
    it: `{"x_only": {"allowed_domains": ["x.com"]}}`.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, value in block.items():
        if key in META_KEYS:
            continue
        if isinstance(value, dict):
            out[key] = dict(value)
        elif isinstance(value, list):
            out[key] = {str(v): v for v in value}
        else:
            out[key] = {str(value): value}
    return out


def expand_conditions(block: dict) -> list[dict]:
    """One provider block -> [{"label", "values"}]. Product, exclude, then include.

    **Labels contain exactly what varies, regardless of how it was written.** A
    held factor never appears, even when it was authored as a named dict — so
    `{"search": {"strict": {...}}}` alone yields `base`, not `strict`.

    That is deliberate and it surprises people. A label exists to distinguish
    conditions within a study; a value present in every record distinguishes
    nothing, and including it means an unrelated constant changing would appear
    to change the conditions. The authored name is not lost — it stays in the
    spec file, which is written into the run directory, and in `config_resolved`
    on every record.
    """
    levels = _levels(block)
    varying = [k for k, v in levels.items() if len(v) > 1]
    includes = block.get("include", [])
    use_grid = bool(varying) or not includes

    conditions: list[dict] = []
    keys = list(levels)

    if use_grid:
        for combo in itertools.product(*[list(levels[k].items()) for k in keys]):
            labelled = dict(zip(keys, combo))
            if _excluded(labelled, block.get("exclude", [])):
                continue
            label = "|".join(labelled[k][0] for k in varying) if varying else "base"
            conditions.append({"label": label,
                               "values": {k: v for k, (_, v) in labelled.items()}})

    for entry in includes:
        entry = dict(entry)
        label = entry.pop("label", None) or "include"
        missing = [k for k in varying if k not in entry]
        if missing:
            raise InvestigationError(
                f"include entry {label!r} omits swept factor(s) {missing}. An "
                f"include cell must pin every factor the grid sweeps, or its "
                f"position in the design is undefined.")
        base = {k: next(iter(v.values())) for k, v in levels.items() if len(v) == 1}
        conditions.append({"label": label, "values": {**base, **entry}})

    if not conditions:
        raise InvestigationError(
            "no conditions survive — `exclude` removed every cell in the grid")
    return conditions


def _excluded(labelled: dict, rules: list[dict]) -> bool:
    """Does this cell match any exclude rule. A list value means 'any of these'.

    An exclude naming a factor that is not in the grid raises. Silently matching
    nothing is the worst available behaviour: the study runs with cells the
    author believed were removed, every arm looks legitimate, and the only
    symptom is a corpus larger than expected — which nobody checks.
    """
    for rule in rules:
        unknown = [k for k in rule if k not in labelled]
        if unknown:
            raise InvestigationError(
                f"exclude rule names factor(s) {unknown} that are not in this "
                f"provider block. Known factors: {sorted(labelled)}. An exclude "
                f"that matches nothing removes nothing, and the study would run "
                f"cells you meant to drop.")
        hit = True
        for k, want in rule.items():
            got = labelled[k][0]
            if got != want and not (isinstance(want, list) and got in want):
                hit = False
                break
        if hit:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Intent resolution
# ═══════════════════════════════════════════════════════════════════════════════

def resolve(model: str, passed: dict, on_unmet: str = "error"
            ) -> tuple[dict, dict, list[tuple[str, Any]]]:
    """Intent dict -> (resolved, config_passed, unmet).

    ORDER IS THE SPECIFICATION. Each step is numbered to match §5, and each
    number is a place where getting the sequence wrong produces a silent defect
    rather than an error.

    `config_passed` is returned unchanged so the record can carry BOTH what the
    spec asked for and what was actually sent — Inspect's `task_args` /
    `task_args_passed` distinction. Analysis then groups on the resolved config
    rather than on a condition label, which is what makes the label's
    provider-scoping harmless.
    """
    if on_unmet not in ("error", "exclude"):
        raise InvestigationError(
            f"on_unmet={on_unmet!r} is not valid. Two options, deliberately: "
            f"'error' refuses to run, 'exclude' drops the cell from the grid. A "
            f"third — run anyway and record the discrepancy — was considered and "
            f"removed: it produces an arm labelled `reasoning: off` that has "
            f"reasoning on, and a condition label that lies is what this whole "
            f"design is arranged against.")

    # (1) Unknown model raises. Never a family default: a guessed capability
    #     produces an arm whose condition is a fiction, which is worse than a
    #     failed load because it looks like data.
    caps = caps_for(model)

    provider = provider_of(model)
    unmet: list[tuple[str, Any]] = []
    resolved: dict[str, Any] = {}

    unrecognised: list[str] = []
    for intent, value in passed.items():
        # (2) provider-keyed escape hatches are merged last, at step (6)
        if intent in ("anthropic", "openai", "gemini"):
            continue
        if intent not in INTENTS and intent not in META_KEYS:
            unrecognised.append(intent)
        # (3) meta keys pass through untouched — they shape the grid, not the call
        if intent in META_KEYS:
            resolved[intent] = value
            continue

        # (4) CAPABILITY CHECK BEFORE MAPPING. An adaptive-thinking model gets
        #     adaptive regardless of the effort level asked for, so checking
        #     after mapping would map a value the model ignores.
        if not caps.supports(intent, value):
            unmet.append((intent, value))
            continue

        # (5) translate. The mapping itself is the provider's job — it knows its
        #     own vocabulary — so this records intent and `build()` applies it.
        resolved[intent] = value

    # (6) escape hatch merges LAST and WINS. It is the deliberate override;
    #     nothing should silently undo it.
    resolved.update(passed.get(provider, {}))

    # (7) guards run AFTER the escape hatch, because that is where a bad value is
    #     most likely to enter — the hatch is unvalidated by design.
    _guard_inert(model, caps, resolved)
    _guard_token_budget(model, caps, resolved)

    # An unrecognised intent is usually a typo, and the API would catch it — but
    # only once the battery is running, whereas load costs nothing. So this warns
    # rather than raising: a genuinely new provider parameter must still be
    # expressible, and refusing it would be the client-side validation this
    # design deliberately avoids. The warning names the near-miss because
    # `reasonning` rejected by a provider reads as an API problem.
    if unrecognised:
        import difflib
        import warnings
        for name in unrecognised:
            near = difflib.get_close_matches(name, sorted(INTENTS | META_KEYS), n=1)
            hint = f" — did you mean {near[0]!r}?" if near else ""
            warnings.warn(
                f"{name!r} is not a known intent{hint} It will be sent as-is and "
                f"the provider will decide. Put it in a {provider!r} block to "
                f"silence this if it is deliberate.",
                stacklevel=2)

    if unmet and on_unmet == "error":
        lines = [f"  {intent}={value!r}: {caps.why_unsupported(intent, value)}"
                 for intent, value in unmet]
        raise UnmetIntentError(
            f"{model} cannot honour:\n" + "\n".join(lines) + "\n"
            f"  Options: write what this provider actually does (conditions are "
            f"per-provider for exactly this reason), or set "
            f"on_unmet='exclude' to drop the cell from the grid.\n"
            f"  Running anyway is not offered — it would produce an arm whose "
            f"condition label does not describe it.")

    return resolved, dict(passed), unmet


def _guard_inert(model: str, caps: ModelCaps, resolved: dict) -> None:
    """Catch dead sampling parameters that entered via the ESCAPE HATCH.

    A top-level `temperature` never reaches here: it is a recognised intent,
    `supports()` returns False, and it routes through the unmet path which
    explains why per provider. This guard exists for the one route that bypasses
    all of that — the provider block, which is unvalidated by design and is
    therefore where a bad value is most likely to enter.

    Why guard at all, when the design says not to duplicate the API's
    validation: on one provider there IS no API validation to duplicate. Gemini
    accepts a sampling parameter and discards it, measured — `temperature: 0.0`
    gave six distinct answers across six repetitions naming three different
    products. Nothing downstream would ever object.
    """
    if caps.sampling_meaningful:
        return
    for param in ("temperature", "top_p", "top_k", "topP", "topK"):
        if param in resolved:
            raise InvestigationError(
                f"{param!r} on {model}: {caps.why_unsupported('temperature', None)}. "
                f"Remove it, or put it in a provider block deliberately if the "
                f"record should show it was sent.")


def _guard_token_budget(model: str, caps: ModelCaps, resolved: dict) -> None:
    """Warn when max_tokens must cover thinking as well as output.

    On a combined-budget provider, thinking has been measured taking ~96% of the
    allowance. A 4096 budget where thinking runs 4,000 leaves nothing for the
    answer, and the truncation reads as a model failure rather than a config one.
    """
    if not caps.combined_token_budget:
        return
    budget = resolved.get("max_tokens")
    if isinstance(budget, int) and budget < 8192:
        raise InvestigationError(
            f"max_tokens={budget} on {model}, where the budget covers thinking "
            f"AND output — thinking has been measured taking ~96% of it. Under "
            f"8192 the answer is likely to be truncated in a way that looks like "
            f"a model failure. Raise it, or set it in the provider block if the "
            f"truncation is the point.")


# ═══════════════════════════════════════════════════════════════════════════════
# Files
# ═══════════════════════════════════════════════════════════════════════════════

def load_investigation_dict(path: str | pathlib.Path) -> tuple[dict, str]:
    """Read and validate a spec file. Returns (spec, sha256).

    The sha is of the file bytes, so a run records the exact spec that produced
    it even if the file is later edited. Separate from `probe_hash`, which
    answers a different question: the file sha says "was this the same file",
    the probe hash says "was this the same instrument".
    """
    p = pathlib.Path(path)
    raw = p.read_bytes()
    spec = json.loads(raw)
    validate_investigation(spec)
    return spec, hashlib.sha256(raw).hexdigest()
