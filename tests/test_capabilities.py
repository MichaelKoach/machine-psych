"""Capability layer tests, including the two mechanical sweeps.

Most of these assert behaviour. Two do not: `test_capability_fields_consumed` and
`test_no_unread_measures` are **greps over the source**, and they exist because
the audits that found real defects in the predecessor modules were greps rather
than readings. An unread field and a fill on a Tier 3 column are both invisible
to review — there is nothing on the page to look at.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import fields

import pytest
import requests

from machine_psych.capabilities import (
    CAPABILITIES,
    CONSUMED_BY,
    DELIBERATELY_UNREAD,
    ENUMS,
    REQUIRED_FIELDS,
    ModelCaps,
    UnknownModelError,
    caps_for,
    known_models,
    model_of,
    provider_of,
)
from machine_psych.providers.base import Provider

PKG = pathlib.Path(__file__).resolve().parent.parent / "machine_psych"
PROVIDERS = {"anthropic", "openai", "gemini"}


# ═══════════════════════════════════════════════════════════════════════════════
# supports() — the value matters, not just the key
# ═══════════════════════════════════════════════════════════════════════════════

def test_supports_discriminates_within_a_provider():
    """Sonnet 5 can disable thinking and Fable 5 cannot.

    This is the case that justifies keying capabilities per MODEL rather than per
    provider. At provider granularity the difference is invisible, and a
    reasoning-off arm on Fable would run with thinking on while carrying a
    condition label saying otherwise.
    """
    assert caps_for("anthropic/claude-sonnet-5").supports("reasoning", "off")
    assert not caps_for("anthropic/claude-fable-5").supports("reasoning", "off")


def test_supports_takes_the_value_not_the_key():
    """`reasoning` is supported on Gemini; `reasoning: "off"` is not."""
    g = caps_for("gemini/gemini-3.7-flash")
    assert g.supports("reasoning", "high")
    assert not g.supports("reasoning", "off")


def test_inert_parameters_are_unsupported_everywhere():
    """Sampling is dead on all three, but only Gemini fails silently.

    Anthropic and OpenAI reject temperature outright, so the API catches the
    mistake. Gemini accepts and ignores it — measured, temperature 0.0 gave six
    distinct answers across six repetitions. That is the case client-side
    validation exists for.
    """
    for model in known_models():
        for p in ("temperature", "top_p", "top_k"):
            assert not caps_for(model).supports(p, 0.5), f"{model} {p}"


def test_why_unsupported_is_specific_for_every_false():
    """A generic 'not supported' has, in practice, read as a broken harness.

    The reason is what tells a study designer what to do instead.
    """
    probes = [("reasoning", "off"), ("verbosity", "low"), ("search", True),
              ("max_tool_calls", 3), ("temperature", 0.5), ("top_p", 0.9)]
    for model in known_models():
        caps = caps_for(model)
        for intent, value in probes:
            if not caps.supports(intent, value):
                msg = caps.why_unsupported(intent, value)
                assert not msg.startswith(f"{intent!r} is not supported"), (
                    f"{model} {intent}={value!r} falls through to the generic message")


# ═══════════════════════════════════════════════════════════════════════════════
# Lookup: unknown models raise, and the message is the point
# ═══════════════════════════════════════════════════════════════════════════════

def test_unknown_model_raises_rather_than_defaulting():
    with pytest.raises(UnknownModelError):
        caps_for("openai/gpt-99")


def test_unknown_model_message_survives_formatting():
    """UnknownModelError must not inherit from KeyError.

    KeyError.__str__ is repr() of its argument, which collapses a formatted
    multi-line message into one escaped line. The message names what is known and
    how to add a model; that is the whole reason the exception exists.
    """
    try:
        caps_for("openai/gpt-99")
    except UnknownModelError as e:
        text = str(e)
    assert "\n" in text, "message was flattened — check the base exception class"
    assert "probes establish" in text, "message does not say how to add a model"
    # KeyError's repr escapes newlines into literal backslash-n; a real newline
    # is the discriminator. An earlier version of this test asserted the message
    # did not START with a quote, which fails on the correct message too — the
    # model name is legitimately quoted. The test was wrong, not the code.
    assert "\\n" not in text, "newlines were escaped — KeyError semantics leaked"


def test_bare_model_name_is_rejected():
    """Names are 'provider/model' so the provider is explicit, never inferred
    from a string pattern that happens to look like one vendor's."""
    with pytest.raises(UnknownModelError):
        provider_of("claude-sonnet-5")
    assert provider_of("anthropic/claude-sonnet-5") == "anthropic"
    assert model_of("anthropic/claude-sonnet-5") == "claude-sonnet-5"


# ═══════════════════════════════════════════════════════════════════════════════
# The measured values, asserted independently of the table
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model,field,expected", [
    # Each of these was established by running requests, and each one being wrong
    # would silently mis-shape a corpus rather than raising.
    ("gemini/gemini-3.7-flash", "citation_offsets", "byte"),
    ("openai/gpt-5.6-sol", "citation_offsets", "char"),
    ("anthropic/claude-sonnet-5", "citation_offsets", "quoted"),
    ("anthropic/claude-sonnet-5", "retrieval_set", True),
    ("openai/gpt-5.6-sol", "retrieval_set", False),
    ("gemini/gemini-3.7-flash", "retrieval_set", False),
    ("gemini/gemini-3.7-flash", "in_tok_processed_field", "raw_prompt_token"),
    ("openai/gpt-5.6-sol", "in_tok_processed_field", "input_tokens"),
    ("gemini/gemini-3.7-flash", "thinking_in_output_tokens", False),
    ("anthropic/claude-sonnet-5", "thinking_in_output_tokens", True),
    ("anthropic/claude-sonnet-5", "answer_extraction", "heuristic"),
    ("openai/gpt-5.6-sol", "answer_extraction", "structural"),
    ("gemini/gemini-3.7-flash", "search_conditional", True),
    ("anthropic/claude-sonnet-5", "search_conditional", False),
    ("gemini/gemini-3.7-flash", "combined_token_budget", True),
    ("openai/gpt-5.6-sol", "verbosity", True),
    ("gemini/gemini-3.7-flash", "verbosity", False),
])
def test_measured_capability(model, field, expected):
    assert getattr(caps_for(model), field) == expected


def test_only_anthropic_exposes_the_retrieval_set():
    """The uncited candidates exist on one provider.

    That is what makes 'what distinguishes cited from uncited sources' answerable
    at all, and it is why the scraping layer needs Anthropic specifically.
    """
    have = {m for m in known_models() if caps_for(m).retrieval_set}
    assert have and all(m.startswith("anthropic/") for m in have)


def test_readable_reasoning_is_no_longer_gemini_only():
    """It was, and the claim outlived the fact.

    "Only Gemini exposes readable reasoning" was written into the capability
    table, the design document and the project notes, and justified a Tier 3
    column on the grounds that one provider had something the others could not
    do. OpenAI's `reasoning.summary` was validated as a parameter at the time and
    returned EMPTY arrays in blocking mode; on 2026-08-27 it populates.

    This test asserts the current measurement rather than the exclusivity,
    because the exclusivity is the thing that changed and an assertion phrased
    around it would have to be rewritten again next time.
    """
    have = {m.split("/")[0] for m in known_models()
            if caps_for(m).readable_reasoning}
    assert have == {"gemini", "openai"}
    assert not caps_for("anthropic/claude-sonnet-5").readable_reasoning, (
        "Anthropic thinking blocks carry a signature and an empty string")


def test_tier3_columns_can_span_providers():
    """A Tier 3 column is not a synonym for one provider.

    `readable_reasoning` was true on one provider and is now true on two. The
    parser gates on the capability rather than on the provider name, which is why
    that change was a one-line table edit — a parser branching on `if provider ==
    "gemini"` would have silently kept extracting nothing from OpenAI.
    """
    import inspect

    from machine_psych.providers import Anthropic
    src = inspect.getsource(Anthropic)
    assert 'readable_reasoning' in src, "parser must gate on the capability"
    assert '== "gemini"' not in src and "== 'gemini'" not in src, (
        "parser must not branch on provider identity")


# ═══════════════════════════════════════════════════════════════════════════════
# Mechanical sweeps — the checks that reading cannot do
# ═══════════════════════════════════════════════════════════════════════════════

def test_every_capability_field_has_an_assigned_consumer():
    """A field with no reader becomes stale data that looks load-bearing.

    Two constants in the predecessor modules — `EXPLICIT_DEFAULTS` and the
    recovered enumerations — sat unread while appearing to be part of the
    machinery, and one of them was a second source of truth that would have
    silently diverged from the builder it documented.

    Most fields here have no reader YET, because the consuming modules are not
    built. CONSUMED_BY records who owes each one, so 'not yet' cannot drift into
    'never' unnoticed.
    """
    declared = {f.name for f in fields(ModelCaps)}
    assert declared == set(CONSUMED_BY), (
        f"unassigned: {sorted(declared - set(CONSUMED_BY))}, "
        f"phantom: {sorted(set(CONSUMED_BY) - declared)}")


def test_assigned_consumers_actually_read_their_fields():
    """Once a module exists, it must read what it was assigned.

    Skips modules that are not built yet; that is the point — this test tightens
    automatically as the package fills in, rather than needing to be remembered.
    """
    unread = []
    for field_name, owner in CONSUMED_BY.items():
        if owner.startswith(("DOCUMENTATION", "PENDING")):
            continue
        module = owner.split()[0]                       # 'spec.resolve' -> 'spec'
        # A named function that does not exist yet is a pending consumer, not a
        # broken assignment — the estimator is in the roadmap, not the package.
        head, _, member = module.partition(".")
        path = PKG / (head + ".py")
        if path.exists() and member and f"def {member}" not in path.read_text():
            continue
        if module.startswith("providers"):
            path = PKG / "providers"
            if not any(p.name not in ("__init__.py", "base.py")
                       for p in path.glob("*.py")):
                continue
            src = "".join(p.read_text() for p in path.glob("*.py"))
        else:
            if not path.exists():
                continue
            src = path.read_text()
        if f".{field_name}" not in src:
            unread.append((field_name, owner))
    assert not unread, f"assigned but not read: {unread}"


def test_no_fillna_on_tier3_columns():
    """A fill is a claim, and on a capability column the claim is false.

    `fillna(0)` on a token sum says a failed record contributed nothing — true.
    `fillna(False)` on `grounded` says a provider declined to do something it
    cannot do — false, and it destroys the distinction between 'cannot' and 'did
    not' that the whole Tier 3 design turns on. The predecessor Gemini loader did
    exactly this, harmlessly within one provider and wrongly in a merged corpus.
    """
    tier3 = ["grounded", "thought_text", "n_thought_steps", "n_sources_retrieved",
             "n_answer_blocks", "n_process_blocks"]
    offenders = []
    for path in PKG.rglob("*.py"):
        src = path.read_text()
        for col in tier3:
            for m in re.finditer(rf'{col}\s*[^\n]{{0,40}}fillna\(', src):
                offenders.append(f"{path.name}:{src[:m.start()].count(chr(10)) + 1} {col}")
    assert not offenders, f"fillna on Tier 3: {offenders}"


def test_deliberately_unread_is_provider_scoped():
    """Keys are 'provider/field' so the same field name on two providers can be
    excluded for different reasons — or excluded on one and read on another."""
    for key, reason in DELIBERATELY_UNREAD.items():
        assert "/" in key, f"{key} is not provider-scoped"
        assert key.split("/")[0] in PROVIDERS, f"{key} names an unknown provider"
        assert len(reason) > 20, f"{key} has no real reason: {reason!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Table integrity
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_entries_are_complete_modelcaps():
    for key, caps in CAPABILITIES.items():
        assert isinstance(caps, ModelCaps), key
        unset = [f for f in REQUIRED_FIELDS
                 if f not in ("domain_filter", "citation_offsets")
                 and getattr(caps, f) is None]
        assert not unset, f"{key} leaves {unset} unset"


def test_none_is_valid_for_cannot_fields():
    """None means the provider CANNOT, not that a field was forgotten.

    An earlier completeness check conflated the two and rejected every valid
    Anthropic entry — the absence-versus-unavailability distinction the design is
    built on, violated by the check enforcing it.

    `citation_offsets` used to be the example here and no longer is: fixture
    collection found Anthropic citations present after all, in a third form that
    carries the quote directly. The lesson is the reason this test kept its name
    — a None recorded from measurement is only as current as the measurement.
    """
    caps = caps_for("anthropic/claude-sonnet-5")
    assert caps.domain_filter is not None
    assert caps_for("gemini/gemini-3.7-flash").domain_filter == "prompt"


def test_three_citation_mechanisms_not_two():
    """Byte offsets, character offsets, and quoted text are three mechanisms.

    'quoted' is the easiest to work with and happens to be what the analysis
    layer wants — the rule is compare quotes, not offsets, and this provider
    supplies the quote with no arithmetic to get wrong.
    """
    assert caps_for("gemini/gemini-3.7-flash").citation_offsets == "byte"
    assert caps_for("openai/gpt-5.6-sol").citation_offsets == "char"
    assert caps_for("anthropic/claude-sonnet-5").citation_offsets == "quoted"


def test_only_anthropic_reports_page_age():
    """How fresh the sources behind a recommendation are is a direct question for
    this research, and one provider answers it."""
    have = {m for m in known_models() if caps_for(m).page_age}
    assert have and all(m.startswith("anthropic/") for m in have)


def test_no_alias_keys():
    """Aliases are not keys. `gpt-5.6` resolves to `-sol`, and the three variants
    named substantially different companies on the same prompt — only one
    appeared in all of them. An alias key would make the served model an
    uncontrolled variable.
    """
    for key in CAPABILITIES:
        assert key != "openai/gpt-5.6", "gpt-5.6 is an alias over three variants"
        assert key != "gemini/gemini-flash-latest", "latest-tags float"


def test_capability_keys_are_provider_scoped():
    for key in CAPABILITIES:
        assert key.count("/") == 1 and key.split("/")[0] in PROVIDERS, key


def test_measured_on_is_a_date():
    for key, caps in CAPABILITIES.items():
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", caps.measured_on), key


def test_enums_cover_every_provider():
    """Recorded enumerations are the baseline compare_capabilities diffs against.

    They have gone stale once already: `reasoning.context` was written down as
    all_turns|none|previous_turn and is actually auto|current_turn|all_turns, and
    a spec was written against the wrong values before anyone noticed.
    """
    assert set(ENUMS) == PROVIDERS
    for provider, mapping in ENUMS.items():
        for param, values in mapping.items():
            assert isinstance(values, tuple) and values, f"{provider}.{param}"


# ═══════════════════════════════════════════════════════════════════════════════
# Transport exception classification
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("exc,expected", [
    (requests.exceptions.Timeout("t"), "unavailable"),
    (requests.exceptions.ConnectionError("c"), "unavailable"),
    (requests.exceptions.ChunkedEncodingError("e"), "unavailable"),
    (requests.exceptions.InvalidURL("u"), "error"),
    (requests.exceptions.MissingSchema("s"), "error"),
    (ValueError("v"), "error"),
])
def test_transport_exceptions_classified(exc, expected):
    """A network blip must retry; a malformed URL must not.

    The spec's runner said `if exc: break`, which treats every transport
    exception as permanent. On a forty-minute battery against a provider with a
    measured 1-in-6 transient failure rate, a single connection reset would lose
    a record that a retry recovers — which is the failure retry exists for.
    """
    assert Provider.classify_exception(exc) == expected


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit()])
def test_interrupts_are_not_classified(exc):
    """Classifying an interrupt would swallow it.

    Returning 'error' looks safe — it means "do not retry" — and is not. A caller
    wrapping the retry loop in `except BaseException` would break the loop,
    record a failed call, and CONTINUE THE RUN after the user pressed stop.
    Re-raising sends it to the handler that returns partial results.
    """
    with pytest.raises(type(exc)):
        Provider.classify_exception(exc)


def test_the_contract_is_what_the_runner_actually_calls():
    """Every provider method the runner calls must exist on the ABC.

    This used to parse the pseudocode in MERGE_SPEC §6b and compare it to the
    contract — a check that the DESIGN DOCUMENT and the code agreed, which was
    the right check while the code was being built from the document.

    The code is built and directly tested now, so the runner IS the truth and a
    document cannot contradict it. Reads the runner instead, which also means the
    check no longer skips when the spec is not alongside the package.
    """
    import ast

    from machine_psych import runner

    # Parsed, not grepped. A first version regexed `impl.(\w+)\(` and caught
    # `corpus.model.dropna().unique()` — the fifth time in this project a check
    # matched text rather than structure.
    source = pathlib.Path(runner.__file__).read_text()
    provider_vars = {"provider", "impl"}
    called = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in provider_vars):
            called.add(node.func.attr)
    available = {n for n in dir(Provider) if not n.startswith("_")}
    assert called <= available, (
        f"the runner calls methods the contract lacks: {sorted(called - available)}")
    assert called, "no provider calls found — the regex has drifted from the code"


# ═══════════════════════════════════════════════════════════════════════════════
# Package structure — from the post-merge audit
# ═══════════════════════════════════════════════════════════════════════════════

def test_every_module_imports_standalone():
    """No module may depend on the package __init__ being loaded first.

    `runner.py` had `from . import provenance` at module level and worked only
    because that function happened to be defined ABOVE the submodule imports in
    `__init__.py`. Moving it thirty lines down would have stopped the package
    importing entirely — a working-by-accident arrangement that no test covered.

    Configuration and provenance now live in `paths`, a layer-0 module that
    imports nothing from the package.
    """
    import importlib
    for module in ("paths", "capabilities", "spec", "providers.base",
                   "providers.anthropic", "providers.openai", "providers.gemini",
                   "runner", "corpus", "analysis", "export"):
        importlib.import_module(f"machine_psych.{module}")


def test_no_module_imports_the_package_root():
    """The specific arrangement that made the above fragile."""
    import re
    for path in PKG.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        src = path.read_text()
        assert not re.search(r"^from \. import (?!paths\b)", src, re.MULTILINE), (
            f"{path.name} imports from the package root; import a MODULE instead")


def test_paths_module_has_no_package_dependencies():
    """Layer 0 by construction, not by convention."""
    import ast
    tree = ast.parse((PKG / "paths.py").read_text())
    relative = [n for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level and n.level > 0]
    assert not relative, "paths must import nothing from the package"


def test_configuration_has_one_source_of_truth():
    """`runner.set_base` is a facade over `paths`, not a second copy.

    An earlier version held the paths in `runner` and `corpus` reached back into
    that module to read them — a low-level module depending on a high-level one,
    which is what produced the cycle.
    """
    from machine_psych import paths, runner
    assert runner.set_base is paths.set_base
    assert not hasattr(runner, "RECORDS_DIR"), (
        "runner must read paths.RECORDS_DIR at call time, not hold its own")


def test_public_callables_are_documented():
    import machine_psych as mp
    undocumented = [n for n in mp.__all__
                    if callable(getattr(mp, n, None))
                    and not (getattr(mp, n).__doc__ or "").strip()]
    assert not undocumented


def test_paths_are_a_function_not_exported_constants():
    """`set_base` rebinds module globals.

    A constant imported into the package namespace would keep whatever it held at
    import time and silently report the wrong directory after a rebase — the same
    stale-value trap that `paths` exists to document, reintroduced by the
    convenience of re-exporting it.
    """
    import tempfile

    import machine_psych as mp
    assert "BASE" not in mp.__all__
    assert "RECORDS_DIR" not in mp.__all__

    tmp = tempfile.mkdtemp()
    mp.set_base(tmp)
    assert str(mp.where()["base"]) == tmp


def test_measured_on_reflects_actual_measurement():
    """The three models re-measured on 2026-08-28 carry that date; others do not.

    An edit set every model to the fixture-collection date, claiming currency for
    seven that were never re-measured. `measured_on` exists to say how stale an
    entry is, and a blanket update inverts it — the field then reports when
    someone last touched the file rather than when anyone last checked the API.
    """
    # The three models this project actually dispatches to, each carrying the
    # date it was last CHECKED. They diverge because they are checked when
    # something prompts it, not on a schedule — which is the honest state and
    # the reason this field exists.
    remeasured = {
        # `thinking.type: enabled` is valid and requires budget_tokens
        "anthropic/claude-sonnet-5": "2026-09-05",
        "openai/gpt-5.6-sol": "2026-09-05",
        # per-model probe: `max` is NOT supported here, though the shared block
        # claimed it
        "openai/gpt-5.5": "2026-09-05",
        # per-model probe: `minimal` is NOT supported here, though the schema
        # lists it — and the shakedown sent it
        "gemini/gemini-3.7-flash": "2026-09-05",
    }
    for model in known_models():
        date = caps_for(model).measured_on
        if model in remeasured:
            assert date == remeasured[model], f"{model} date is stale"
        else:
            assert date < "2026-08-28", (
                f"{model} claims a measurement date it never had")


def test_error_does_not_name_an_unbuilt_script():
    """The message told you to run `tests/characterise.py`, which does not exist.

    That is the unread-constant pattern applied to a procedure: a table
    documenting how to add a model by naming an imaginary file. It now describes
    the four probes and points at the spec, and this test fails if the script is
    named again before it is written.
    """
    try:
        caps_for("openai/gpt-99")
    except UnknownModelError as e:
        text = str(e)
    script = PKG.parent / "tests" / "characterise.py"
    if not script.exists():
        assert "characterise.py" not in text, (
            "the error names a script that has not been built")


def test_reasoning_levels_are_per_model_not_inherited_blindly():
    """The schema and the model disagree, and the model wins.

    Measured 2026-09-05: a BOGUS value returns the API-wide list, identical on
    every model; a PLAUSIBLE value returns the per-model set, and those differ.
    Every enum in this table was recovered by the bogus technique, so every one
    describes the schema — and three of seven parameters probed turned out to
    have per-model restrictions it could not see.

    The concrete cost: the shakedown sent `reasoning: minimal` to
    gemini-3.7-flash, which rejects it. That study failed on an unrelated bug
    first, so running it never revealed this.
    """
    assert "minimal" not in caps_for("gemini/gemini-3.7-flash").reasoning_levels
    assert "minimal" in ENUMS["gemini"]["generation_config.thinking_level"], (
        "the SCHEMA does list it — that is the whole point of the distinction")

    assert "max" not in caps_for("openai/gpt-5.5").reasoning_levels
    assert "max" in caps_for("openai/gpt-5.6-sol").reasoning_levels
    assert "max" in ENUMS["openai"]["reasoning.effort"]


def test_a_spec_cannot_request_a_level_the_model_rejects():
    """`resolve` must refuse a level absent from THIS model's set.

    Otherwise the request is built, sent, and rejected by the API — a spec error
    surfacing as a run failure, thirty seconds and one call later than it needed
    to.
    """
    from machine_psych.spec import UnmetIntentError, resolve
    try:
        resolve("gemini/gemini-3.7-flash",
                {"reasoning": "minimal", "max_tokens": 16384})
    except (UnmetIntentError, Exception) as exc:
        assert "minimal" in str(exc), (
            "rejected, but the message does not name the offending level")
    else:
        raise AssertionError(
            "accepted `minimal` on a model that rejects it — the spec layer is "
            "not reading reasoning_levels")


# ═══════════════════════════════════════════════════════════════════════════════
# From the 2026-09-05 simplification audit
# ═══════════════════════════════════════════════════════════════════════════════

def test_incomplete_capability_entries_are_rejected():
    """This was a `__init_subclass__` hook on the Provider base class, with ZERO
    test coverage — which is itself a signal it was not earning its place.

    It ran per subclass over a filtered VIEW of this same dict (the objects are
    literally identical), so it validated data it did not own, three times, in a
    file that does not define it. One loop where the data lives does the job.
    """
    import dataclasses

    from machine_psych.capabilities import _validate_capabilities

    good = CAPABILITIES["anthropic/claude-sonnet-5"]
    CAPABILITIES["anthropic/__broken__"] = dataclasses.replace(
        good, reasoning_off=None)
    try:
        with pytest.raises(TypeError, match="reasoning_off"):
            _validate_capabilities()
    finally:
        del CAPABILITIES["anthropic/__broken__"]


def test_a_key_without_its_provider_prefix_is_rejected():
    import dataclasses

    from machine_psych.capabilities import _validate_capabilities

    CAPABILITIES["bare-model-name"] = dataclasses.replace(
        CAPABILITIES["anthropic/claude-sonnet-5"])
    try:
        with pytest.raises(TypeError, match="provider/model"):
            _validate_capabilities()
    finally:
        del CAPABILITIES["bare-model-name"]


def test_nullable_capability_fields_are_allowed_to_be_none():
    """`None` is LEGITIMATE for `citation_offsets` — Anthropic has no offsets at
    all, and "this provider cannot" is exactly what None means.

    An earlier version of this check treated None as missing and rejected every
    valid Anthropic entry.
    """
    from machine_psych.capabilities import _validate_capabilities
    assert caps_for("anthropic/claude-sonnet-5").citation_offsets == "quoted"
    _validate_capabilities()   # the real table, which contains Nones, passes


def test_the_readme_describes_the_directories_that_exist():
    """Docs drift silently and nobody notices until someone follows them.

    Two gaps found in the 2026-09-05 audit: `drift/` appeared in neither root
    doc, and nothing said that `pip install` excludes it — which produced a real
    `ModuleNotFoundError` in a live session.
    """
    root = PKG.parent
    readme = (root / "README.md").read_text()
    setup = (root / "SETUP.md").read_text()

    assert "drift/" in readme, "the README does not mention the drift directory"
    assert "not installed" in readme.lower() or "NOT installed" in readme
    assert "pip install" in setup and "drift" in setup, (
        "SETUP does not warn that pip install excludes drift/")

    for named in ("machine_psych/", "tests/"):
        assert named in readme, f"{named} is not described"


def test_provider_names_are_derived_not_listed():
    """`spec.resolve` hardcoded `("anthropic", "openai", "gemini")` to recognise
    escape-hatch keys, so a FOURTH provider's hatch would be rejected as an
    unrecognised intent — silently, in the one place that had to be edited to add
    a provider and gave no sign of it.

    Six audit passes looked for things that were too general; none asked whether
    anything was too specific.
    """

    from machine_psych.capabilities import known_providers

    assert known_providers() == {"anthropic", "openai", "gemini"}

    source = (PKG / "spec.py").read_text()
    tree = __import__("ast").parse(source)
    for node in __import__("ast").walk(tree):
        if (isinstance(node, __import__("ast").Constant)
                and isinstance(node.value, str)
                and node.value in ("anthropic", "openai", "gemini")):
            line = source.split("\n")[node.lineno - 1].strip()
            assert line.startswith("#"), (
                f"spec.py names a provider in CODE at line {node.lineno}: {line}")


def test_an_unknown_provider_says_so_rather_than_listing_everything():
    """One message for both cases lied.

    With an unknown PROVIDER, `known_models(prov)` returns nothing and the
    fallback printed every model from every provider under the heading "Known
    for perplexity" — a misleading message on the exact path someone adding a
    provider takes.
    """
    try:
        caps_for("perplexity/sonar-pro")
    except UnknownModelError as exc:
        text = str(exc)
    assert "No provider 'perplexity'" in text
    assert "known providers:" in text
    assert "gpt-5.6-sol" not in text, (
        "an unknown provider was answered with another provider's models")

    try:
        caps_for("openai/gpt-99")
    except UnknownModelError as exc:
        text = str(exc)
    assert "Known for openai" in text
    assert "gpt-5.6-sol" in text, "a known provider should list its siblings"
