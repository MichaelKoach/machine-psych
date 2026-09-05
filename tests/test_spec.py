"""Spec layer tests.

The `resolve()` tests are ordered to match the numbered steps in the module,
because with resolution the ORDER is the specification — a capability check after
mapping, or a guard before the escape hatch, produces a silent defect rather than
an error.
"""

from __future__ import annotations

import warnings

import pytest

from machine_psych.capabilities import UnknownModelError, caps_for, known_models
from machine_psych.spec import (
    InvestigationError, UnmetIntentError, expand_conditions, join_text,
    probe_hash, prompt_hash, resolve, validate_investigation,
)


def _spec(**over):
    base = {
        "investigation_id": "t",
        "metadata": {"name": "t"},
        "studies": [{
            "study_id": "s",
            "probes": [{"probe_id": "p", "prompt_paths": [["What is X?"]]}],
            "providers": {"anthropic/claude-sonnet-5": {"repetitions": 1}},
        }],
    }
    base.update(over)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# join_text
# ═══════════════════════════════════════════════════════════════════════════════

def test_join_text_handles_the_json_newline_convention():
    """JSON strings cannot contain literal newlines, so a long prompt is a list."""
    assert join_text(["a", "b"]) == "a\nb"
    assert join_text("a\nb") == "a\nb"
    assert join_text(None) is None


# ═══════════════════════════════════════════════════════════════════════════════
# probe_hash — instrument identity
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_hash_ignores_cosmetic_changes():
    """Renaming a probe or rewriting its rationale does not change the instrument.

    If it did, the hash would answer 'did anything about this probe change'
    rather than 'is this the same question', and the second is what a corpus
    needs to know when comparing two runs.
    """
    a = {"probe_id": "p", "prompt_paths": [["What is X?"]], "system_prompt": None}
    b = {"probe_id": "RENAMED", "prompt_paths": [["What is X?"]],
         "system_prompt": None, "rationale": ["entirely rewritten"], "note": "n"}
    assert probe_hash(a) == probe_hash(b)


def test_probe_hash_catches_substantive_changes():
    a = {"probe_id": "p", "prompt_paths": [["What is X?"]], "system_prompt": None}
    assert probe_hash(a) != probe_hash({**a, "prompt_paths": [["What is Y?"]]})
    assert probe_hash(a) != probe_hash({**a, "system_prompt": "Be brief."})
    assert probe_hash(a) != probe_hash({**a, "prompt_paths": [["What is X?", "Why?"]]})


def test_probe_hash_is_form_independent():
    """A prompt written as a list and the same prompt written as one string are
    the same instrument, so they must hash the same."""
    listed = {"prompt_paths": [[["line one", "line two"]]]}
    joined = {"prompt_paths": [["line one\nline two"]]}
    assert probe_hash(listed) == probe_hash(joined)


def test_prompt_hash_joins_across_providers():
    """The mechanism the cross-provider design rests on.

    Three independent modules computing this found nine shared prompts across
    three corpora with no coordination and no shared identifier.
    """
    assert prompt_hash("same text") == prompt_hash("same text")
    assert prompt_hash("a") != prompt_hash("b")


# ═══════════════════════════════════════════════════════════════════════════════
# validate_investigation — structure only
# ═══════════════════════════════════════════════════════════════════════════════

def test_valid_spec_passes():
    assert validate_investigation(_spec())


@pytest.mark.parametrize("mutate,fragment", [
    (lambda s: s.pop("investigation_id"), "investigation_id"),
    (lambda s: s.update(studies=[]), "no studies"),
    (lambda s: s["studies"][0].pop("study_id"), "study_id"),
    (lambda s: s["studies"][0].update(probes=[]), "no probes"),
    (lambda s: s["studies"][0].pop("providers"), "providers"),
    (lambda s: s["studies"][0]["probes"][0].pop("prompt_paths"), "prompt_paths"),
    (lambda s: s["studies"][0]["probes"][0].update(prompt_paths=[[""]]), "empty"),
    (lambda s: s["studies"][0]["probes"][0].update(
        prompt_paths=[["a"], ["a", "b"]]), "rectangular"),
])
def test_structural_problems_raise(mutate, fragment):
    s = _spec()
    mutate(s)
    with pytest.raises(InvestigationError) as e:
        validate_investigation(s)
    assert fragment in str(e.value)


def test_ragged_probe_names_the_offending_paths():
    """A probe whose paths differ in turn count produces records that are not
    comparable turn-for-turn, which is the whole reason they are grouped."""
    s = _spec()
    s["studies"][0]["probes"][0]["prompt_paths"] = [["a"], ["a", "b"], ["a"]]
    with pytest.raises(InvestigationError) as e:
        validate_investigation(s)
    assert "path 1 has 2" in str(e.value)


def test_duplicate_ids_raise():
    s = _spec()
    s["studies"][0]["probes"].append({"probe_id": "p", "prompt_paths": [["y"]]})
    with pytest.raises(InvestigationError, match="duplicate probe_id"):
        validate_investigation(s)


def test_bare_provider_key_raises():
    """Names are 'provider/model' so the provider is explicit rather than
    inferred from a model name that happens to look like one vendor's."""
    s = _spec()
    s["studies"][0]["providers"] = {"claude-sonnet-5": {}}
    with pytest.raises(InvestigationError, match="provider/model"):
        validate_investigation(s)


def test_unknown_model_in_a_spec_raises_with_the_add_procedure():
    s = _spec()
    s["studies"][0]["providers"] = {"openai/gpt-99": {}}
    with pytest.raises(UnknownModelError, match="characterise"):
        validate_investigation(s)


def test_validation_does_not_check_parameter_legality():
    """Deliberately. Every provider rejects unknown parameters and names the
    offending field, and those rejections stay current for free. Duplicating them
    here would add a second thing to keep in sync with three moving APIs.
    """
    s = _spec()
    s["studies"][0]["providers"]["anthropic/claude-sonnet-5"] = {
        "some_future_parameter": "whatever"}
    assert validate_investigation(s)


# ═══════════════════════════════════════════════════════════════════════════════
# expand_conditions
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("block,n", [
    ({"model": "m", "max_tokens": 4096}, 1),
    ({"reasoning": ["off", "high"]}, 2),
    ({"reasoning": ["off", "high"], "verbosity": ["low", "med", "high"]}, 6),
    ({"search": {"off": False, "on": True}}, 2),
    ({"a": ["x", "y"], "b": ["1", "2"], "exclude": [{"a": "x", "b": "1"}]}, 3),
    ({"max_tokens": 4096, "include": [{"label": "p"}, {"label": "q"}]}, 2),
])
def test_expansion_counts(block, n):
    assert len(expand_conditions(block)) == n


def test_label_mentions_only_what_varies():
    """A label restating every held value is unreadable and, worse, changes when
    an unrelated constant changes — so two runs of the same design would appear
    to have different conditions."""
    got = expand_conditions({"model": "m", "max_tokens": 4096,
                             "reasoning": ["off", "high"]})
    assert [c["label"] for c in got] == ["off", "high"]
    assert got[0]["values"]["max_tokens"] == 4096      # held values still carried


def test_held_only_is_labelled_base():
    assert expand_conditions({"model": "m"})[0]["label"] == "base"


def test_dict_form_carries_the_value_and_names_the_levels():
    """A dict is the explicitly-labelled form, and its labels appear when it varies.

    This is the rule that trips people: `{"allowed_domains": ["x.com"]}` is ONE
    CONDITION LABELLED "allowed_domains", not a held complex value. To hold a
    complex value, name it: `{"x_only": {"allowed_domains": [...]}}`.
    """
    got = expand_conditions({"search": {"off": False, "strict": {"allowed_domains": ["x.com"]}}})
    assert [c["label"] for c in got] == ["off", "strict"]
    assert got[1]["values"]["search"] == {"allowed_domains": ["x.com"]}


def test_a_held_dict_does_not_appear_in_the_label():
    """Held factors never appear in labels, however they were written.

    A single-entry dict is held, so it yields `base` rather than its authored
    name. That surprises people and is deliberate: a label distinguishes
    conditions, and a value present in every record distinguishes nothing.
    Including it would also mean an unrelated constant changing appeared to
    change the design. The name survives in the spec file and in
    `config_resolved`.
    """
    got = expand_conditions({"search": {"strict": {"allowed_domains": ["x.com"]}}})
    assert got[0]["label"] == "base"
    assert got[0]["values"]["search"] == {"allowed_domains": ["x.com"]}


def test_held_dict_stays_held_when_something_else_varies():
    got = expand_conditions({"search": {"strict": {"allowed_domains": ["x.com"]}},
                             "reasoning": ["off", "high"]})
    assert [c["label"] for c in got] == ["off", "high"]
    assert all(c["values"]["search"] == {"allowed_domains": ["x.com"]} for c in got)


def test_include_must_pin_every_swept_factor():
    with pytest.raises(InvestigationError, match="omits swept factor"):
        expand_conditions({"a": ["x", "y"], "include": [{"label": "p"}]})


def test_exclude_removing_everything_raises():
    """An empty grid is a design error that would otherwise run zero calls and
    look like success."""
    with pytest.raises(InvestigationError, match="no conditions survive"):
        expand_conditions({"a": ["x", "y"], "exclude": [{"a": ["x", "y"]}]})


def test_meta_keys_are_not_swept():
    got = expand_conditions({"reasoning": ["off", "high"], "repetitions": 5,
                             "note": "a comment"})
    assert len(got) == 2
    assert "repetitions" not in got[0]["values"]


# ═══════════════════════════════════════════════════════════════════════════════
# resolve — the order is the specification
# ═══════════════════════════════════════════════════════════════════════════════

def test_1_unknown_model_raises_before_anything_else():
    with pytest.raises(UnknownModelError):
        resolve("openai/gpt-99", {"reasoning": "low"})


def test_2_provider_blocks_do_not_leak_as_intents():
    resolved, _, _ = resolve("openai/gpt-5.6-sol",
                             {"reasoning": "low", "anthropic": {"x": 1}})
    assert "anthropic" not in resolved


def test_3_meta_keys_pass_through():
    resolved, _, _ = resolve("openai/gpt-5.6-sol", {"repetitions": 5})
    assert resolved["repetitions"] == 5


def test_4_capability_check_precedes_mapping():
    """Checking after mapping would map a value the model ignores.

    The case: an always-adaptive model gets adaptive thinking regardless of the
    effort level asked for, so a mapping applied first would silently produce a
    request whose effort parameter has no effect while the condition label says
    it does.
    """
    with pytest.raises(UnmetIntentError):
        resolve("anthropic/claude-fable-5", {"reasoning": "off"})
    assert resolve("anthropic/claude-sonnet-5", {"reasoning": "off"})[0]


def test_6_escape_hatch_merges_last_and_wins():
    """The hatch is the deliberate override; nothing should silently undo it."""
    resolved, passed, _ = resolve(
        "openai/gpt-5.6-sol",
        {"reasoning": "low", "openai": {"reasoning": "overridden", "custom": 1}})
    assert resolved["reasoning"] == "overridden"
    assert resolved["custom"] == 1
    assert passed["reasoning"] == "low", "config_passed must record what was ASKED"


def test_7_guards_run_after_the_escape_hatch():
    """The hatch is unvalidated by design, so it is where a bad value is most
    likely to enter. A guard running before it would miss exactly those."""
    with pytest.raises(InvestigationError, match="temperature"):
        resolve("openai/gpt-5.6-sol", {"openai": {"temperature": 0.5}})


# ── the unmet policy ─────────────────────────────────────────────────────────

def test_unmet_raises_by_default():
    with pytest.raises(UnmetIntentError):
        resolve("gemini/gemini-3.7-flash",
                {"reasoning": "off", "max_tokens": 16384})


def test_unmet_message_says_what_to_do():
    """A bare 'unsupported' has, in practice, read as a broken harness."""
    try:
        resolve("gemini/gemini-3.7-flash", {"reasoning": "off", "max_tokens": 16384})
    except UnmetIntentError as e:
        text = str(e)
    assert "cannot be disabled" in text
    assert "on_unmet='exclude'" in text
    assert "Running anyway is not offered" in text


def test_exclude_returns_the_unmet_list_without_a_fallback():
    """The unmet intent is ABSENT from resolved, never set to a nearby value.

    A fallback is the warn-and-downgrade behaviour this design rejects: it
    produces an arm labelled `reasoning: off` that has reasoning on.
    """
    resolved, _, unmet = resolve("gemini/gemini-3.7-flash",
                                 {"reasoning": "off", "max_tokens": 16384},
                                 on_unmet="exclude")
    assert unmet == [("reasoning", "off")]
    assert "reasoning" not in resolved


def test_run_anyway_is_not_an_option():
    """LiteLLM's third option — call without the parameter — has no correct use
    here. It produces an arm whose condition label does not describe it."""
    with pytest.raises(InvestigationError, match="Two options"):
        resolve("openai/gpt-5.6-sol", {}, on_unmet="run")


# ── guards ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", known_models())
def test_sampling_parameters_are_refused_everywhere(model):
    kw = {"max_tokens": 16384}
    with pytest.raises(InvestigationError):
        resolve(model, {"temperature": 0.5, **kw})


def test_sampling_message_distinguishes_rejected_from_ignored():
    """A measured distinction that a single message would lose.

    Anthropic and OpenAI return 400 for a sampling parameter; Gemini takes it and
    discards it. Telling an OpenAI user their rejected parameter is being
    silently ignored is wrong, and it was — the first version of this guard used
    one message for all three.
    """
    def msg(model):
        try:
            resolve(model, {"temperature": 0.5, "max_tokens": 16384})
        except InvestigationError as e:
            return str(e)
    assert "rejected" in msg("openai/gpt-5.6-sol")
    assert "rejected" in msg("anthropic/claude-sonnet-5")
    assert "silently ignored" in msg("gemini/gemini-3.7-flash")


def test_combined_budget_guard():
    """On a provider where max_tokens covers thinking too, a small budget
    truncates the answer in a way that reads as a model failure."""
    with pytest.raises(InvestigationError, match="thinking"):
        resolve("gemini/gemini-3.7-flash", {"max_tokens": 4096})
    assert resolve("gemini/gemini-3.7-flash", {"max_tokens": 16384})[0]


def test_combined_budget_guard_does_not_fire_elsewhere():
    assert resolve("openai/gpt-5.6-sol", {"max_tokens": 4096})[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Unrecognised intents — warn at load, never raise
# ═══════════════════════════════════════════════════════════════════════════════

def test_typo_warns_with_a_suggestion():
    """A typo currently reaches the API, which rejects it — after the battery has
    started. Load costs nothing, so it is the right place to notice."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve("openai/gpt-5.6-sol", {"reasonning": "low"})
    assert caught and "reasoning" in str(caught[0].message)


def test_unrecognised_intent_is_still_sent():
    """Warn, never raise. A genuinely new provider parameter must remain
    expressible; refusing it would be exactly the client-side validation this
    design avoids, and provider APIs gain parameters between releases."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        resolved, _, _ = resolve("openai/gpt-5.6-sol", {"reasonning": "low"})
    assert resolved["reasonning"] == "low"


def test_known_intents_and_escape_hatches_are_silent():
    """A warning that fires on correct input trains people to ignore warnings."""
    for passed in ({"reasoning": "low", "repetitions": 3, "max_tokens": 4096},
                   {"openai": {"deliberately_unknown": 1}}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve("openai/gpt-5.6-sol", passed)
        assert not caught, f"spurious warning for {passed}"


# ═══════════════════════════════════════════════════════════════════════════════
# Silent-failure guards, from the comprehensive audit
# ═══════════════════════════════════════════════════════════════════════════════

def test_typod_exclude_raises_rather_than_matching_nothing():
    """The worst available behaviour is silence.

    An exclude naming a factor that is not in the grid matched nothing, so the
    study ran with cells the author believed were removed. Every arm looked
    legitimate; the only symptom was a corpus larger than expected, which nobody
    checks.
    """
    with pytest.raises(InvestigationError, match="not in this provider block"):
        expand_conditions({"a": ["x", "y"], "b": ["1", "2"],
                           "exclude": [{"aa": "x"}]})


def test_correct_excludes_still_work():
    assert len(expand_conditions({"a": ["x", "y"], "b": ["1", "2"],
                                  "exclude": [{"a": "x"}]})) == 2


def test_exclude_accepts_a_list_as_any_of_these():
    got = expand_conditions({"a": ["x", "y", "z"], "exclude": [{"a": ["x", "y"]}]})
    assert [c["label"] for c in got] == ["z"]


def test_per_provider_prompts_are_rejected():
    """Prompts are study-level so every provider is asked the same questions.

    Per-provider prompts would produce arms that cannot be compared, and the
    failure is silent: every join is on prompt_hash, so mismatched prompts read
    as MISSING DATA rather than as a design error. An earlier version of this
    validation hashed the probes and checked the result was non-empty — a check
    that could not fail, since probes are already validated upstream.
    """
    spec = {"investigation_id": "t", "studies": [{
        "study_id": "s",
        "probes": [{"probe_id": "p", "prompt_paths": [["q"]]}],
        "providers": {"anthropic/claude-sonnet-5": {"prompt_paths": [["other"]]}}}]}
    with pytest.raises(InvestigationError, match="study-level"):
        validate_investigation(spec)


def test_prompt_hash_is_exported():
    """The corpus loader joins records across providers with it, so it has to be
    part of the public surface rather than an internal helper."""
    import machine_psych.spec as module
    assert "prompt_hash" in module.__all__
