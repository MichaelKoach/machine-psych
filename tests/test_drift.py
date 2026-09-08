"""Drift detector tests.

Everything here runs offline. The tiers make network calls in use, but their
comparison logic, exit codes and refusal behaviour do not — and those are the
parts that decide whether the detector is trustworthy.

The tests are organised around the failure modes from the hypothetical sweep
rather than around the functions, because the failures are what the design is
for. Several assert that something is NOT reported, which matters as much as the
positives: a detector that reports everything is a detector nobody reads.

**The Printer tests below outlived the tier that used them.** `drift/tier3.py`
compared fixture shapes on a schedule and was cut — it sampled a stochastic
process once and could not tell a changed model from a different output, which
produced a false positive on its first real run. The Printer itself is sound and
is now the diagnostic used when a fixture refresh makes a test fail: `compare(
shape_of(old), shape_of(new))` answers "what changed in the shape".
"""

from __future__ import annotations

import json
import pathlib

import pytest

from drift import characterise as ch
from drift import tier1, tier2
from drift.printer import compare, print_response, shape_of

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load(provider: str, name: str) -> dict:
    return json.loads((FIXTURES / provider / f"{name}.json").read_text())["response"]


def copy(body: dict) -> dict:
    return json.loads(json.dumps(body))


# ═══════════════════════════════════════════════════════════════════════════════
# The noise problem — the failure that kills the system silently
# ═══════════════════════════════════════════════════════════════════════════════

def test_different_content_same_shape_is_silent():
    """THE test. If this fails the detector is useless in a fortnight.

    Re-collected responses differ constantly — different search results, different
    phrasing, different counts — because the subject is stochastic. A differ that
    reports all of it produces pages of output on every run and gets skimmed, and
    then a real change scrolls past unread.
    """
    one = shape_of([load("anthropic", "grounded_ok")])
    two = shape_of([load("anthropic", "multiturn_t0")])
    assert compare(one, two) == []


def test_a_count_moving_within_a_category_is_silent():
    """Three searches becoming four is not drift.

    Comparing the numbers would report it as loudly as three becoming zero, and
    only the category boundary carries signal.
    """
    base = load("anthropic", "grounded_ok")

    def with_searches(n):
        body = copy(base)
        body["usage"]["server_tool_use"]["web_search_requests"] = n
        return body

    assert compare(shape_of([with_searches(5)]), shape_of([with_searches(6)])) == []


def test_a_count_crossing_zero_is_reported():
    """And three becoming zero IS drift — a provider that stopped searching."""
    base = load("anthropic", "grounded_ok")

    def with_searches(n):
        body = copy(base)
        body["usage"]["server_tool_use"]["web_search_requests"] = n
        return body

    diffs = compare(shape_of([with_searches(5)]), shape_of([with_searches(0)]))
    assert any("zero" in d for d in diffs)


# ═══════════════════════════════════════════════════════════════════════════════
# The changes that actually happened
# ═══════════════════════════════════════════════════════════════════════════════

def test_catches_the_openai_tool_usage_removal():
    """The real regression: `usage.tool_usage` vanished and `n_queries` silently
    returned None on every record for a full session."""
    after = load("openai", "grounded_ok")
    before = copy(after)
    before["usage"]["tool_usage"] = {"web_search": {"num_requests": 5}}
    diffs = compare(shape_of([before]), shape_of([after]))
    assert any("tool_usage" in d and "REMOVED" in d for d in diffs)


def test_catches_a_type_change():
    """A field going int to string keeps its path and breaks every reader."""
    before = load("openai", "grounded_ok")
    after = copy(before)
    after["usage"]["input_tokens"] = "49041"
    assert compare(shape_of([before]), shape_of([after]))


def test_catches_a_new_block_type():
    """A new unit type is a capability appearing, and the most interesting kind
    of drift — it is how `code_execution_tool_result` would have announced
    itself."""
    before = load("anthropic", "grounded_ok")
    after = copy(before)
    after["content"].append({"type": "code_execution_tool_result"})
    diffs = compare(shape_of([before]), shape_of([after]))
    assert any("code_execution_tool_result" in d for d in diffs)


def test_catches_a_field_becoming_sometimes_absent():
    """The absent-vs-present distinction, which is how this project's providers
    signal zero."""
    body = load("gemini", "grounded_ok")
    stripped = copy(body)
    for step in stripped.get("steps", []):
        for content in step.get("content") or []:
            for ann in content.get("annotations") or []:
                ann.pop("start_index", None)
    assert compare(shape_of([body]), shape_of([stripped]))


# ═══════════════════════════════════════════════════════════════════════════════
# The Printer
# ═══════════════════════════════════════════════════════════════════════════════

def test_scrubbed_keys_keep_their_presence():
    """Scrubbing replaces the VALUE, never the key.

    Losing a field and having a field change are different events and must not
    look alike — if scrubbing deleted the key, a removed field would be invisible.
    """
    printed = print_response({"text": "anything at all", "type": "text"})
    assert set(printed) == {"text", "type"}
    assert printed["text"] == "<text>"
    assert printed["type"] == "text", "structural values are never scrubbed"


def test_long_unlisted_text_is_scrubbed_by_length():
    """The scrub list will always be incomplete.

    A 6,000-character answer under a key nobody listed is pure noise, so length
    is a second net beneath the name-based one.
    """
    printed = print_response({"some_new_field": "x" * 500})
    assert printed["some_new_field"].startswith("<long:")


def test_structural_keys_holding_dicts_do_not_break():
    """`caller` is a scalar on two providers and `{"type": "direct"}` on
    Anthropic. An earlier version returned structural values raw and crashed the
    schema builder on the dict."""
    printed = print_response({"caller": {"type": "direct"}})
    assert printed == {"caller": {"type": "direct"}}


def test_the_shape_has_both_structure_and_values():
    """Two parts, because seeding GenSON to collect values needs every path named
    in advance and a seed deep enough for three providers is combinatorial — the
    first attempt exhausted memory at depth five."""
    shape = shape_of([load("anthropic", "grounded_ok")])
    assert "structure" in shape and "values" in shape
    assert any("type" in path for path in shape["values"])


# ═══════════════════════════════════════════════════════════════════════════════
# Exit codes — incomplete must never read as clean
# ═══════════════════════════════════════════════════════════════════════════════

def test_tier1_reports_incomplete_without_a_key(capsys):
    """A run that could not look has found no differences, which must not read as
    a clean result."""
    from machine_psych import paths
    saved = dict(paths._API_KEYS)
    paths._API_KEYS.clear()
    try:
        assert tier1.run(["anthropic"]) == 2
    finally:
        paths._API_KEYS.update(saved)
    assert "INCOMPLETE" in capsys.readouterr().out


def test_tier2_reports_incomplete_without_a_key(capsys):
    from machine_psych import paths
    saved = dict(paths._API_KEYS)
    paths._API_KEYS.clear()
    try:
        assert tier2.run(["anthropic/claude-sonnet-5"]) == 2
    finally:
        paths._API_KEYS.update(saved)
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    assert "none were looked for" in out








# ═══════════════════════════════════════════════════════════════════════════════
# characterise
# ═══════════════════════════════════════════════════════════════════════════════

def test_characterise_refuses_to_guess_the_offset_unit():
    """Byte and character offsets are IDENTICAL in shape and differ only on
    non-ASCII text.

    Guessing produces spans that are silently wrong and drift further into the
    answer — measured wrong on 70 of 107 citations once. Unknown is the honest
    answer from one response.
    """
    for provider in ("openai", "gemini"):
        read = ch._read_grounded(load(provider, "grounded_ok"))
        assert read["citation_offsets"] is None

    anthropic = ch._read_grounded(load("anthropic", "grounded_ok"))
    assert anthropic["citation_offsets"] == "quoted", (
        "quoted IS decidable from shape — the text is right there")


def test_characterise_reports_the_budget_rather_than_concluding():
    """An earlier version decided `combined_token_budget` from one call and got
    Gemini wrong — its truncated response has thinking AND text, and it does have
    a combined budget.

    That was a verdict function computing a conclusion from insufficient
    evidence, which is the pattern this project has flagged four times.
    """
    read = ch._read_budget(load("gemini", "incomplete_no_output"), "gemini")
    assert read["combined_token_budget"] is None
    assert "thinking tokens" in read["_budget_observation"]


def test_rendered_block_marks_unmeasured_fields():
    """A defaulted capability produces an arm whose condition is a fiction, which
    is worse than a blank because it looks like data."""
    block = ch._render({"model": "openai/gpt-9",
                        "measured": {"reasoning_off": True, "retrieval_set": None},
                        "notes": []})
    assert "reasoning_off=True," in block
    assert "UNMEASURED" in block


def test_answer_extraction_works_on_all_three_providers():
    for provider in ("anthropic", "openai", "gemini"):
        text = ch._answer_text(load(provider, "grounded_ok"), provider)
        assert len(text) > 100, provider


# ═══════════════════════════════════════════════════════════════════════════════
# Probe construction
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("provider,model,parameter,expected_path", [
    ("anthropic", "claude-sonnet-5", "output_config.effort", ["output_config", "effort"]),
    ("openai", "gpt-5.6-sol", "reasoning.effort", ["reasoning", "effort"]),
    ("gemini", "gemini-3.7-flash", "generation_config.thinking_level",
     ["generation_config", "thinking_level"]),
])
def test_probe_bodies_nest_the_parameter_correctly(provider, model, parameter,
                                                   expected_path):
    body = tier2._body_with(provider, model, parameter, "BOGUS")
    node = body
    for part in expected_path:
        node = node[part]
    assert node == "BOGUS"


def test_response_side_enums_are_not_probed():
    """`error.type` and `role` describe what comes BACK. There is nothing to send,
    and sending something would probe a different parameter."""
    assert tier2._body_with("openai", "m", "error.type", "x") is None
    assert tier2._body_with("gemini", "m", "role", "x") is None


def test_valid_values_are_recovered_from_a_rejection():
    """The technique that recovered every enumeration in the table originally."""
    message = {"error": {"message":
               "Input should be 'minimal', 'low', 'medium', 'high' or 'xhigh'"}}
    assert set(tier2._values_in_error(message)) == {
        "minimal", "low", "medium", "high", "xhigh"}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — from the first real run, which produced 186 lines of noise
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def roster_at(tmp_path, monkeypatch):
    """A tier 1 with a controlled roster and a temporary baseline."""
    def setup(approved, live):
        monkeypatch.setattr(tier1, "ROSTER", tmp_path / "roster.json")
        (tmp_path / "roster.json").write_text(json.dumps({"anthropic": approved}))
        monkeypatch.setattr(tier1, "roster",
                            lambda p: (live if p == "anthropic" else [], None))
    return setup


BASE = ["claude-fable-5", "claude-opus-4-6", "claude-opus-5", "claude-sonnet-5"]


def test_the_catalogue_is_filtered_to_dispatchable_models():
    """The first real run diffed the whole catalogue — embeddings, TTS, image,
    video, Whisper, GPT-3.5 — against a table of ten text models, and printed 186
    lines of which about four mattered.

    That is the noise failure tier 3 was designed against, walked into by tier 1.
    """
    for model in ("text-embedding-3-large", "whisper-1", "tts-1-hd", "sora-2",
                  "gpt-realtime", "gpt-image-2", "omni-moderation-latest",
                  "veo-3.1-generate-preview", "lyria-3.5", "gemini-embedding-2",
                  "gemini-2.5-computer-use-preview-10-2025"):
        assert not tier1.dispatchable(model), model

    for model in ("gpt-5.6-sol", "gpt-6-astra", "claude-opus-4-8",
                  "claude-fable-5-1", "gemini-3.8-flash", "gemini-3.5-flash"):
        assert tier1.dispatchable(model), model


def test_new_is_measured_against_the_roster_not_the_capability_table(roster_at, capsys):
    """`CAPABILITIES` holds the models we USE, not every model that exists.

    Diffing a live roster against it reported `gpt-5.2` as NEW — which it is not;
    it is simply not one we characterised, and it would have reported as NEW on
    every run forever.
    """
    roster_at(BASE, BASE)          # live matches the roster; none are in CAPABILITIES
    assert tier1.run(["anthropic"]) == 0
    assert "NEW" not in capsys.readouterr().out


def test_a_genuinely_new_model_is_reported(roster_at, capsys):
    roster_at(BASE, [*BASE, "claude-opus-4-9"])
    assert tier1.run(["anthropic"]) == 1
    assert "NEW        claude-opus-4-9" in capsys.readouterr().out


def test_a_model_in_use_disappearing_is_BROKEN_not_merely_vanished(roster_at, capsys):
    """One event, reported once, as the thing that matters.

    Printing it as both VANISHED and BROKEN doubled the line and buried which one
    was serious — and BROKEN is serious, because a spec naming that model fails.
    """
    roster_at(BASE, [m for m in BASE if m != "claude-sonnet-5"])
    assert tier1.run(["anthropic"]) == 1
    out = capsys.readouterr().out
    mentions = [l for l in out.split("\n")
                if "sonnet-5" in l and ("BROKEN" in l or "VANISHED" in l)]
    assert len(mentions) == 1
    assert "BROKEN" in mentions[0]


def test_a_model_we_do_not_use_vanishing_is_reported_as_vanished(roster_at, capsys):
    roster_at(BASE, [m for m in BASE if m != "claude-opus-4-6"])
    assert tier1.run(["anthropic"]) == 1
    assert "VANISHED   claude-opus-4-6" in capsys.readouterr().out


def test_no_approved_roster_is_incomplete(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tier1, "ROSTER", tmp_path / "absent.json")
    monkeypatch.setattr(tier1, "roster", lambda p: (BASE, None))
    assert tier1.run(["anthropic"]) == 2
    assert "NO APPROVED ROSTER" in capsys.readouterr().out


def test_the_first_run_shows_what_would_be_approved(tmp_path, monkeypatch, capsys):
    """Without this the approval gate is defeated on the run that needs it most.

    An earlier version printed "everything below reads as new" in the header and
    "unchanged" for every provider in the body — two contradictory claims and
    nothing to read. A person told to review before approving was shown nothing
    to review, so approving could only be blind.

    With no baseline there is no diff, so the full roster IS the thing being
    approved and it has to be visible.
    """
    monkeypatch.setattr(tier1, "ROSTER", tmp_path / "absent.json")
    monkeypatch.setattr(tier1, "roster", lambda p: (BASE, None))
    tier1.run(["anthropic"])
    out = capsys.readouterr().out

    assert "unchanged" not in out, (
        "claims nothing changed while also claiming everything is new")
    for model in BASE:
        assert model in out, f"{model} would be approved unseen"
    assert "(in use)" in out, "does not distinguish characterised models"


def test_approve_refuses_a_partial_roster(tmp_path, monkeypatch, capsys):
    """Approving what could only be half-seen would record the unreachable
    provider as empty, and the next run would read that as every model vanishing."""
    monkeypatch.setattr(tier1, "ROSTER", tmp_path / "roster.json")
    monkeypatch.setattr(tier1, "roster",
                        lambda p: (([], "no key") if p == "gemini" else (BASE, None)))
    assert tier1.approve(["anthropic", "gemini"]) == 2
    assert not (tmp_path / "roster.json").exists()
    assert "Refusing to approve a partial roster" in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — from its first real run, which was mostly false positives
# ═══════════════════════════════════════════════════════════════════════════════

def test_the_probe_value_is_not_reported_as_a_finding():
    """Providers quote the offending input back.

    "expected 'low' or 'high', got 'X'" — so an extractor taking every quoted
    token returns its own probe value and reports it as newly accepted. Seven of
    nine findings on the first real tier 2 run were exactly this.
    """
    message = {"error": {"message":
               f"Input should be 'low', 'medium' or 'high', got '{tier2.BOGUS}'"}}
    values = tier2._values_in_error(message)
    assert tier2.BOGUS not in values
    assert set(values) == {"low", "medium", "high"}


def test_a_bare_parameter_goes_inside_gemini_generation_config():
    """Sending it top-level gets rejected for being an UNKNOWN FIELD, which the
    differential probe read as "temperature is now rejected" — a finding
    manufactured entirely by putting the parameter in the wrong place."""
    body = tier2._body_with("gemini", "m", "temperature", 0.7)
    assert "temperature" not in body
    assert body["generation_config"]["temperature"] == 0.7


def test_a_dotted_parameter_keeps_its_own_path():
    """Dotted names already carry their nesting; the gemini fix must not
    double-wrap them."""
    body = tier2._body_with("gemini", "m", "generation_config.thinking_level", "X")
    assert body["generation_config"]["thinking_level"] == "X"
    assert "generation_config" not in body["generation_config"]


def test_enum_findings_say_listed_not_accepted():
    """A rejection message names what the API says it accepts.

    Asserting from that message that a value IS accepted is one inference too
    many — the finding reports what was LISTED and leaves the conclusion to a
    person.

    Asserted on the Finding this produces, not on the source text. An earlier
    version grepped the file and failed on its own explanatory comments, which is
    a test measuring the wrong artifact.
    """
    report = tier2.Report()
    # Two calls now: the bogus probe for the schema, then the plausible probe for
    # the per-model set.
    calls = iter([(400, {"error": {"message":
                  "Input should be 'low', 'high' or 'brand_new_value'"}}),
                  (200, {})])
    import machine_psych.capabilities as caps
    saved = caps.ENUMS["openai"]
    try:
        caps.ENUMS["openai"] = {"reasoning.effort": ("low", "high")}
        tier2._dispatch_saved = tier2._dispatch
        tier2._dispatch = lambda *a, **k: next(calls)
        tier2._probe_enums("openai", "openai/gpt-5.6-sol", report)
    finally:
        caps.ENUMS["openai"] = saved
        tier2._dispatch = tier2._dispatch_saved

    assert report.findings
    detail = report.findings[0].detail
    assert "in the schema but not in ENUMS" in detail
    assert "brand_new_value" in detail
    assert "accepted" not in detail, (
        "a rejection message names what the API validates against, not what "
        "this model takes")


def test_the_plausible_probe_reaches_the_per_model_set():
    """The half the first version lacked, and the half that matters.

    A bogus value returns the API-WIDE schema, identical across every model of a
    provider. A plausible value — real elsewhere, maybe not here — returns the
    PER-MODEL set. Measured 2026-09-05: `minimal` is in OpenAI's schema and
    rejected by gpt-5.6-sol, gpt-5.5 and gpt-6-astra, each naming a different
    valid set.
    """
    assert tier2.PLAUSIBLE["reasoning.effort"] == "minimal"
    assert tier2.PLAUSIBLE["generation_config.thinking_level"] == "minimal"
    for parameter in ("reasoning.effort", "output_config.effort",
                      "generation_config.thinking_level"):
        assert parameter in tier2.PLAUSIBLE, (
            f"{parameter} has no plausible value, so only its schema is probed")


def test_a_required_companion_field_is_not_a_rejection():
    """Anthropic's `thinking.type: enabled` returns "thinking.enabled.
    budget_tokens: Field required" — acceptance with a CONDITION, not refusal.

    Reading that as a rejection would report a valid value as unsupported.

    Driven through `_probe_enums` with a stubbed dispatch rather than grepping
    the source for the word "required". A source-grep test fails on a rename and
    passes on a broken reimplementation — it tests the text, not the thing.
    """
    import machine_psych.capabilities as caps_mod

    report = tier2.Report()
    calls = iter([
        (400, {"error": {"message": f"Input tag '{tier2.BOGUS}' does not match "
                                    f"any of the expected tags: 'adaptive', "
                                    f"'disabled', 'enabled'"}}),
        (400, {"error": {"message": "thinking.enabled.budget_tokens: Field required"}}),
    ])
    saved_enums = caps_mod.ENUMS["anthropic"]
    saved_dispatch = tier2._dispatch
    try:
        caps_mod.ENUMS["anthropic"] = {
            "thinking.type": ("adaptive", "disabled", "enabled")}
        tier2._dispatch = lambda *a, **k: next(calls)
        tier2._probe_enums("anthropic", "anthropic/claude-sonnet-5", report)
    finally:
        caps_mod.ENUMS["anthropic"] = saved_enums
        tier2._dispatch = saved_dispatch

    assert not report.findings, (
        f"a companion-field error was reported as a problem: "
        f"{[f.detail for f in report.findings]}")


def test_the_field_name_is_not_read_as_an_enum_value():
    """One provider quotes the discriminator alongside the values.

    "Input tag '...' found using 'type' does not match any of the expected tags:
    'adaptive', 'disabled', 'enabled'" — an extractor taking every quoted token
    returns `type` and reports it as a valid enum member.
    """
    message = {"error": {"message":
        "thinking: Input tag '__drift_probe_invalid__' found using 'type' does "
        "not match any of the expected tags: 'adaptive', 'disabled', 'enabled'"}}
    assert tier2._values_in_error(message) == ["adaptive", "disabled", "enabled"]


def test_an_unquoted_random_order_list_is_parsed():
    """One provider names its valid set UNQUOTED and in a different order on each
    call — "medium, low, high" then "high, low, medium". Quoted extraction alone
    finds nothing there."""
    message = {"error": {"message":
        "'minimal' is not a supported thinking level for this model. "
        "Allowed values are: medium, low, high."}}
    assert set(tier2._values_in_error(message)) >= {"low", "medium", "high"}


def test_the_conjunction_is_stripped_from_the_tail():
    """"…, 'xhigh', and 'max'" splits into a chunk reading "and 'xhigh'", and an
    uncleaned split reports `and 'xhigh` as a valid value."""
    message = {"error": {"message":
        "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."}}
    values = tier2._values_in_error(message)
    assert values == ["high", "low", "medium", "none", "xhigh"]
    assert not any(v.startswith("and") for v in values)


def test_a_missing_parameter_is_not_an_empty_enum():
    """`reasoning.mode` does not exist on gpt-5.5 — "not supported with this
    model", naming no values.

    An extractor reads an empty set from that and reports "accepts nothing",
    which looks like a NARROWING when it is an ABSENCE. The absence is the bigger
    fact: a whole parameter exists on one model of a family and not another.
    """
    import machine_psych.capabilities as caps_mod

    report = tier2.Report()
    calls = iter([
        (400, {"error": {"message": f"Invalid value: '{tier2.BOGUS}'. Supported "
                                    f"values are: 'standard' and 'pro'."}}),
        (400, {"error": {"message": "`reasoning.mode` is not supported with this model."}}),
    ])
    saved_enums = caps_mod.ENUMS["openai"]
    saved_dispatch = tier2._dispatch
    try:
        caps_mod.ENUMS["openai"] = {"reasoning.mode": ("standard", "pro")}
        tier2._dispatch = lambda *a, **k: next(calls)
        tier2._probe_enums("openai", "openai/gpt-5.5", report)
    finally:
        caps_mod.ENUMS["openai"] = saved_enums
        tier2._dispatch = saved_dispatch

    assert report.findings, "the missing parameter was not reported at all"
    detail = report.findings[-1].detail
    assert "DOES NOT EXIST" in detail
    assert "narrowed" in detail or "not a narrowed" in detail


def test_drift_does_not_rebuild_provider_auth():
    """`base.py` records the lesson: copies are what drifted last time, in seven
    of ten shared functions. Then `drift/` made two more.

    Both scripts rebuilt the URL map and the three header shapes locally, because
    `dispatch` discards the HTTP status and a rejection probe needs it. The fix
    was to expose the status (`dispatch_with_status`), not to keep the copies.
    """
    for name in ("tier2.py", "collect_fixtures.py"):
        src = (pathlib.Path(tier2.__file__).parent / name).read_text()
        assert "requests.post" not in src, f"{name} dispatches around the provider"
        assert "https://api." not in src, f"{name} rebuilds the URL map"


def test_dispatch_and_dispatch_with_status_share_one_implementation():
    """Two POSTs would be two things to keep in sync — which is the whole point.

    Asserted on the parsed BODY, not the source text. A first version grepped the
    source and failed on the docstring, which mentions `requests.post` while
    explaining why there is only one — a test measuring the comment rather than
    the code, which is the second time that mistake has appeared here.
    """
    import ast
    import inspect

    from machine_psych.providers.base import Provider

    fn = ast.parse(inspect.getsource(Provider.dispatch).strip()).body[0]
    statements = [n for n in fn.body
                  if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    code = " ".join(ast.unparse(n) for n in statements)
    assert "requests.post" not in code, "dispatch has its own POST again"
    assert "dispatch_with_status" in code
