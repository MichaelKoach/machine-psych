"""Collect the test fixtures.

Committed because it was pasted from a conversation twice, and both times it
found real API changes — four in one pass on 2026-08-28, including a parameter
that had been replaced and a usage field that had been removed. A procedure that
valuable should not live in a chat log.

**Fixtures are the only artifact in this project that can contradict the
documentation.** The capability table is data someone typed; the status logic is
prose in a docstring. A synthetic fixture is built from both and agrees with both
by construction — only a recorded response can disagree.

Several are collected specifically to make a WRONG parser fail. The Gemini answer
must contain non-ASCII, or byte and character offsets agree and the test proves
nothing. The Anthropic truncation must carry partial text, or it passes under both
the correct predicate and the wrong one. A fixture where right and wrong produce
the same answer tests nothing.

Usage:

    python drift/collect_fixtures.py                 # writes ./fixtures/
    python drift/collect_fixtures.py --out /tmp/fx   # elsewhere

Then see REFRESH.md — the diff is the point, not the collection.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from machine_psych import paths
from machine_psych.providers import get_provider

__all__ = ["collect_all"]

MODELS = {"anthropic": "claude-sonnet-5", "openai": "gpt-5.6-sol",
          "gemini": "gemini-3.7-flash"}

LONG = ("What are the top-rated CRM tools for small consulting firms? "
        "Compare at least six on price, ratings and features.")
MATH = ("A consulting firm bills three clients. Client A pays 40% of $50,000 "
        "total revenue. Client B pays 1.5x what Client C pays. What does each pay?")
# Non-ASCII ON PURPOSE. Byte and character offsets are identical on ASCII, so a
# parser using the wrong convention passes vacuously without this.
PRICE = ("Compare CRM pricing for consultants. Use en-dashes in ranges, like "
         "$14–$15 per user/month, and cite a source for each price.")


def _transient(status: int, body: dict) -> bool:
    """429 alone is not enough — a billing error wears the same status code.

    Measured: OpenAI's only 429 in ~300 calls was `insufficient_quota`, a
    permanent condition. Retrying it wastes minutes on something that cannot
    clear.
    """
    err = (body or {}).get("error") or {}
    kind = str(err.get("type") or err.get("code") or "")
    message = str(err.get("message") or "").lower()
    if kind == "insufficient_quota" or "no credits" in message:
        return False
    return (status in (429, 500, 502, 503, 529)
            or "overloaded" in message or "try again later" in message)


def collect(root: pathlib.Path, provider: str, name: str, body: dict,
            why: str, retries: int = 4) -> dict:
    out = root / provider
    out.mkdir(parents=True, exist_ok=True)
    # Through the PROVIDER, not a local copy of its URL and auth. Those were
    # duplicated here and in tier2 — two more copies of things `Provider.url_for`
    # and `Provider.headers` already own, in a codebase whose base class records
    # that copies are what drifted last time.
    impl = get_provider(provider, api_key=paths.api_key(provider))
    for attempt in range(1, retries + 1):
        status, response = impl.dispatch_with_status(body, timeout=900)
        if not _transient(status, response):
            break
        wait = 8 * attempt
        print(f"    {name}: {status}, retry in {wait}s")
        time.sleep(wait)

    (out / f"{name}.json").write_text(json.dumps({
        "_why": why,
        "_collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_request": body, "_http_status": status,
        "response": response}, indent=1))

    units = response.get("content") or response.get("output") or response.get("steps") or []
    kinds = [u.get("type") for u in units if isinstance(u, dict)]
    print(f"  {name:<26} {status} "
          f"{response.get('stop_reason') or response.get('status')!s:<11}"
          f" {len(units):>3} {kinds[:4]}")
    time.sleep(0.4)
    return response


def collect_all(root: pathlib.Path) -> None:
    A = lambda **kw: {"model": MODELS["anthropic"], **kw}          # noqa: E731
    def oai(**kw):
        return {"model": MODELS["openai"], "store": False, **kw}

    def G(**kw):
        gen = {"max_output_tokens": kw.pop("max_output_tokens", 16384),
               "thinking_summaries": kw.pop("thinking_summaries", "auto")}
        if "thinking_level" in kw:
            gen["thinking_level"] = kw.pop("thinking_level")
        return {"model": MODELS["gemini"], "store": False,
                "generation_config": gen, **kw}

    SA = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}]
    SO = [{"type": "web_search"}]
    SG = [{"type": "google_search"}]
    go = lambda *a: collect(root, *a)                              # noqa: E731

    print("ANTHROPIC")
    go("anthropic", "ungrounded_ok",
       A(max_tokens=2048, messages=[{"role": "user", "content": "Name one CRM tool."}]),
       "Baseline: one text block, end_turn, no tools.")
    go("anthropic", "grounded_ok", A(max_tokens=8192, tools=SA,
       messages=[{"role": "user", "content": LONG}]),
       "MAIN PARSE TARGET. The only provider returning the RETRIEVAL SET — every "
       "URL seen, not only those cited — plus page_age and quoted citations.")
    go("anthropic", "truncated_partial", A(max_tokens=400,
       messages=[{"role": "user", "content": LONG}]),
       "TRUNCATED: max_tokens WITH real partial text. A fixture truncating to "
       "nothing passes under both the right predicate and the wrong one.")
    go("anthropic", "incomplete_empty", A(max_tokens=5, tools=SA,
       messages=[{"role": "user", "content": LONG}]),
       "NO-ANSWER shape 1: a thinking block with no text, NOT an empty array. "
       "The shape changed when thinking became default.")
    go("anthropic", "incomplete_ends_on_tool", A(max_tokens=200, tools=SA,
       messages=[{"role": "user", "content": LONG}]),
       "NO-ANSWER shape 2: blocks present, last is a tool. Since this provider "
       "began emitting text BEFORE tools, the pre-tool text is PROCESS, not an "
       "answer — status and parse must agree that there is no answer here.")
    go("anthropic", "usage_contradicts_blocks", A(max_tokens=8192, tools=SA,
       messages=[{"role": "user", "content":
                  "Compare G2 and Capterra ratings for HubSpot, Pipedrive, Zoho, "
                  "Copper, Insightly and Capsule. Cite a source for each."}]),
       "n_queries must come from usage, never from counting blocks.")
    t0 = go("anthropic", "multiturn_t0", A(max_tokens=4096, tools=SA,
            messages=[{"role": "user", "content": LONG}]),
            "Turn 0 of a grounded ladder; its content is what passback returns.")
    go("anthropic", "multiturn_t1", A(max_tokens=4096, tools=SA, messages=[
       {"role": "user", "content": LONG},
       {"role": "assistant", "content": t0.get("content", [])},
       {"role": "user", "content": "Which would you pick for a two-person firm?"}]),
       "Turn 1 with turn 0 passed back verbatim.")
    go("anthropic", "thinking_disabled", A(max_tokens=2048,
       thinking={"type": "disabled"},
       messages=[{"role": "user", "content": "Name one CRM tool."}]),
       "The reasoning-off arm, which is impossible on Gemini.")
    go("anthropic", "effort_high", A(max_tokens=8192, thinking={"type": "adaptive"},
       output_config={"effort": "high"}, messages=[{"role": "user", "content": MATH}]),
       "The CURRENT reasoning control. `thinking.type: enabled` also works and "
       "requires budget_tokens — both controls are live.")
    for name, body, why in [
        ("rejects_temperature", A(max_tokens=1024, temperature=0.7,
         messages=[{"role": "user", "content": "hi"}]),
         "400 where Gemini accepts and silently ignores."),
        ("rejects_unknown_param", A(max_tokens=1024, nonsense_parameter=1,
         messages=[{"role": "user", "content": "hi"}]),
         "Unknown parameter named in the error."),
        ("rejects_bad_model", {"model": "claude-nonexistent", "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]},
         "404. Must classify as error and never retry.")]:
        go("anthropic", name, body, why)

    print("\nOPENAI")
    go("openai", "ungrounded_ok", oai(max_output_tokens=2048,
       input="Name one CRM tool."),
       "Baseline. STRUCTURAL answer extraction — the API says which item is the "
       "answer. Note a reasoning item is NOT always present.")
    go("openai", "grounded_ok", oai(max_output_tokens=16384, tools=SO, input=PRICE),
       "MAIN PARSE TARGET. CHARACTER-offset annotations. `usage.tool_usage` was "
       "removed in Aug 2026, so n_queries now falls back to the item count.")
    for budget in (16, 64, 256, 1024, 4096):
        go("openai", f"truncate_{budget}", oai(max_output_tokens=budget, input=LONG),
           f"Budget sweep at {budget}. Single-turn incomplete arms have zero "
           f"message items — but a two-turn ladder at 200 DOES produce partial "
           f"text, so 'complete or not at all' is wrong as a general claim.")
    go("openai", "truncate_grounded_512", oai(max_output_tokens=512, tools=SO,
       input=LONG), "Truncation with search on.")
    o0 = go("openai", "multiturn_t0", oai(max_output_tokens=8192, tools=SO, input=LONG),
            "Turn 0 with store:false, so history is passed explicitly.")
    go("openai", "multiturn_t1", oai(max_output_tokens=8192, tools=SO,
       input=[{"role": "user", "content": LONG}] + (o0.get("output") or [])
       + [{"role": "user", "content": "Which for a two-person firm?"}]),
       "Turn 1 with reasoning items resent — they carry encrypted_content.")
    for level in ("low", "high"):
        go("openai", f"verbosity_{level}", oai(max_output_tokens=8192,
           text={"verbosity": level}, input=LONG),
           f"verbosity={level}. The only output-length control of the three.")
    go("openai", "max_tool_calls_1", oai(max_output_tokens=8192, tools=SO,
       max_tool_calls=1, input=LONG), "The only search cap of the three.")
    go("openai", "effort_none", oai(max_output_tokens=8192,
       reasoning={"effort": "none"}, input=MATH),
       "0 reasoning tokens: a real off arm. Note `minimal` is in the SCHEMA and "
       "rejected by this model — schema is not capability.")
    go("openai", "reasoning_summary", oai(max_output_tokens=8192,
       reasoning={"effort": "high", "summary": "detailed"}, input=MATH),
       "READABLE REASONING, opt-in per call. Was empty at characterisation and "
       "populates now.")
    for name, body, why in [
        ("rejects_temperature", oai(max_output_tokens=1024, temperature=0.7,
         input="hi"), "400 where Gemini accepts and ignores."),
        ("rejects_unknown_param", oai(max_output_tokens=1024, nonsense_parameter=1,
         input="hi"), "Unknown parameter named in the error."),
        ("rejects_bad_model", {"model": "gpt-nonexistent", "max_output_tokens": 1024,
         "input": "hi", "store": False}, "Bad model. Error, never retried.")]:
        go("openai", name, body, why)

    print("\nGEMINI")
    go("gemini", "ungrounded_ok", G(input="Name one CRM tool."),
       "Baseline. Thinking fires unrequested — a family property.")
    go("gemini", "grounded_ok", G(input=PRICE, tools=SG, thinking_level="medium"),
       "MAIN PARSE TARGET. The prompt asks for EN-DASHES on purpose: byte and "
       "character offsets diverge on non-ASCII, so a parser using characters "
       "FAILS instead of passing vacuously.")
    go("gemini", "ungrounded_with_tool", G(input="What is 2 plus 2?", tools=SG),
       "CONDITIONAL GROUNDING: the tool was provided and no search ran.")
    go("gemini", "readable_reasoning", G(input=MATH, thinking_level="high"),
       "Thought text arrives in `summary`, NOT `content`.")
    go("gemini", "no_summaries", G(input="What is 17 times 23?",
       thinking_summaries="none"),
       "Without thinking_summaries the key is ABSENT entirely, not empty.")
    go("gemini", "incomplete_no_output", G(input=LONG, max_output_tokens=5),
       "A tiny budget. NOTE: this sits on a boundary — it has returned both a "
       "partial answer and none, and that variance is why a scheduled shape "
       "diff on this fixture produced a false positive.")
    gt0 = go("gemini", "multiturn_t0", G(input=LONG, tools=SG,
             thinking_level="medium"),
             "Turn 0 of a grounded ladder. ALL steps must go back verbatim.")
    go("gemini", "multiturn_t1", G(tools=SG, thinking_level="medium",
       input=[{"type": "user_input", "content": LONG}] + (gt0.get("steps") or [])
       + [{"type": "user_input", "content": "Which for a two-person firm?"}]),
       "Turn 1 in step_list format. A role/content turn is a 400 here.")
    go("gemini", "rejects_bad_field", {"model": MODELS["gemini"], "input": "hi",
       "store": False, "generation_config": {"include_thoughts": True}},
       "A generateContent field. Rejected in ~0.2s, which reads as an outage.")
    go("gemini", "rejects_bad_model", {"model": "gemini-nonexistent", "input": "hi",
       "store": False, "generation_config": {"max_output_tokens": 1024}},
       "404. Error, never retried.")

    (root / "_NOT_COLLECTED.md").write_text("""# States with no recorded fixture

**Transient failures.** Anthropic produced zero in ~150 calls; OpenAI's only 429
was a billing error. The `unavailable` branch of every status classifier is
therefore tested SYNTHETICALLY, which is weaker than every other test here.

KEEP these hand-built files when replacing the directory — this script does not
recreate them, and a refresh that drops them makes every one look removed:

    anthropic/overloaded.json
    openai/rate_limited.json
    openai/rejects_no_credit.json      (real, but a BILLING error at HTTP 429)
    gemini/unavailable_503.json
    gemini/unavailable_api_error.json

Gemini's are less synthetic than the others — its capacity failures were observed
live at roughly 1 in 6, arriving as `code: "api_error"` with a prose message
rather than an HTTP 503. That shape is real; only the saved file is reconstructed.
""")
    n = len(list(root.rglob("*.json")))
    print(f"\n  {n} fixtures in {root}")
    print("  Now see drift/REFRESH.md — the DIFF is the point, not the collection.")


if __name__ == "__main__":
    args = sys.argv[1:]
    dest = pathlib.Path(args[args.index("--out") + 1]) if "--out" in args \
        else pathlib.Path("fixtures")
    collect_all(dest)
