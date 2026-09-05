# States with no recorded fixture

**Transient failures.** Anthropic produced zero in ~150 calls; OpenAI's only 429
was a billing error, not rate limiting. The `unavailable` branch of every status
classifier is therefore tested SYNTHETICALLY, which is weaker than every other
test in the suite. The synthetic files to keep are:

  anthropic/overloaded.json
  openai/rate_limited.json
  gemini/unavailable_503.json
  gemini/unavailable_api_error.json

Gemini's are less synthetic than the others — its capacity failures were observed
live during characterisation at roughly 1 in 6, arriving as `code: "api_error"`
with a prose message rather than an HTTP 503. That shape is real; only the saved
file is reconstructed.

If a genuine transient failure is ever observed in a run, save the raw body here
and convert the test.

**`stop_details` on Anthropic.** Documented, present in the schema, and null in
all 21 responses observed including seven deliberate truncations. The documented
route is a refusal classifier on the always-adaptive model, untested.
