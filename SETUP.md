# Install and update

The repo is `github.com/MichaelKoach/machine-psych`. This is how it gets used from
a notebook, and how it gets updated when the package changes.

## Install, pinned to a commit

```python
!pip install -q git+https://github.com/MichaelKoach/machine-psych.git@COMMIT_SHA

import machine_psych as mp
mp.install_report()
```

Expected:

```
machine-psych 0.1.0  ·  commit a3f9c21b
```

**Pin the commit.** A version string is something a person typed; a SHA is
verifiable, and it is what lets a corpus be traced to the exact parsing code that
produced it. Get the SHA with `git rev-parse HEAD`, or from the commit list on
GitHub.

**If it prints `commit unknown`,** provenance recording is broken. `provenance()`
reads pip's `direct_url.json` to recover the SHA; a local install correctly
reports "not from git", but the git path only exists once the package is
installed from a URL. Fix it before anything depends on it.

## Keys and where output goes

Installing gets you the package. **Two more things are needed before it can run**,
and neither has a default that will work for someone else.

```python
import machine_psych as mp
from machine_psych import paths

paths.set_api_key("anthropic", "sk-ant-...")
paths.set_api_key("openai",    "sk-...")
paths.set_api_key("gemini",    "...")

mp.set_base("/path/that/persists")
```

**Keys are held in memory only** — never written to disk, never read from a file
in the repo. In Colab they come from the secrets panel via
`userdata.get("YOUR_SECRET_NAME")`; the names are whatever you called them.

**`set_base` decides where records are written**, and records are written as they
arrive rather than at the end. It defaults to the current working directory, which
is fine locally and **wrong in Colab** — `/content` is wiped on disconnect, so a
long battery would vanish. Point it at Drive there.

`MACHINE_PSYCH_BASE` sets it from the environment if that suits better.

**Two functions answer "where am I pointed":** `mp.where()` returns the base and
the directories under it; `mp.install_report()` prints the version and the commit
it was installed from, or says plainly that provenance cannot be recorded when
installed from a local path rather than git.

Run both first if something is not working — a corpus written to the wrong base
and a package installed without provenance both fail quietly.

## Update after a change

Colab caches by package name, so a new commit is not picked up by a plain
re-install:

```python
!pip install -q --force-reinstall --no-deps \
    git+https://github.com/MichaelKoach/machine-psych.git@NEW_SHA
```

Then **restart the runtime.** Python caches imported modules, and this has
already cost time four times on this project — a patched module on disk is not a
patched module in memory. The symptom is a bug you have already fixed still
happening, or `mp.__version__` reporting the old number.

## Running the drift checks

`pip install` does NOT give you `drift/`. `pyproject.toml` packages
`machine_psych*` only, deliberately — the drift scripts make live calls and cost
money, so they are not part of the library. `from drift import tier1` after a pip
install fails with `ModuleNotFoundError`, and it has.

Clone instead:

```python
!rm -rf /content/mp
!git clone -q https://github.com/MichaelKoach/machine-psych.git /content/mp
import sys; sys.path.insert(0, "/content/mp")
```

**Then restart the runtime before importing.** A clone into a path that was
already on `sys.path` during a previous import is invisible until the caches are
cleared, and a session restart is the reliable way — it has been quicker than
diagnosing it every time.

## Pushing changes

```bash
cd path/to/machine-psych
git add .
git commit -m "what changed"
git push
git rev-parse HEAD        # the new SHA to pin
```

## What goes in, and what does not

**In:** the package, tests, the briefing document.

**Out:** API keys, corpora, investigation specs containing client material,
anything under `Output Log/`. The `.gitignore` covers the obvious cases, but hold
the rule independently of the file: **this repo is public.**
