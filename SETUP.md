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

## Update after a change

Colab caches by package name, so a new commit is not picked up by a plain
re-install:

```python
!pip install -q --force-reinstall --no-deps \
    git+https://github.com/MichaelKoach/machine-psych.git@NEW_SHA
```

Then **restart the runtime.** Python caches imported modules, and this has
already cost time twice on this project — a patched module on disk is not a
patched module in memory. The symptom is a bug you have already fixed still
happening, or `mp.__version__` reporting the old number.

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
