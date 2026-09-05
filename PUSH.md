# Pushing this to the repo

Unzip over `~/Documents/machine-psych`, overwriting. Then:

```bash
cd ~/Documents/machine-psych

# 1. verify BEFORE committing — a missing file shows up here, not after the push
python3 -m pip install --quiet -e . && python3 -m pip install --quiet pytest
python3 -m pytest tests/ -q
```

Expect **313 passed, 1 skipped**. A different count means something did not
survive the unzip; stop and say so rather than pushing.

```bash
# 2. see what is staged
git add .
git status
```

Roughly 70 files. If `build/`, `.egg-info`, `__pycache__` or `.pytest_cache`
appear, the `.gitignore` did not overwrite — they are artifacts of the editable
install in step 1, not source.

```bash
# 3. push
git commit -m "Merged three-provider harness: 13 modules, 313 tests, recorded fixtures"
git push
git rev-parse HEAD
```

## Then confirm the delivery path

In Colab, with that SHA:

```python
!pip install -q --force-reinstall --no-deps \
    git+https://github.com/MichaelKoach/machine-psych.git@YOUR_SHA
```

**Restart the runtime** — Python caches imported modules, and a patched module on
disk is not a patched module in memory. This has cost time twice.

```python
import machine_psych as mp
mp.install_report()
```

Want: `machine-psych 0.1.0 · commit <sha>`

`commit unknown` means provenance recording is broken, and every corpus collected
after that loses its trace back to the code that produced it.

## What is in here

- `machine_psych/` — 13 modules, ~4,500 lines
- `tests/` — 9 files, 313 tests
- `tests/fixtures/` — 46 recorded API responses across three providers

The fixtures are real responses. The prompts are generic CRM questions with no
client material, but the repo is public, so glance before pushing.

`tests/fixtures/_NOT_COLLECTED.md` records the states that could not be captured
live — chiefly transient failures, which no provider produced in ~300 calls, so
that branch of every status classifier is tested synthetically.
