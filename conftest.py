"""Put the repo root on `sys.path` so `drift` imports however pytest is invoked.

`drift/` is deliberately NOT part of the installed package — `pyproject.toml`
declares `machine_psych*` only, because the drift scripts make live API calls and
cost money. But `tests/test_drift.py` has to import them.

**`python -m pytest` prepends the current directory to `sys.path`; plain `pytest`
does not.** Every local run in this project used the first form, so `drift`
imported by accident of invocation and nobody noticed. CI runs plain `pytest`,
and was the first thing to try the other way — 400 tests passing locally, a
collection error there.

That is exactly the class of problem CI exists to find: a dependency on the
developer's habits rather than on anything the repo declares.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
