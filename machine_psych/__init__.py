"""machine-psych — qualitative behavioural research on language models.

The model is the research subject. The question is how it sorts a category, what
it recommends, what it associates with a brand, and what moves any of that.

    import machine_psych as mp
    mp.install_report()
    mp.set_base("/content/drive/MyDrive/...")
    mp.set_api_key("anthropic", userdata.get("..."))

    run = mp.load_investigation("discuss_cross_provider")   # sends nothing
    results, raw = mp.run_investigation(run)
    corpus = mp.load_corpus("discuss_cross_provider")

    mp.index(corpus)
    view, report = mp.normalize(corpus)                     # what was set aside
    mp.citations(corpus)                                    # and what still is not

Installed from git and pinned by commit, so a corpus records verifiable
provenance rather than a version string someone typed.
"""

from .paths import VERSION, install_report, provenance, set_api_key, set_base

# A plain rebind, not a call. Computing this at import time coupled the package
# to the order of its own imports for no benefit.
__version__ = VERSION


def where() -> dict:
    """The directories currently in use.

    A FUNCTION, not exported constants. `set_base` rebinds module globals, so
    `mp.BASE` imported as a name would keep whatever it held at import time and
    silently show the wrong directory after a rebase — the same stale-value trap
    the paths module exists to document.
    """
    from . import paths
    return {"base": paths.BASE,
            "investigations": paths.INVESTIGATIONS_DIR,
            "records": paths.RECORDS_DIR}

from .analysis import (
    NormalizeReport,
    format_stability,
    index,
    mentions,
    normalize,
    read,
    verdict,
)
from .capabilities import (
    CAPABILITIES,
    ENUMS,
    ModelCaps,
    UnknownModelError,
    caps_for,
    known_models,
    provider_of,
)
from .corpus import (
    capability_note,
    citations,
    load_corpus,
    load_record,
    queries,
    sources,
    thoughts,
    units,
)
from .export import export_corpus
from .providers import PROVIDERS, ParsedResponse, Provider, get_provider
from .runner import (
    estimate,
    list_investigations,
    list_runs,
    load_investigation,
    run_investigation,
    save_investigation,
)
from .spec import (
    InvestigationError,
    UnmetIntentError,
    expand_conditions,
    probe_hash,
    prompt_hash,
    resolve,
    validate_investigation,
)

__all__ = [
    # capabilities
    "CAPABILITIES",
    "ENUMS",
    # providers
    "PROVIDERS",
    # specs
    "InvestigationError",
    "ModelCaps",
    # analysis
    "NormalizeReport",
    "ParsedResponse",
    "Provider",
    "UnknownModelError",
    "UnmetIntentError",
    "__version__",
    # corpus
    "capability_note",
    "caps_for",
    "citations",
    # running
    "estimate",
    "expand_conditions",
    # export
    "export_corpus",
    "format_stability",
    "get_provider",
    "index",
    "install_report",
    "known_models",
    "list_investigations",
    "list_runs",
    "load_corpus",
    "load_investigation",
    "load_record",
    "mentions",
    "normalize",
    "probe_hash",
    "prompt_hash",
    "provenance",
    "provider_of",
    "queries",
    "read",
    "resolve",
    "run_investigation",
    "save_investigation",
    "set_api_key",
    "set_base",
    "sources",
    "thoughts",
    "units",
    "validate_investigation",
    "verdict",
    "where",
]
