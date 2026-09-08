"""Provider registry.

An explicit dict rather than auto-registration. There are three providers, all
imported here by name, so a hook that registered them on class creation bought
nothing a literal cannot do — and a literal is greppable, which the hook was not.
"""

from .base import ParsedResponse, Provider, get_provider
from .anthropic import Anthropic
from .openai import OpenAI
from .gemini import Gemini

PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": Anthropic,
    "openai": OpenAI,
    "gemini": Gemini,
}

# `get_provider` looks the name up here.
from .base import _bind_registry  # noqa: E402
_bind_registry(PROVIDERS)

__all__ = ["ParsedResponse", "Provider", "PROVIDERS", "get_provider",
           "Anthropic", "OpenAI", "Gemini"]
