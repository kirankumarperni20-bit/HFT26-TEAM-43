"""Backend package for AI Research Assistant (Team 43)."""

import sys

# Provide backward-compatibility alias so that 'llm' is importable when running from repository root
try:
    import backend.llm as _backend_llm

    if "llm" not in sys.modules:
        sys.modules["llm"] = _backend_llm
except ImportError:
    pass
