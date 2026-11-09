"""Paid Search two-arm geo dark-out + branded cannibalisation test (module 2)."""
from .settings import ARMS, SearchSettings, get_settings  # noqa: F401


def run(*args, **kw):
    """Run the full pipeline (lazy import keeps `import incrlab.search` light)."""
    from .pipeline import run as _run

    return _run(*args, **kw)
