"""Process-wide singletons (store), kept tiny and import-cycle-free."""

from functools import lru_cache

from app.config import settings
from app.tracing.store import Store


@lru_cache
def store() -> Store:
    return Store(settings().db_path)
