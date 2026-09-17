"""Resource limits for a single-process, small-library workspace."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeLimits:
    # Logical UTF-8 vector payload, not the SQLite file size (WAL/backups are extra).
    vector_bytes: int = 512 * 1024**2
    owner_vector_bytes: int = 128 * 1024**2
    minimum_free_bytes: int = 256 * 1024**2

    @classmethod
    def from_environment(cls):
        values = {}
        for field, name in (("vector_bytes", "AGENT_VECTOR_BYTES_LIMIT"),
                            ("owner_vector_bytes", "AGENT_OWNER_VECTOR_BYTES_LIMIT"),
                            ("minimum_free_bytes", "AGENT_MIN_FREE_BYTES")):
            if name in os.environ:
                try:
                    value = int(os.environ[name])
                except ValueError:
                    raise RuntimeError(f"{name} must be a positive integer") from None
                if value <= 0:
                    raise RuntimeError(f"{name} must be positive")
                values[field] = value
        return cls(**values)


def vector_reservation(dimensions):
    """Conservative upper bound for one normalized JSON vector's UTF-8 bytes."""
    if type(dimensions) is not int or not 1 <= dimensions <= 4096:
        raise ValueError("Embedding dimensions must be 1..4096")
    return dimensions * 32 + 64
