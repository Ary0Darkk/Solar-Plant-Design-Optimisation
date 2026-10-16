"""
Population batching utilities for parallel MPI evaluation.
"""

from typing import Any, List, Sequence


def batch_items(items: Sequence[Any], batch_size: int = 16) -> List[List[Any]]:
    """
    Partition a sequence of items (e.g. individuals) into sublists of size up to batch_size.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]
