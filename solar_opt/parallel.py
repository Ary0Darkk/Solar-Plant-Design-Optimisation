"""
MPI Communication Wrapper and Multiprocessing Partitioning.
"""

from typing import Any, List


class StandaloneCommunicator:
    """Mock MPI communicator for single-node / local workstations."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def send(self, obj: Any, dest: int = 0, tag: int = 0) -> None:
        pass

    def recv(self, source: int = 0, tag: int = 0) -> Any:
        return None


def get_communicator() -> Any:
    """Return active MPI communicator or StandaloneCommunicator fallback."""
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD
    except Exception:
        pass

    return StandaloneCommunicator()


def partition_batches(items: List[Any], num_partitions: int) -> List[List[Any]]:
    """Partition a list of items into N balanced sub-batches."""
    if num_partitions <= 0:
        raise ValueError("num_partitions must be >= 1")

    k, m = divmod(len(items), num_partitions)
    batches = []
    for i in range(num_partitions):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        batches.append(items[start:end])

    return batches
