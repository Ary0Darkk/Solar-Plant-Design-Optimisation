"""
MPI Communication Wrapper and Multiprocessing Partitioning.
"""

from typing import Any, List, Optional


class StandaloneCommunicator:
    """Mock MPI communicator for single-node / local workstations."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def Barrier(self) -> None:
        pass

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def gather(self, obj: Any, root: int = 0) -> List[Any]:
        return [obj]

    def scatter(self, obj_list: Optional[List[Any]], root: int = 0) -> Any:
        if obj_list is not None and len(obj_list) > 0:
            return obj_list[0]
        return None


def get_communicator() -> Any:
    """Return active MPI communicator or StandaloneCommunicator fallback."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            return comm
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
