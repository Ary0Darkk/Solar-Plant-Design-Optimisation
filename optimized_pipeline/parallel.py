"""
Hybrid MPI and standalone multiprocessing communicator abstraction.
"""

from typing import Any, List, Protocol, Sequence, runtime_checkable

try:
    from mpi4py import MPI
except Exception:
    MPI = None


@runtime_checkable
class Communicator(Protocol):
    """Protocol defining the core MPI communicator methods."""

    def Get_rank(self) -> int:
        ...

    def Get_size(self) -> int:
        ...

    def bcast(self, data: Any, root: int = 0) -> Any:
        ...

    def send(self, data: Any, dest: int, tag: int = 0) -> None:
        ...

    def recv(self, source: int = 0, tag: int = 0) -> Any:
        ...


class StandaloneComm:
    """Mock communicator for single-node execution without MPI runtime."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, data: Any, root: int = 0) -> Any:
        return data

    def send(self, data: Any, dest: int, tag: int = 0) -> None:
        pass

    def recv(self, source: int = 0, tag: int = 0) -> Any:
        return None


def get_communicator() -> Communicator:
    """Return active MPI communicator or fallback to StandaloneComm."""
    if MPI is not None:
        try:
            return MPI.COMM_WORLD
        except Exception:
            pass
    return StandaloneComm()


def partition_batches(items: Sequence[Any], batch_size: int = 16) -> List[List[Any]]:
    """Partition a list of items into batches of up to batch_size."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]
