"""
MPI and standalone communicator abstractions for single-node and multi-node execution.
"""

from typing import Any, Protocol, runtime_checkable

try:
    from mpi4py import MPI
except Exception:
    MPI = None


@runtime_checkable
class Communicator(Protocol):
    """Protocol defining the core MPI communicator methods required by solar_opt."""

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
    """Mock MPI communicator for running in single-node environments without MPI."""

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
    """
    Return active MPI communicator (MPI.COMM_WORLD) if available,
    otherwise fallback to StandaloneComm.
    """
    if MPI is not None:
        try:
            return MPI.COMM_WORLD
        except Exception:
            pass
    return StandaloneComm()
