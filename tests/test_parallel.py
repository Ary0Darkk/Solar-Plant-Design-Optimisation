"""
Unit tests for parallel communication layer and fallback communicator.
"""

from optimized_pipeline.parallel import Communicator, StandaloneComm, get_communicator


def test_standalone_communicator_primitives():
    """Verify that StandaloneComm functions properly in single-process mode."""
    comm = StandaloneComm()
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1

    # Broadcast preserves payload
    data = {"season": "summer", "pop": 28}
    assert comm.bcast(data, root=0) == data

    # Send and recv mocks
    comm.send({"test": 123}, dest=0, tag=0)
    assert comm.recv(source=0, tag=0) is None


def test_get_communicator():
    """Verify get_communicator returns a valid Communicator instance."""
    comm = get_communicator()
    assert isinstance(comm, Communicator)
    assert comm.Get_rank() == 0
    assert comm.Get_size() >= 1
