"""
Tests for parallel communicator and batch partitioning.
"""

import pytest

from solar_opt.parallel import StandaloneComm, get_communicator, partition_batches


def test_standalone_comm():
    comm = StandaloneComm()
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1
    assert comm.bcast("test_payload") == "test_payload"


def test_get_communicator():
    comm = get_communicator()
    assert hasattr(comm, "Get_rank")
    assert hasattr(comm, "Get_size")
    assert hasattr(comm, "bcast")


def test_partition_batches():
    items = list(range(10))
    batches = partition_batches(items, batch_size=3)
    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]


def test_partition_batches_invalid_size():
    with pytest.raises(ValueError):
        partition_batches([1, 2, 3], batch_size=0)
