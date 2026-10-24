"""
Tests for MPI communicator and batch partitioning.
"""

import pytest

from solar_opt.parallel import (
    StandaloneCommunicator,
    get_communicator,
    partition_batches,
)


def test_standalone_comm():
    comm = StandaloneCommunicator()
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1
    assert comm.bcast("test") == "test"
    assert comm.gather("val") == ["val"]
    assert comm.scatter(["item1", "item2"]) == "item1"


def test_get_communicator():
    comm = get_communicator()
    assert comm.Get_rank() >= 0
    assert comm.Get_size() >= 1


def test_partition_batches():
    items = list(range(10))
    batches = partition_batches(items, num_partitions=3)
    assert len(batches) == 3
    assert sum(len(b) for b in batches) == 10
    assert batches[0] == [0, 1, 2, 3]
    assert batches[1] == [4, 5, 6]
    assert batches[2] == [7, 8, 9]


def test_partition_batches_invalid_size():
    with pytest.raises(ValueError):
        partition_batches([1, 2, 3], num_partitions=0)
