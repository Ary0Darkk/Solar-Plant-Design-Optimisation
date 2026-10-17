"""
Standalone MPI & Hybrid Parallelism Diagnostic Utility for HPC Clusters.

Tests MPI communication, node distribution, PySAM loading, and local multiprocessing
across all allocated cluster nodes.

Usage on HPC cluster:
    mpirun -np 4 python test_hpc_mpi.py
    mpirun -np 7 --hostfile $PBS_NODEFILE python test_hpc_mpi.py
"""

import multiprocessing as mp
import os
import socket
import sys


def _square(x: int) -> int:
    """Helper top-level function for multiprocessing pickling."""
    return x * x


def main():
    try:
        from mpi4py import MPI
    except Exception as exc:
        print(f"[FAIL] mpi4py could not load the C MPI library: {exc}")
        print("Troubleshooting Tips for HPC:")
        print("  1. Load an MPI module on your cluster, e.g.:")
        print("       module load openmpi")
        print("       module load intel-mpi")
        print("  2. Reinstall mpi4py inside the virtual environment after loading the module:")
        print("       pip install --force-reinstall --no-cache-dir mpi4py")
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    # 1. Hostname and CPU reporting
    host_info = {
        "rank": rank,
        "hostname": hostname,
        "pid": os.getpid(),
        "cpu_count": mp.cpu_count(),
    }
    all_hosts = comm.gather(host_info, root=0)

    if rank == 0:
        print("=" * 65)
        print("     NYS-PySAM HPC MPI & Hybrid Parallelism Pre-flight Check")
        print("=" * 65)
        print(f"MPI Library Vendor : {MPI.get_vendor()}")
        print(f"MPI World Size     : {size} rank(s)")
        print("-" * 65)
        print("Compute Node Allocation:")
        for h in all_hosts:
            print(
                f"  Rank {h['rank']:2d} -> Host: {h['hostname']:<16} "
                f"(PID {h['pid']:<6}, {h['cpu_count']:3d} CPUs available)"
            )
        print("-" * 65)

    # 2. Test Collective Broadcast from Rank 0
    test_payload = (
        {"msg": "MPI_BCAST_TEST_OK", "season": "winter", "pop_size": 28} if rank == 0 else None
    )
    bcast_result = comm.bcast(test_payload, root=0)
    assert bcast_result["msg"] == "MPI_BCAST_TEST_OK", (
        f"Broadcast verification failed on Rank {rank}"
    )

    # 3. Test Point-to-Point Master-Worker Handshake
    if size > 1:
        if rank == 0:
            for w in range(1, size):
                comm.send(f"PING_RANK_{w}", dest=w, tag=42)
            for w in range(1, size):
                reply = comm.recv(source=w, tag=43)
                print(f"[Rank 0] Handshake OK with Rank {w:2d} -> Reply: '{reply}'")
        else:
            comm.recv(source=0, tag=42)
            comm.send(f"PONG_FROM_RANK_{rank}_ON_{hostname}", dest=0, tag=43)

    # 4. Test PySAM and Nested Multiprocessing on every rank
    try:
        import PySAM.PhysicalTroughNone  # noqa: F401
        import PySAM.TroughPhysical  # noqa: F401

        pysam_status = "PySAM OK"
    except Exception as exc:
        pysam_status = f"PySAM FAILED: {exc}"

    # Verify nested multiprocessing pool (used by each rank to run daily PySAM simulations)
    try:
        workers = min(4, mp.cpu_count())
        ctx = mp.get_context()
        with ctx.Pool(processes=workers) as pool:
            squares = pool.map(_square, [1, 2, 3, 4])
            pool_status = "Multiprocessing Pool OK" if squares == [1, 4, 9, 16] else "Pool FAILED"
    except Exception as exc:
        pool_status = f"Multiprocessing FAILED: {exc}"

    rank_status = comm.gather((rank, hostname, pysam_status, pool_status), root=0)

    if rank == 0:
        print("-" * 65)
        print("Subsystem Verification Across All MPI Ranks:")
        all_passed = True
        for r, h, py_st, pl_st in rank_status:
            print(f"  Rank {r:2d} ({h:<16}): {py_st:<12} | {pl_st}")
            if "FAILED" in py_st or "FAILED" in pl_st:
                all_passed = False
        print("=" * 65)
        if all_passed:
            print("  >>> ALL MPI & HYBRID PARALLEL CHECKS PASSED SUCCESSFULLY! <<<")
            print("  Your cluster environment is fully ready for multi-node runs.")
        else:
            print("  >>> SOME SUBSYSTEM CHECKS FAILED! Check error messages above. <<<")
        print("=" * 65)


if __name__ == "__main__":
    main()
