# NYS Design Optimisation using PySAM

A parallel simulation--optimisation pipeline for a Concentrated Solar
Power (CSP) parabolic-trough plant. It combines **NREL SAM/PySAM**,
**DEAP genetic algorithms**, **MPI**, Python multiprocessing, and
**MLflow/DagsHub** to search plant-design and operating parameters
across four seasons.

Repository:
[Ary0Darkk/NYS-Design-Optimisation-using-PySAM](https://github.com/Ary0Darkk/NYS-Design-Optimisation-using-PySAM)

## Workflow

``` mermaid
flowchart TD
    A[Load configuration and seasonal days] --> B[Initialize DEAP genetic algorithm]
    B --> C[Create and batch population]
    C --> D[Rank 0 distributes batches with MPI]
    D --> E[Multiprocessing evaluates candidates]
    E --> F[Run PySAM TroughPhysical simulations]
    F --> G[Calculate daily objective and fitness]
    G --> H[Aggregate fitness per candidate]
    H --> I[Selection, crossover and mutation]
    I --> J[Log metrics and save checkpoint]
    J --> K{More generations?}
    K -- Yes --> C
    K -- No --> L[Export best design and results]
    J -. tracking .-> M[MLflow / DagsHub]
```

## Core capabilities

-   **Physics-based simulation:** PySAM `TroughPhysical` model.
-   **Evolutionary search:** DEAP genetic algorithm with bounded design
    and operating variables.
-   **Seasonal runs:** `winter`, `summer`, `monsoon`, and `post`; each
    uses seven configured representative days.
-   **Hybrid parallelism:** MPI distributes candidate batches across
    ranks; multiprocessing evaluates simulations within each rank.
-   **Experiment tracking:** MLflow metrics and DagsHub remote tracking.
-   **Resumable runs:** Checkpoints preserve GA state and random-number
    generator states.
-   **Failure accounting:** Failed or invalid evaluations receive
    configured penalties and are tracked separately.

## Architecture

  -----------------------------------------------------------------------
  Area                                Responsibility
  ----------------------------------- -----------------------------------
  `main.py`                           CLI entry point, MPI setup, and run
                                      orchestration

  `config.py`                         GA settings, seasonal
                                      representative days, bounds, and
                                      simulation configuration

  `simulation/`                       PySAM model setup, parameter
                                      overrides, and output extraction

  `optimisation/`                     GA, fitness evaluation, batching,
                                      and MPI worker coordination

  `utilities/`                        Logging and helper functions

  `results/`, `checkpoints/`          Outputs and resumable GA state

  PBS scripts                         HPC resource requests and seasonal
                                      job submission
  -----------------------------------------------------------------------

*Directory names are logical roles; confirm exact paths against the
current repository.*

## Optimisation model

Each individual contains five design genes plus three operating genes
per representative day. For seven representative days, one individual
requires up to seven daily simulations.

Daily objective values are aggregated into the individual's seasonal
fitness. The GA then selects candidates and applies crossover and
mutation over successive generations. Check `config.py` and the
objective implementation for current bounds, units, penalty values, and
optimisation direction.

## Setup

Requirements: Python 3.12, `uv`, PySAM, DEAP, `mpi4py`, an MPI runtime
compatible with `mpi4py`, and DagsHub credentials for remote tracking.

``` bash
git clone https://github.com/Ary0Darkk/NYS-Design-Optimisation-using-PySAM.git
cd NYS-Design-Optimisation-using-PySAM
uv sync
source .venv/bin/activate

uv run python -c "import deap, mlflow, mpi4py, PySAM; print('Imports successful')"
```

Verify the cluster-supported MPI launcher separately. Installing
`mpi4py` alone does not guarantee that `mpirun` or `mpiexec` is
available.

## Run

Run one season locally:

``` bash
uv run python main.py --season winter
uv run python main.py --season summer
uv run python main.py --season monsoon
uv run python main.py --season post
```

Example local MPI run:

``` bash
LOCAL_WORKERS=4 mpirun -np 4 uv run python main.py --season winter
```

The MPI launcher must match the installed MPI implementation and cluster
configuration.

## HPC with PBS

A typical seven-node resource request:

``` bash
#PBS -q large
#PBS -l select=7:ncpus=112:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=12:00:00
```

Set `LOCAL_WORKERS` to match the allocated CPUs per MPI rank and prevent
numerical libraries from oversubscribing threads:

``` bash
export LOCAL_WORKERS=112
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpirun -np 7 uv run python main.py --season winter
```

The launcher above is an example, not a confirmed Praganak command. Use
the MPI module/launcher supported by the cluster. Check queue limits and
available resources before submitting.

Submit a seasonal PBS script with:

``` bash
qsub -v SEASON=winter job7.pbs
```

Use `summer`, `monsoon`, or `post` for the other seasons.

## Tracking, checkpoints, and reliability

-   MLflow records configured GA parameters, generation metrics, fitness
    statistics, and simulation success/penalty counts.
-   DagsHub provides remote experiment tracking when credentials are
    configured.
-   Initialise remote tracking from MPI rank 0 only.
-   Checkpoints support resuming compatible runs. Keep configuration and
    output paths consistent, and avoid concurrent jobs writing to the
    same checkpoint files.
-   Inspect both fitness and penalty rates when comparing results; a
    high fitness score is less meaningful if many simulations fail.

## Useful PBS commands

``` bash
qstat -u "$USER"       # List your jobs
qstat -f JOB_ID        # Inspect a job
qstat -xf JOB_ID       # Inspect a completed job, if retained
qstat -Qf large        # Inspect queue limits
```

## Reproducibility notes

Record the source revision, configuration, PySAM/SAM inputs, dependency
versions, random seeds, representative days, objective definition, and
checkpoint state for comparable runs.

## References

-   [NREL SAM](https://sam.nrel.gov/)
-   [PySAM documentation](https://nrel-pysam.readthedocs.io/)
-   [DEAP documentation](https://deap.readthedocs.io/)
-   [MLflow documentation](https://mlflow.org/docs/latest/)
-   [DagsHub](https://dagshub.com/)
