# Concentrated Solar Power (CSP) Plant Design & Operation Optimizer

A high-performance Genetic Algorithm (GA) optimization framework for Concentrated Solar Power (CSP) parabolic trough plants using PySAM and DEAP. Jointly optimizes physical solar field design parameters alongside dynamic daily operational control strategies across representative seasonal weather conditions.

## Key Features

- **26-Gene Hybrid Chromosome**: Optimizes 5 physical plant design variables and 21 seasonal operational controls (3 controls x 7 representative days per season).
- **Physical Feasibility Guardrails**: Enforces maintenance corridors (`Row_Distance - Aperture >= 3.5m`) and anti-chattering thermal deadbands (`T_startup - T_shutdown >= 15°C`).
- **Vectorized Economic Objective**: Maximizes net operational revenue against dynamic time-of-day tariffs while accounting for parasitic pumping and thermal losses.
- **Hybrid HPC Parallelism**: Scales across multi-node HPC clusters using hybrid MPI (inter-node coordination) and multiprocessing (intra-node core parallelization).
- **Artifact Export & Tracking**: Generates convergence curves, summary Excel workbooks, executive Markdown reports, and DagsHub / MLflow experiment tracking.

## Decision Variables

| Category | Parameter | Bounds | Description |
|---|---|---|---|
| **Design** | `solar_multiple` | [1.0, 4.0] | Solar field sizing multiple |
| | `tes_hours` | [0.0, 14.0] h | Thermal energy storage capacity |
| | `q_pb_design` | [50.0, 250.0] MWt | Power block design thermal rating |
| | `field_aperture` | [100000, 1500000] m² | Total aperture reflective area |
| | `row_distance` | [10.0, 30.0] m | Spacing between collector rows |
| **Operational** (x7 days) | `defocus_control` | [0.0, 1.0] | Optical collector defocus fraction |
| | `target_htf_temp` | [350.0, 400.0] °C | Target heat transfer fluid outlet temp |
| | `tes_dispatch_frac` | [0.0, 1.0] | Storage charge/discharge fraction |

## Installation

Ensure Python 3.12+ and `uv` are installed:

```bash
git clone https://github.com/Ary0Darkk/Solar-Plant-Design-Optimisation.git
cd Solar-Plant-Design-Optimisation
uv sync
```

## Configuration

All optimization, algorithm, and simulation settings are configured in `config.yaml`:

```yaml
ga:
  pop_size: 112
  num_generations: 50
  crossover_prob: 0.8
  mutation_prob: 0.2
  max_tasks_per_child: 50

simulation:
  max_simulation_workers: 112
```

## Running Optimization

### Local Workstation

```bash
uv run python main.py --season summer
```

Run all seasons sequentially:

```bash
uv run python main.py --season all
```

### HPC Multi-Node (PBS Pro)

Submit the PBS batch job:

```bash
qsub job.pbs
```

The script requests 4 compute nodes with 112 cores each (448 total cores), launching with:

```bash
mpirun python main.py --season summer
```

## Running Tests

```bash
uv run pytest
```

## Output Artifacts

Each completed run exports to:
- `results/GA_results/`: Excel sheets (`.xlsx`), summary reports (`.md`), and raw metrics (`.json`).
- `plots/GA_plots/`: High-resolution fitness convergence curves (`.png` and `.pdf`).
- `mlruns/`: Local MLflow experiment traces (or synced to remote DagsHub if credentials are set).
