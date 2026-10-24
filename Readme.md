# Concentrated Solar Power (CSP) Plant Design & Operation Optimizer

A high-performance, season-wise Genetic Algorithm (GA) optimization framework for Concentrated Solar Power (CSP) parabolic trough plants. Powered by **NREL PySAM**, **DEAP**, **NumPy**, and **MPI**, the pipeline optimizes physical solar field design parameters alongside dynamic daily operational control strategies across representative seasonal weather conditions.

---

## Key Features

- **26-Gene Hybrid Chromosome**: Jointly optimizes 5 physical plant design variables and 21 seasonal operational variables (3 variables across 7 representative days per season).
- **24-Hour PySAM Slicing**: Bypasses legacy 8,760-hour simulation overhead by isolating exact 24-hour physical simulation windows, slashing memory usage by 99.8% (~672 bytes per slice vs ~490 KB).
- **Physical Feasibility Guardrails**:
  - *Maintenance Corridor Clearance*: Enforces $\text{Row\_Distance} - W_{\text{aperture}} \ge 3.5\,\text{m}$.
  - *Anti-Chattering Thermal Deadband*: Enforces $T_{\text{startup}} - T_{\text{shutdown}} \ge 15\,^\circ\text{C}$.
- **Vectorized Economic Objective Function**: Maximizes net operational revenue against dynamic time-of-day tariffs while penalizing parasitic pumping loads and thermal losses with smart overcast standby logic.
- **Adaptive Execution Scaling**: Automatically detects execution environment; targets **112 worker processes** on HPC clusters (PBS / Slurm) and safely scales to **4 workers** on local workstations to prevent thrashing.
- **True Elitism & Gzip State Checkpointing**: Preserves Hall of Fame elites across generations and atomically persists checkpoint state with deterministic configuration signatures.
- **Multi-Format Publication Artifacts**: Exports high-resolution 300 DPI PNG and vector PDF convergence curves, multi-sheet Excel workbooks, executive Markdown reports, CSV/JSON metrics, and DagsHub/MLflow experiment logs.

---

## Mathematical Formulation

### 1. Objective Function

For each season $S$, the optimization maximizes the sum of daily net operational revenue across 7 representative days:

$$\max_{\mathbf{x}} \quad \mathcal{F}(\mathbf{x}) = \sum_{d=1}^{7} \text{Net Revenue}_d$$

$$\text{Net Revenue}_d = \sum_{t=1}^{24} \Big( E_t \cdot P_t - P_{\text{parasitic}, t} \cdot P_t - 0.4 \cdot Q_{\text{loss}, t} \cdot P_t \Big)$$

Where:
- $E_t$: Net electrical energy generated at hour $t$ [MWh].
- $P_t$: Dynamic time-of-day tariff at hour $t$ [INR/MWh].
- $P_{\text{parasitic}, t}$: Pumping and solar tracker power consumption [MWh] ($W_{\text{field pump}} + W_{\text{cycle pump}} + W_{\text{SCA track}}$).
- $Q_{\text{loss}, t}$: Total thermal losses ($Q_{\text{startup}} + Q_{\text{piping}} + Q_{\text{receiver}}$) scaled by thermal-to-electric conversion efficiency ($\eta \approx 0.40$).
- *Standby Logic*: If $E_t = 0$ and net value is negative (cloudy/night hours), net cost is clamped to $0.0\,\text{INR}$ to reflect field standby mode.

### 2. Decision Variables (26-Gene Chromosome)

| Category | Parameter | Description | Bounds | Units |
| :--- | :--- | :--- | :--- | :--- |
| **Design** (5) | `specified_total_aperture` | Total reflective aperture area | $[8000, 12000]$ | $\text{m}^2$ |
| | `Row_Distance` | Pitch distance between collector rows | $[10, 25]$ | $\text{m}$ |
| | `ColperSCA` | Collector modules per Solar Collector Assembly | $[2, 10]$ | count |
| | `W_aperture` | Reflective aperture width | $[4, 8]$ | $\text{m}$ |
| | `L_SCA` | Single assembly length | $[50, 200]$ | $\text{m}$ |
| **Operational** (3 per day $\times$ 7 days) | `m_dot` | Heat transfer fluid (HTF) mass flow rate limit | $[2, 12]$ | $\text{kg/s}$ |
| | `T_startup` | Field startup temperature threshold | $[300, 380]$ | $^\circ\text{C}$ |
| | `T_shutdown` | Field shutdown temperature threshold | $[250, 300]$ | $^\circ\text{C}$ |

---

## Repository Structure

```text
Solar-Plant-Design-Optimisation/
├── config.yaml                    # Global pipeline configuration
├── job.pbs                        # Unified HPC PBS batch script
├── main.py                        # Primary entry point
├── pyproject.toml                 # Dependencies, scripts, and build configuration
├── clean_mlflow_runs.py           # Remote DagsHub experiment management utility
├── electricity_data/              # Demand profiles and dynamic tariff data
├── simulation/                    # Plant configuration and weather datasets
├── solar_opt/                     # Core Optimization Package
│   ├── checkpoint.py              # Atomic gzip-compressed state persistence
│   ├── chromosome.py              # 26-gene chromosome encoder & DEAP types
│   ├── cli.py                     # CLI parser with scheduler detection
│   ├── config.py                  # Dataclass models & YAML configuration
│   ├── engine.py                  # Genetic algorithm engine & elitism loop
│   ├── objective.py               # Vectorized daily economic objective
│   ├── operators.py               # Bounded Gaussian mutation & crossover
│   ├── parallel.py                # MPI communicator & standalone fallback
│   ├── simulation.py              # 24-hour PySAM simulation & memory cleanup
│   ├── tariff.py                  # Read-only float32 tariff caching
│   ├── tracker.py                 # Multi-format artifact exporters & MLflow
│   └── weather.py                 # Representative seasonal calendar mapping
└── tests/                         # Pytest test suite
```

---

## Installation & Setup

### Prerequisites
- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/) (recommended for ultrafast environment management)

```bash
# Clone the repository
git clone https://github.com/Ary0Darkk/Solar-Plant-Design-Optimisation.git
cd Solar-Plant-Design-Optimisation

# Install dependencies using uv
uv sync

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

---

## Running the Optimizer

### Command-Line Interface (CLI)

```bash
python main.py --help
```

```text
options:
  -h, --help            Show this help message and exit
  --season {winter,summer,monsoon,post}
                        Season to optimize (default: summer)
  --generations GENERATIONS
                        Number of GA generations override (default: 100)
  --pop-size POP_SIZE   Population size override (default: 112)
  --workers WORKERS     Worker processes per rank (default: 112 on HPC, 4 locally)
  --max-tasks-per-child MAX_TASKS_PER_CHILD
                        Recycle worker process after N simulations (default: 25)
  --seed SEED           Random seed for reproducibility (default: 41)
  --min-price MIN_PRICE Minimum electricity tariff in INR/kWh (default: 3.0)
  --max-price MAX_PRICE Maximum electricity tariff in INR/kWh (default: 8.0)
  --job-id JOB_ID       Scheduler Job ID (auto-detected from PBS_JOBID / SLURM_JOB_ID)
  --config CONFIG       Path to YAML configuration file (default: config.yaml)
  --experiment EXPERIMENT
                        MLflow / DagsHub experiment name (default: CSP_Seasonal_GA)
  --no-tracking         Disable remote experiment tracking
  --simulate            Execute a single 24-hour PySAM simulation and display results
  --day DAY             Representative day index (1-7) for single simulation (default: 1)
  --log-mlflow          Log single simulation results as a run in MLflow
```

### Experiment Tracking & DagsHub MLflow Dashboard

The pipeline integrates directly with cloud-hosted **MLflow on DagsHub** with zero local database or disk storage footprint:

- **Centralized Cloud Dashboard**: Access runs, generational metric charts, and visual artifacts directly at:  
  👉 **[DagsHub MLflow Server](https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow)**
- **Cloud Metrics & Plots**: Generational convergence curves (Max, Avg, Min fitness), evaluation counts, and timing metrics stream directly to the cloud.
- **Published Artifacts**: High-resolution 300 DPI PNG plots, vector PDFs, Excel workbooks, and executive markdown reports are uploaded directly to the remote DagsHub artifact store.
- **Offline / Standalone Runs**: Pass `--no-tracking` to run simulations and GA optimizations completely offline without network calls.

---

### Local Execution
On local workstations, the pipeline automatically scales to 4 worker processes to prevent memory thrashing:

```bash
# Fast test run on winter season
python main.py --season winter --pop-size 14 --generations 5

# Full summer optimization
python main.py --season summer

# Offline run without remote tracking
python main.py --season post --no-tracking
```

### HPC Batch Execution (PBS)

Submit batch jobs to HPC clusters (e.g. C-DAC Pratyush, Param):

```bash
# Submit summer season optimization
qsub job.pbs

# Submit winter season optimization
qsub -v SEASON=winter job.pbs

# Optimize all four seasons sequentially in one job
qsub -v SEASON=all job.pbs
```

---

## Output Artifacts

Every completed optimization run writes five distinct artifact formats into `results/GA_results/` and `plots/GA_plots/`:

1. **Dual Convergence Plots**:
   - High-resolution **300 DPI PNG** (`<season>_convergence_<timestamp>.png`).
   - Publication-ready **vector PDF** (`<season>_convergence_<timestamp>.pdf`).
2. **Multi-Sheet Excel Summary** (`<season>_summary_<timestamp>.xlsx`):
   - *Optimal Solution*: Best fitness, job ID, and 5 plant design variables.
   - *GA History*: Generational statistics (`gen`, `nevals`, `avg`, `std`, `min`, `max`).
   - *Operational Schedule*: Daily schedule across all 7 representative days.
3. **Executive Markdown Report** (`<season>_executive_report_<timestamp>.md`):
   - Formatted markdown summary with design parameters, operational schedule, and fitness gains.
4. **Machine-Readable Outputs**:
   - Structured JSON solution (`best_solution_<season>.json`).
   - GA evolutionary logbook CSV.
5. **DagsHub / MLflow Remote Tracking**:
   - Automatically records scheduler metadata, generational metrics, and logs all generated artifacts.

---

## Testing & Code Quality

```bash
# Run pytest test suite
uv run pytest -v

# Run code formatting and linting checks
uv run ruff check .
uv run ruff format --check .
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
