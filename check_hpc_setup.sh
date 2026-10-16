#!/bin/bash

# =========================================================
# HPC Pre-flight Verification Script for NYS-PySAM
# Run on the HPC login node:
#   bash check_hpc_setup.sh
# =========================================================

echo "========================================================="
echo "  NYS-PySAM HPC Pre-flight Diagnostic Check"
echo "========================================================="

ERRORS=0
WARNINGS=0

# 1. Clean line endings
echo ""
echo "[1/5] Checking line endings of PBS scripts..."
for script in *.pbs *.sh; do
    if [ -f "$script" ]; then
        # Automatically strip carriage returns
        if grep -q $'\r' "$script" 2>/dev/null || file "$script" 2>/dev/null | grep -q "CRLF"; then
            echo "  [FIXING] $script had Windows CRLF line endings -> converting to UNIX LF..."
            sed -i 's/\r$//' "$script"
            echo "  [OK]     $script converted to LF."
        else
            echo "  [OK]     $script has clean UNIX LF line endings."
        fi
    fi
done

# 2. Check Python virtual environment
echo ""
echo "[2/5] Checking Python virtual environment (.venv)..."
if [ -f ".venv/bin/activate" ]; then
    echo "  [OK]   .venv/bin/activate exists."
    source .venv/bin/activate
    echo "         Active Python: $(which python)"
else
    echo "  [FAIL] .venv/bin/activate NOT found!"
    echo "         On Linux HPC, please run: uv sync"
    echo "         Or: python3 -m venv .venv && source .venv/bin/activate && pip install -r pyproject.toml"
    ERRORS=$((ERRORS + 1))
fi

# 3. Test core package imports
echo ""
echo "[3/5] Testing Python dependencies..."
if command -v python >/dev/null 2>&1; then
    python -c "
import sys
failures = []
for pkg in ['PySAM', 'deap', 'mpi4py', 'mlflow', 'pandas', 'openpyxl']:
    try:
        __import__(pkg)
    except ImportError:
        failures.append(pkg)
if failures:
    print('  [FAIL] Missing packages:', ', '.join(failures))
    sys.exit(1)
else:
    print('  [OK]   All core dependencies imported successfully.')
"
    if [ $? -ne 0 ]; then
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  [FAIL] Python interpreter not available."
    ERRORS=$((ERRORS + 1))
fi

# 4. Check MPI runtime availability
echo ""
echo "[4/5] Checking MPI launcher..."
if command -v mpirun >/dev/null 2>&1; then
    echo "  [OK]   mpirun found: $(which mpirun)"
else
    echo "  [FAIL] mpirun not found in PATH!"
    echo "         You likely need to load an HPC module, e.g.:"
    echo "           module load openmpi"
    echo "         Run 'module avail' to see available MPI modules on your cluster."
    ERRORS=$((ERRORS + 1))
fi

# 5. Check directories
echo ""
echo "[5/5] Ensuring output directories exist..."
mkdir -p logs results/GA_results checkpoints/GA
echo "  [OK]   logs/ and results/ directories ready."

echo ""
echo "========================================================="
if [ $ERRORS -eq 0 ]; then
    echo "  Summary: ALL CHECKS PASSED!"
    echo "  Ready to submit:"
    echo "    qsub -v SEASON=summer job7.pbs"
else
    echo "  Summary: $ERRORS error(s) found. Please resolve above items."
fi
echo "========================================================="
exit $ERRORS
