#!/bin/bash
# Batch execution script for L12 DFT calculations
# Usage: bash run_all.sh
#
# Adjust VASP_CMD and NPROCS to your environment.
# For NIMS GPU cluster, modify the job submission commands as needed.

VASP_CMD="mpirun -np ${NPROCS:-16} vasp_std"
BASE_DIR=$(cd $(dirname $0) && pwd)

# --- Job submission function ---
run_calc() {
    local dir="$1"
    local name=$(basename "$dir")
    cd "$dir"

    if [ -f "OUTCAR" ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "SKIP: $name (already converged)"
        cd "$BASE_DIR"
        return 0
    fi

    if [ ! -f "POTCAR" ]; then
        echo "ERROR: $name - POTCAR not found. Run make_potcar.sh first."
        cd "$BASE_DIR"
        return 1
    fi

    echo "RUN: $name"
    $VASP_CMD > vasp.log 2>&1

    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  OK: $name converged"
    else
        echo "  WARNING: $name may not have converged"
    fi
    cd "$BASE_DIR"
}

# --- Run all calculations ---
echo "Starting L12 DFT calculations..."
echo "Total: 42 calculations"

run_calc "$BASE_DIR/Fe3Mn"
run_calc "$BASE_DIR/Mn3Fe"
run_calc "$BASE_DIR/Cr3Mn"
run_calc "$BASE_DIR/Mn3Cr"
run_calc "$BASE_DIR/Al3Mn"
run_calc "$BASE_DIR/Mn3Al"
run_calc "$BASE_DIR/Cr3Mo"
run_calc "$BASE_DIR/Mo3Cr"
run_calc "$BASE_DIR/Fe3Mo"
run_calc "$BASE_DIR/Mo3Fe"
run_calc "$BASE_DIR/Mo3Ni"
run_calc "$BASE_DIR/Ni3Mo"
run_calc "$BASE_DIR/Ir3Pd"
run_calc "$BASE_DIR/Pd3Ir"
run_calc "$BASE_DIR/Ir3Ru"
run_calc "$BASE_DIR/Ru3Ir"
run_calc "$BASE_DIR/Pd3Pt"
run_calc "$BASE_DIR/Pt3Pd"
run_calc "$BASE_DIR/Pd3Rh"
run_calc "$BASE_DIR/Rh3Pd"
run_calc "$BASE_DIR/Pd3Ru"
run_calc "$BASE_DIR/Ru3Pd"
run_calc "$BASE_DIR/Pt3Ru"
run_calc "$BASE_DIR/Ru3Pt"
run_calc "$BASE_DIR/Ni3Pd"
run_calc "$BASE_DIR/Pd3Ni"
run_calc "$BASE_DIR/Os3Pd"
run_calc "$BASE_DIR/Pd3Os"
run_calc "$BASE_DIR/Os3Pt"
run_calc "$BASE_DIR/Pt3Os"
run_calc "$BASE_DIR/Os3Rh"
run_calc "$BASE_DIR/Rh3Os"
run_calc "$BASE_DIR/Os3Ru"
run_calc "$BASE_DIR/Ru3Os"
run_calc "$BASE_DIR/Cr3V"
run_calc "$BASE_DIR/V3Cr"
run_calc "$BASE_DIR/Fe3V"
run_calc "$BASE_DIR/V3Fe"
run_calc "$BASE_DIR/Ni3V"
run_calc "$BASE_DIR/V3Ni"
run_calc "$BASE_DIR/Al3Cr"
run_calc "$BASE_DIR/Cr3Al"

echo "All calculations completed."
