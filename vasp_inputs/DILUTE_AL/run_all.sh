#!/bin/bash
# Batch execution for dilute-Al calculations
# Runs: mpirun -np $NP $VASPBIN (default NP=8)

if [ -z "$VASPBIN" ]; then
    echo "Error: VASPBIN is not set."; exit 1
fi
NP="${NP:-8}"

BASE=$(cd "$(dirname "$0")" && pwd)
LOG="$BASE/run_status.log"
echo "=== Dilute-Al calculations ===" | tee "$LOG"
echo "Total: 60" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

echo "[1/60] fcc_Ni_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Ni_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[2/60] fcc_Ni_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Ni_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[3/60] fcc_Ni_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Ni_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[4/60] fcc_Ni_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Ni_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[5/60] fcc_Ni_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Ni_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[6/60] fcc_Ni_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Ni_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[7/60] fcc_Co_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Co_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[8/60] fcc_Co_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Co_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[9/60] fcc_Co_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Co_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[10/60] fcc_Co_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Co_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[11/60] fcc_Co_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Co_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[12/60] fcc_Co_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Co_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[13/60] fcc_Pd_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Pd_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[14/60] fcc_Pd_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Pd_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[15/60] fcc_Pd_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Pd_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[16/60] fcc_Pd_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Pd_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[17/60] fcc_Pd_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Pd_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[18/60] fcc_Pd_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Pd_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[19/60] fcc_Rh_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Rh_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[20/60] fcc_Rh_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Rh_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[21/60] fcc_Rh_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Rh_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[22/60] fcc_Rh_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Rh_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[23/60] fcc_Rh_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Rh_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[24/60] fcc_Rh_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Rh_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[25/60] fcc_Ir_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Ir_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[26/60] fcc_Ir_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Ir_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[27/60] fcc_Ir_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Ir_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[28/60] fcc_Ir_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Ir_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[29/60] fcc_Ir_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Ir_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[30/60] fcc_Ir_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Ir_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[31/60] fcc_Cu_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Cu_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[32/60] fcc_Cu_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Cu_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[33/60] fcc_Cu_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Cu_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[34/60] fcc_Cu_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Cu_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[35/60] fcc_Cu_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Cu_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[36/60] fcc_Cu_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Cu_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[37/60] fcc_Ag_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Ag_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[38/60] fcc_Ag_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Ag_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[39/60] fcc_Ag_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Ag_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[40/60] fcc_Ag_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Ag_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[41/60] fcc_Ag_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Ag_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[42/60] fcc_Ag_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Ag_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[43/60] fcc_Au_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Au_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[44/60] fcc_Au_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Au_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[45/60] fcc_Au_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Au_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[46/60] fcc_Au_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Au_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[47/60] fcc_Au_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Au_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[48/60] fcc_Au_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Au_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[49/60] fcc_Pt_n0..." | tee -a "$LOG"
cd "$BASE/fcc_Pt_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[50/60] fcc_Pt_n1..." | tee -a "$LOG"
cd "$BASE/fcc_Pt_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[51/60] fcc_Pt_n2..." | tee -a "$LOG"
cd "$BASE/fcc_Pt_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[52/60] fcc_Pt_n3..." | tee -a "$LOG"
cd "$BASE/fcc_Pt_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[53/60] fcc_Pt_n4..." | tee -a "$LOG"
cd "$BASE/fcc_Pt_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[54/60] fcc_Pt_imp108..." | tee -a "$LOG"
cd "$BASE/fcc_Pt_imp108"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[55/60] bcc_Nb_n0..." | tee -a "$LOG"
cd "$BASE/bcc_Nb_n0"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[56/60] bcc_Nb_n1..." | tee -a "$LOG"
cd "$BASE/bcc_Nb_n1"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[57/60] bcc_Nb_n2..." | tee -a "$LOG"
cd "$BASE/bcc_Nb_n2"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[58/60] bcc_Nb_n3..." | tee -a "$LOG"
cd "$BASE/bcc_Nb_n3"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[59/60] bcc_Nb_n4..." | tee -a "$LOG"
cd "$BASE/bcc_Nb_n4"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "[60/60] bcc_Nb_imp128..." | tee -a "$LOG"
cd "$BASE/bcc_Nb_imp128"
if [ -f CONTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  SKIP (already converged)" | tee -a "$LOG"
elif [ ! -f POTCAR ]; then
    echo "  SKIP (no POTCAR)" | tee -a "$LOG"
else
    mpirun -np $NP $VASPBIN > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED" | tee -a "$LOG"
    else
        echo "  WARNING: not converged" | tee -a "$LOG"
    fi
fi
cd "$BASE"

echo "Finished: $(date)" | tee -a "$LOG"
