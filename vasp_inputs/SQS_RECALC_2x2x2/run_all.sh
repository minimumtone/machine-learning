#!/bin/bash
# Run all SQS_RECALC_2x2x2 calculations.
# Usage: bash run_all.sh [NJOBS_PARALLEL] [NPROCS_PER_JOB]   (default 8x4)
# Requires: $VASPBIN, $VASP_PP_PATH
set -e
NJOBS=${1:-8}
NP=${2:-4}
BASEDIR=$(cd "$(dirname "$0")" && pwd)
cd "$BASEDIR"
bash make_potcar.sh

run_one() {
    d="$1"
    cd "$BASEDIR/$d"
    if [ -f static_OUTCAR ] || grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "SKIP $d"; return 0
    fi
    echo "START $d"
    mpirun -np $NP "$VASPBIN" > vasp.out 2>&1 || { echo "FAIL $d"; return 0; }
    # static run at relaxed geometry
    cp CONTCAR POSCAR
    sed -e 's/NSW    = 200/NSW    = 0/' -e 's/IBRION = 2/IBRION = -1/' INCAR > INCAR.static
    mv INCAR INCAR.relax && mv INCAR.static INCAR
    mpirun -np $NP "$VASPBIN" > vasp_static.out 2>&1 || echo "FAIL-STATIC $d"
    cp OUTCAR static_OUTCAR
    mv INCAR.relax INCAR
    echo "DONE $d"
}
export -f run_one
export BASEDIR NP VASPBIN

find . -mindepth 2 -maxdepth 2 -type d | sed 's|^\./||' | \
    xargs -P $NJOBS -I{} bash -c 'run_one "$@"' _ {}
echo "All done. Extract with: python ../extract_vasp_results.py"
