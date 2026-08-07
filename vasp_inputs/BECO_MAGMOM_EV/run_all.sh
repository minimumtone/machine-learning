#!/bin/bash
# Run Be-Co MAGMOM and fixed-volume E--V calculations.
# Usage: bash run_all.sh [NPROCS] (default: 8)
set -u
NPROCS="${1:-8}"
VASPBIN="${VASPBIN:?Set VASPBIN to the VASP executable.}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"
bash make_potcar.sh

cd "$BASE_DIR/MAGMOM/NM"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in MAGMOM/NM" >&2; exit 1; fi
echo "START MAGMOM/NM"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL MAGMOM/NM"
echo "DONE MAGMOM/NM"

cd "$BASE_DIR/MAGMOM/FM_low"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in MAGMOM/FM_low" >&2; exit 1; fi
echo "START MAGMOM/FM_low"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL MAGMOM/FM_low"
echo "DONE MAGMOM/FM_low"

cd "$BASE_DIR/MAGMOM/FM_ref"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in MAGMOM/FM_ref" >&2; exit 1; fi
echo "START MAGMOM/FM_ref"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL MAGMOM/FM_ref"
echo "DONE MAGMOM/FM_ref"

cd "$BASE_DIR/MAGMOM/FM_high"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in MAGMOM/FM_high" >&2; exit 1; fi
echo "START MAGMOM/FM_high"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL MAGMOM/FM_high"
echo "DONE MAGMOM/FM_high"

cd "$BASE_DIR/MAGMOM/AFM"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in MAGMOM/AFM" >&2; exit 1; fi
echo "START MAGMOM/AFM"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL MAGMOM/AFM"
echo "DONE MAGMOM/AFM"

cd "$BASE_DIR/EV/FM_ref/V0p94"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/FM_ref/V0p94" >&2; exit 1; fi
echo "START EV/FM_ref/V0p94"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/FM_ref/V0p94"
echo "DONE EV/FM_ref/V0p94"

cd "$BASE_DIR/EV/FM_ref/V0p96"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/FM_ref/V0p96" >&2; exit 1; fi
echo "START EV/FM_ref/V0p96"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/FM_ref/V0p96"
echo "DONE EV/FM_ref/V0p96"

cd "$BASE_DIR/EV/FM_ref/V0p98"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/FM_ref/V0p98" >&2; exit 1; fi
echo "START EV/FM_ref/V0p98"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/FM_ref/V0p98"
echo "DONE EV/FM_ref/V0p98"

cd "$BASE_DIR/EV/FM_ref/V1p00"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/FM_ref/V1p00" >&2; exit 1; fi
echo "START EV/FM_ref/V1p00"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/FM_ref/V1p00"
echo "DONE EV/FM_ref/V1p00"

cd "$BASE_DIR/EV/FM_ref/V1p02"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/FM_ref/V1p02" >&2; exit 1; fi
echo "START EV/FM_ref/V1p02"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/FM_ref/V1p02"
echo "DONE EV/FM_ref/V1p02"

cd "$BASE_DIR/EV/FM_ref/V1p04"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/FM_ref/V1p04" >&2; exit 1; fi
echo "START EV/FM_ref/V1p04"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/FM_ref/V1p04"
echo "DONE EV/FM_ref/V1p04"

cd "$BASE_DIR/EV/FM_ref/V1p06"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/FM_ref/V1p06" >&2; exit 1; fi
echo "START EV/FM_ref/V1p06"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/FM_ref/V1p06"
echo "DONE EV/FM_ref/V1p06"

cd "$BASE_DIR/EV/NM/V0p94"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/NM/V0p94" >&2; exit 1; fi
echo "START EV/NM/V0p94"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/NM/V0p94"
echo "DONE EV/NM/V0p94"

cd "$BASE_DIR/EV/NM/V0p96"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/NM/V0p96" >&2; exit 1; fi
echo "START EV/NM/V0p96"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/NM/V0p96"
echo "DONE EV/NM/V0p96"

cd "$BASE_DIR/EV/NM/V0p98"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/NM/V0p98" >&2; exit 1; fi
echo "START EV/NM/V0p98"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/NM/V0p98"
echo "DONE EV/NM/V0p98"

cd "$BASE_DIR/EV/NM/V1p00"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/NM/V1p00" >&2; exit 1; fi
echo "START EV/NM/V1p00"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/NM/V1p00"
echo "DONE EV/NM/V1p00"

cd "$BASE_DIR/EV/NM/V1p02"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/NM/V1p02" >&2; exit 1; fi
echo "START EV/NM/V1p02"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/NM/V1p02"
echo "DONE EV/NM/V1p02"

cd "$BASE_DIR/EV/NM/V1p04"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/NM/V1p04" >&2; exit 1; fi
echo "START EV/NM/V1p04"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/NM/V1p04"
echo "DONE EV/NM/V1p04"

cd "$BASE_DIR/EV/NM/V1p06"
if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in EV/NM/V1p06" >&2; exit 1; fi
echo "START EV/NM/V1p06"
mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || echo "FAIL EV/NM/V1p06"
echo "DONE EV/NM/V1p06"

echo "All calculations finished."
