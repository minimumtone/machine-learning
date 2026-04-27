#!/bin/bash
# POTCAR generation script
# Usage: bash make_potcar.sh
# Requires: $VASPPOT environment variable pointing to VASP pseudopotential directory
#   e.g., export VASPPOT=/path/to/potpaw_PBE.64

if [ -z "$VASPPOT" ]; then
    echo "Error: VASPPOT environment variable is not set."
    echo "Set it to the PAW-PBE pseudopotential directory, e.g.:"
    echo "  export VASPPOT=/path/to/potpaw_PBE.64"
    exit 1
fi

echo "Using VASPPOT=$VASPPOT"
echo "Generating POTCAR files for L12 calculations..."

# --- Fe3Mn ---
echo "  Fe3Mn: Fe_pv + Mn_pv"
cat "$VASPPOT"/Fe_pv/POTCAR "$VASPPOT"/Mn_pv/POTCAR > Fe3Mn/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Fe3Mn"; fi

# --- Mn3Fe ---
echo "  Mn3Fe: Mn_pv + Fe_pv"
cat "$VASPPOT"/Mn_pv/POTCAR "$VASPPOT"/Fe_pv/POTCAR > Mn3Fe/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Mn3Fe"; fi

# --- Cr3Mn ---
echo "  Cr3Mn: Cr_pv + Mn_pv"
cat "$VASPPOT"/Cr_pv/POTCAR "$VASPPOT"/Mn_pv/POTCAR > Cr3Mn/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Cr3Mn"; fi

# --- Mn3Cr ---
echo "  Mn3Cr: Mn_pv + Cr_pv"
cat "$VASPPOT"/Mn_pv/POTCAR "$VASPPOT"/Cr_pv/POTCAR > Mn3Cr/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Mn3Cr"; fi

# --- Al3Mn ---
echo "  Al3Mn: Al + Mn_pv"
cat "$VASPPOT"/Al/POTCAR "$VASPPOT"/Mn_pv/POTCAR > Al3Mn/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Al3Mn"; fi

# --- Mn3Al ---
echo "  Mn3Al: Mn_pv + Al"
cat "$VASPPOT"/Mn_pv/POTCAR "$VASPPOT"/Al/POTCAR > Mn3Al/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Mn3Al"; fi

# --- Cr3Mo ---
echo "  Cr3Mo: Cr_pv + Mo_pv"
cat "$VASPPOT"/Cr_pv/POTCAR "$VASPPOT"/Mo_pv/POTCAR > Cr3Mo/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Cr3Mo"; fi

# --- Mo3Cr ---
echo "  Mo3Cr: Mo_pv + Cr_pv"
cat "$VASPPOT"/Mo_pv/POTCAR "$VASPPOT"/Cr_pv/POTCAR > Mo3Cr/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Mo3Cr"; fi

# --- Fe3Mo ---
echo "  Fe3Mo: Fe_pv + Mo_pv"
cat "$VASPPOT"/Fe_pv/POTCAR "$VASPPOT"/Mo_pv/POTCAR > Fe3Mo/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Fe3Mo"; fi

# --- Mo3Fe ---
echo "  Mo3Fe: Mo_pv + Fe_pv"
cat "$VASPPOT"/Mo_pv/POTCAR "$VASPPOT"/Fe_pv/POTCAR > Mo3Fe/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Mo3Fe"; fi

# --- Mo3Ni ---
echo "  Mo3Ni: Mo_pv + Ni_pv"
cat "$VASPPOT"/Mo_pv/POTCAR "$VASPPOT"/Ni_pv/POTCAR > Mo3Ni/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Mo3Ni"; fi

# --- Ni3Mo ---
echo "  Ni3Mo: Ni_pv + Mo_pv"
cat "$VASPPOT"/Ni_pv/POTCAR "$VASPPOT"/Mo_pv/POTCAR > Ni3Mo/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ni3Mo"; fi

# --- Ir3Pd ---
echo "  Ir3Pd: Ir + Pd"
cat "$VASPPOT"/Ir/POTCAR "$VASPPOT"/Pd/POTCAR > Ir3Pd/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ir3Pd"; fi

# --- Pd3Ir ---
echo "  Pd3Ir: Pd + Ir"
cat "$VASPPOT"/Pd/POTCAR "$VASPPOT"/Ir/POTCAR > Pd3Ir/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pd3Ir"; fi

# --- Ir3Ru ---
echo "  Ir3Ru: Ir + Ru_pv"
cat "$VASPPOT"/Ir/POTCAR "$VASPPOT"/Ru_pv/POTCAR > Ir3Ru/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ir3Ru"; fi

# --- Ru3Ir ---
echo "  Ru3Ir: Ru_pv + Ir"
cat "$VASPPOT"/Ru_pv/POTCAR "$VASPPOT"/Ir/POTCAR > Ru3Ir/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ru3Ir"; fi

# --- Pd3Pt ---
echo "  Pd3Pt: Pd + Pt"
cat "$VASPPOT"/Pd/POTCAR "$VASPPOT"/Pt/POTCAR > Pd3Pt/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pd3Pt"; fi

# --- Pt3Pd ---
echo "  Pt3Pd: Pt + Pd"
cat "$VASPPOT"/Pt/POTCAR "$VASPPOT"/Pd/POTCAR > Pt3Pd/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pt3Pd"; fi

# --- Pd3Rh ---
echo "  Pd3Rh: Pd + Rh_pv"
cat "$VASPPOT"/Pd/POTCAR "$VASPPOT"/Rh_pv/POTCAR > Pd3Rh/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pd3Rh"; fi

# --- Rh3Pd ---
echo "  Rh3Pd: Rh_pv + Pd"
cat "$VASPPOT"/Rh_pv/POTCAR "$VASPPOT"/Pd/POTCAR > Rh3Pd/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Rh3Pd"; fi

# --- Pd3Ru ---
echo "  Pd3Ru: Pd + Ru_pv"
cat "$VASPPOT"/Pd/POTCAR "$VASPPOT"/Ru_pv/POTCAR > Pd3Ru/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pd3Ru"; fi

# --- Ru3Pd ---
echo "  Ru3Pd: Ru_pv + Pd"
cat "$VASPPOT"/Ru_pv/POTCAR "$VASPPOT"/Pd/POTCAR > Ru3Pd/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ru3Pd"; fi

# --- Pt3Ru ---
echo "  Pt3Ru: Pt + Ru_pv"
cat "$VASPPOT"/Pt/POTCAR "$VASPPOT"/Ru_pv/POTCAR > Pt3Ru/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pt3Ru"; fi

# --- Ru3Pt ---
echo "  Ru3Pt: Ru_pv + Pt"
cat "$VASPPOT"/Ru_pv/POTCAR "$VASPPOT"/Pt/POTCAR > Ru3Pt/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ru3Pt"; fi

# --- Ni3Pd ---
echo "  Ni3Pd: Ni_pv + Pd"
cat "$VASPPOT"/Ni_pv/POTCAR "$VASPPOT"/Pd/POTCAR > Ni3Pd/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ni3Pd"; fi

# --- Pd3Ni ---
echo "  Pd3Ni: Pd + Ni_pv"
cat "$VASPPOT"/Pd/POTCAR "$VASPPOT"/Ni_pv/POTCAR > Pd3Ni/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pd3Ni"; fi

# --- Os3Pd ---
echo "  Os3Pd: Os_pv + Pd"
cat "$VASPPOT"/Os_pv/POTCAR "$VASPPOT"/Pd/POTCAR > Os3Pd/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Os3Pd"; fi

# --- Pd3Os ---
echo "  Pd3Os: Pd + Os_pv"
cat "$VASPPOT"/Pd/POTCAR "$VASPPOT"/Os_pv/POTCAR > Pd3Os/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pd3Os"; fi

# --- Os3Pt ---
echo "  Os3Pt: Os_pv + Pt"
cat "$VASPPOT"/Os_pv/POTCAR "$VASPPOT"/Pt/POTCAR > Os3Pt/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Os3Pt"; fi

# --- Pt3Os ---
echo "  Pt3Os: Pt + Os_pv"
cat "$VASPPOT"/Pt/POTCAR "$VASPPOT"/Os_pv/POTCAR > Pt3Os/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Pt3Os"; fi

# --- Os3Rh ---
echo "  Os3Rh: Os_pv + Rh_pv"
cat "$VASPPOT"/Os_pv/POTCAR "$VASPPOT"/Rh_pv/POTCAR > Os3Rh/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Os3Rh"; fi

# --- Rh3Os ---
echo "  Rh3Os: Rh_pv + Os_pv"
cat "$VASPPOT"/Rh_pv/POTCAR "$VASPPOT"/Os_pv/POTCAR > Rh3Os/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Rh3Os"; fi

# --- Os3Ru ---
echo "  Os3Ru: Os_pv + Ru_pv"
cat "$VASPPOT"/Os_pv/POTCAR "$VASPPOT"/Ru_pv/POTCAR > Os3Ru/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Os3Ru"; fi

# --- Ru3Os ---
echo "  Ru3Os: Ru_pv + Os_pv"
cat "$VASPPOT"/Ru_pv/POTCAR "$VASPPOT"/Os_pv/POTCAR > Ru3Os/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ru3Os"; fi

# --- Cr3V ---
echo "  Cr3V: Cr_pv + V_sv"
cat "$VASPPOT"/Cr_pv/POTCAR "$VASPPOT"/V_sv/POTCAR > Cr3V/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Cr3V"; fi

# --- V3Cr ---
echo "  V3Cr: V_sv + Cr_pv"
cat "$VASPPOT"/V_sv/POTCAR "$VASPPOT"/Cr_pv/POTCAR > V3Cr/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for V3Cr"; fi

# --- Fe3V ---
echo "  Fe3V: Fe_pv + V_sv"
cat "$VASPPOT"/Fe_pv/POTCAR "$VASPPOT"/V_sv/POTCAR > Fe3V/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Fe3V"; fi

# --- V3Fe ---
echo "  V3Fe: V_sv + Fe_pv"
cat "$VASPPOT"/V_sv/POTCAR "$VASPPOT"/Fe_pv/POTCAR > V3Fe/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for V3Fe"; fi

# --- Ni3V ---
echo "  Ni3V: Ni_pv + V_sv"
cat "$VASPPOT"/Ni_pv/POTCAR "$VASPPOT"/V_sv/POTCAR > Ni3V/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Ni3V"; fi

# --- V3Ni ---
echo "  V3Ni: V_sv + Ni_pv"
cat "$VASPPOT"/V_sv/POTCAR "$VASPPOT"/Ni_pv/POTCAR > V3Ni/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for V3Ni"; fi

# --- Al3Cr ---
echo "  Al3Cr: Al + Cr_pv"
cat "$VASPPOT"/Al/POTCAR "$VASPPOT"/Cr_pv/POTCAR > Al3Cr/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Al3Cr"; fi

# --- Cr3Al ---
echo "  Cr3Al: Cr_pv + Al"
cat "$VASPPOT"/Cr_pv/POTCAR "$VASPPOT"/Al/POTCAR > Cr3Al/POTCAR
if [ $? -ne 0 ]; then echo "  WARNING: Failed for Cr3Al"; fi

echo "Done. Generated POTCAR for all calculations."
