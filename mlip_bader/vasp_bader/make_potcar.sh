#!/usr/bin/env bash
# Generate POTCAR files using the repository's PBE POTCAR variants.
set -euo pipefail
if [[ -z "${VASP_PP_PATH:-}" ]]; then echo "Set VASP_PP_PATH first"; exit 1; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
PP_DIR="$VASP_PP_PATH/potpaw_PBE"

cat "$PP_DIR"/Hf_pv/POTCAR "$PP_DIR"/Nb_pv/POTCAR "$PP_DIR"/Ta_pv/POTCAR "$PP_DIR"/Ti_pv/POTCAR "$PP_DIR"/Zr_sv/POTCAR > "HfNbTaTiZr/POTCAR"
cat "$PP_DIR"/Al/POTCAR "$PP_DIR"/Nb_pv/POTCAR "$PP_DIR"/Ti_pv/POTCAR "$PP_DIR"/V_sv/POTCAR > "AlNbTiV/POTCAR"
cat "$PP_DIR"/Al/POTCAR "$PP_DIR"/Nb_pv/POTCAR > "Al-Nb/POTCAR"
cat "$PP_DIR"/Al/POTCAR "$PP_DIR"/V_sv/POTCAR > "Al-V/POTCAR"
cat "$PP_DIR"/Nb_pv/POTCAR "$PP_DIR"/Ti_pv/POTCAR > "Nb-Ti/POTCAR"
cat "$PP_DIR"/Hf_pv/POTCAR "$PP_DIR"/Nb_pv/POTCAR > "Hf-Nb/POTCAR"
cat "$PP_DIR"/Ta_pv/POTCAR "$PP_DIR"/Zr_sv/POTCAR > "Ta-Zr/POTCAR"
cat "$PP_DIR"/Hf_pv/POTCAR > "Hf/POTCAR"
cat "$PP_DIR"/Nb_pv/POTCAR > "Nb/POTCAR"
cat "$PP_DIR"/Ta_pv/POTCAR > "Ta/POTCAR"
cat "$PP_DIR"/Ti_pv/POTCAR > "Ti/POTCAR"
cat "$PP_DIR"/Zr_sv/POTCAR > "Zr/POTCAR"
cat "$PP_DIR"/Al/POTCAR > "Al/POTCAR"
cat "$PP_DIR"/V_sv/POTCAR > "V/POTCAR"
