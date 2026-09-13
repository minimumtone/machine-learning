#!/bin/bash
# POTCAR generation for dilute-Al calculations
# Requires: $VASP_PP_PATH (PAW-PBE directory)

if [ -z "$VASP_PP_PATH" ]; then
    echo "Error: VASP_PP_PATH is not set."; exit 1
fi

FAIL=0
cat "$VASP_PP_PATH"/Ni_pv/POTCAR > fcc_Ni_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ni_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ni_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ni_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ni_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ni_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ni_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ni_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ni_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ni_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ni_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ni_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ni_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ni_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ni_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ni_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ni_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Co/POTCAR > fcc_Co_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Co_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Co/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Co_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Co_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Co/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Co_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Co_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Co/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Co_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Co_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Co/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Co_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Co_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Co/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Co_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Co_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pd/POTCAR > fcc_Pd_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pd_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pd/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pd_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pd_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pd/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pd_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pd_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pd/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pd_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pd_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pd/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pd_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pd_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pd/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pd_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pd_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Rh_pv/POTCAR > fcc_Rh_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Rh_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Rh_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Rh_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Rh_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Rh_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Rh_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Rh_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Rh_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Rh_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Rh_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Rh_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Rh_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Rh_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Rh_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Rh_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Rh_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ir/POTCAR > fcc_Ir_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ir_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ir/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ir_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ir_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ir/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ir_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ir_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ir/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ir_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ir_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ir/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ir_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ir_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ir/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ir_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ir_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Cu_pv/POTCAR > fcc_Cu_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Cu_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Cu_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Cu_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Cu_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Cu_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Cu_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Cu_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Cu_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Cu_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Cu_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Cu_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Cu_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Cu_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Cu_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Cu_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Cu_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ag/POTCAR > fcc_Ag_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ag_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ag/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ag_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ag_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ag/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ag_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ag_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ag/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ag_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ag_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ag/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ag_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ag_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Ag/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Ag_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Ag_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Au/POTCAR > fcc_Au_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Au_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Au/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Au_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Au_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Au/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Au_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Au_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Au/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Au_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Au_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Au/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Au_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Au_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Au/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Au_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Au_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pt/POTCAR > fcc_Pt_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pt_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pt/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pt_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pt_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pt/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pt_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pt_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pt/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pt_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pt_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pt/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pt_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pt_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Pt/POTCAR "$VASP_PP_PATH"/Al/POTCAR > fcc_Pt_imp108/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: fcc_Pt_imp108"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Nb_pv/POTCAR > bcc_Nb_n0/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: bcc_Nb_n0"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Nb_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > bcc_Nb_n1/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: bcc_Nb_n1"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Nb_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > bcc_Nb_n2/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: bcc_Nb_n2"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Nb_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > bcc_Nb_n3/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: bcc_Nb_n3"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Nb_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > bcc_Nb_n4/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: bcc_Nb_n4"; FAIL=$((FAIL+1)); fi
cat "$VASP_PP_PATH"/Nb_pv/POTCAR "$VASP_PP_PATH"/Al/POTCAR > bcc_Nb_imp128/POTCAR 2>/dev/null
if [ $? -ne 0 ]; then echo "  FAIL: bcc_Nb_imp128"; FAIL=$((FAIL+1)); fi

echo "Done. Failed: $FAIL / 60"
