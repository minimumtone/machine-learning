# MACE BCC HEA Voronoi/Bader study

This directory measures, element by element, how much of a binary excess
volume survives in a multicomponent BCC HEA. MACE-MP-0 small relaxes 128-atom
randomly occupied BCC cells. Voronoi volumes provide the immediate
per-element measurement. VASP one-shot inputs are also generated so that
Bader volumes and charges can be obtained later by running VASP and the
Henkelman Bader code.

## Workflow

```bash
python3 mlip_bader/build_and_relax.py
python3 mlip_bader/voronoi_analysis.py
python3 mlip_bader/generate_bader_inputs.py
```

`build_and_relax.py` performs 58 calculations: seven pure references, 15
binary systems with three random seeds each, and two HEAs with three seeds
each. Pure-element MACE-relaxed BCC lattice parameters are used to initialize
the mixed cells. The random occupation is a NumPy `default_rng(seed)` shuffle,
not an icet SQS.

The model is MACE-MP-0 small with float64 arithmetic on CPU. MACE has no spin
degree of freedom; no spin variables are used in the relaxation. Relaxed
structures and results are written under `structures/` and
`relax_results.csv`. Voronoi results are written to
`voronoi_per_atom.csv`, `voronoi_summary.md`, and
`fig_delta_v_per_element.png`.
The relaxation summary includes `a_bcc_A=(2V/N)^(1/3)`, the final
atomic-force maximum `fmax_final` in eV/A, the convergence flag, and the
number of FIRE steps.

## Bader inputs

`generate_bader_inputs.py` writes seed-0 inputs for the two HEAs, five
selected binaries, and seven pure elements under `vasp_bader/`. The input
settings follow `vasp_inputs/generate_sqs_recalc_2x2x2.py`: ENCUT 520 eV,
PREC=Accurate, GGA=PE, ISMEAR=1, SIGMA=0.1, ALGO=Normal, ISPIN=2,
EDIFF=1E-6, and NCORE=4. The calculation is a single point
(`IBRION=-1`, `NSW=0`) with `LCHARG=.TRUE.`, `LAECHG=.TRUE.`, and
`LWAVE=.FALSE.`. The PBE POTCAR variants are reproduced in
`vasp_bader/make_potcar.sh`.

The approximately 13.3 Angstrom cells use explicit `NGXF=NGYF=NGZF=320`.
These values are intended as twice the default grid estimate for this cell
size and provide a fixed dense grid for charge-density post-processing.
`KPOINTS` is Gamma-centered 2x2x2.

After a VASP run, execute:

```bash
./mlip_bader/run_bader.sh mlip_bader/vasp_bader/HfNbTaTiZr
```

This runs `chgsum.pl AECCAR0 AECCAR2`, `bader CHGCAR -ref CHGCAR_sum`, and
`parse_bader.py`. The parser writes `bader_per_atom.csv` with the common
`label, seed, atom_index, element` fields plus Bader volume and charge.

## Caveats

* MACE-MP-0 is not spin-polarized.
* Random occupations are not special quasirandom structures.
* Every cell contains 128 atoms and uses three seeds for mixed systems.
* Voronoi volume is not Bader volume. Voronoi results are geometric
  partitioning; Bader results require a VASP charge density and the external
  Henkelman code.
* The per-element survival ratio is flagged as unreliable when the predicted
  excess-volume magnitude is below 0.05 A^3.
