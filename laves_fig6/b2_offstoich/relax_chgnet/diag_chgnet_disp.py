import numpy as np, torch
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from chgnet.model.dynamics import CHGNetCalculator
torch.set_num_threads(8)
calc = CHGNetCalculator(use_device='cpu')
at = bulk('NiAl', 'cesiumchloride', a=2.88)
at.calc = calc
LBFGS(FrechetCellFilter(at), logfile=None).run(fmax=1e-4, steps=300)
for d, pm in ((0.02, True), (0.03, True), (0.05, False), (0.05, True), (0.08, True)):
    ph = Phonopy(PhonopyAtoms(symbols=at.get_chemical_symbols(), positions=at.positions, cell=at.cell),
                 supercell_matrix=np.eye(3) * 4)
    ph.generate_displacements(distance=d, is_plusminus=pm)
    F = []
    for sc in ph.supercells_with_displacements:
        s = Atoms(sc.symbols, positions=sc.positions, cell=sc.cell, pbc=True); s.calc = calc
        F.append(s.get_forces())
    ph.forces = np.array(F)
    ph.produce_force_constants()
    ph.run_mesh([16, 16, 16])
    f = ph.mesh.frequencies
    ph.run_thermal_properties(temperatures=[1273, 1473])
    fe = ph.thermal_properties.free_energy * 0.010364272 / 2
    print(f'chgnet d={d} pm={pm}: min f={f.min():.3f} max f={f.max():.3f} '
          f'Fvib1273={fe[0]:.4f} Fvib1473={fe[1]:.4f}', flush=True)
