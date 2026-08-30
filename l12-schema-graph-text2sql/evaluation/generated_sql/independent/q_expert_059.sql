SELECT m.formula, mp.magnetic_anisotropy_energy
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY mp.magnetic_anisotropy_energy DESC
LIMIT 1;
