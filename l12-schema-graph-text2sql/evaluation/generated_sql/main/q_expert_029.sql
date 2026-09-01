SELECT AVG(s.lattice_a) AS avg_lattice_a,
       STDDEV_SAMP(s.lattice_a) AS stddev_lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
LIMIT 10000;
