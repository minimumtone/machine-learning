SELECT
  AVG(lattice_a) AS average_lattice_a,
  STDDEV_SAMP(lattice_a) AS stddev_lattice_a
FROM structure
WHERE strukturbericht = 'L12'
  AND lattice_a IS NOT NULL;
