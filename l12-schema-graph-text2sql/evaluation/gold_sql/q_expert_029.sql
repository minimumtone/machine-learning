SELECT AVG(s.lattice_a) AS avg_a, STDDEV(s.lattice_a) AS std_a
FROM structure s
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
LIMIT 10000;
