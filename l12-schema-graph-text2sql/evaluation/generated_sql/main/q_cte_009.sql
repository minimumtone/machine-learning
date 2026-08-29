WITH stable_l12 AS (
    SELECT DISTINCT
           m.entry_id,
           regexp_split_to_table(m.chemical_system, '-') AS element,
           ps.formation_energy_per_atom
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.is_stable = TRUE
)
SELECT element,
       COUNT(DISTINCT entry_id) AS stable_l12_compound_count,
       AVG(formation_energy_per_atom) AS avg_rebased_formation_energy_per_atom
FROM stable_l12
GROUP BY element
ORDER BY stable_l12_compound_count DESC
LIMIT 5;
