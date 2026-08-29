SELECT *
FROM material_entry
WHERE formula IN ('Ni3Al', 'AlNi3')
   OR reduced_formula IN ('Ni3Al', 'AlNi3')
   OR entry_id IN (
       SELECT entry_id
       FROM composition
       GROUP BY entry_id
       HAVING COUNT(DISTINCT element) = 2
          AND COUNT(*) FILTER (WHERE element NOT IN ('Ni', 'Al')) = 0
          AND ABS(SUM(atomic_fraction) FILTER (WHERE element = 'Ni') - 0.75) < 1e-6
          AND ABS(SUM(atomic_fraction) FILTER (WHERE element = 'Al') - 0.25) < 1e-6
   );
