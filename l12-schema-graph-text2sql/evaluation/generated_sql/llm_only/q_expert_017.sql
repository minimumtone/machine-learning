SELECT DISTINCT e.symbol, e.name, e.atomic_number
FROM composition c
JOIN element e ON e.symbol = c.element
ORDER BY e.atomic_number;
