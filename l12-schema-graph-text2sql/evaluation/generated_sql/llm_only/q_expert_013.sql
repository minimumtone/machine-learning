SELECT EXISTS (
    SELECT 1
    FROM structure s
    JOIN composition c ON c.entry_id = s.entry_id
    JOIN element e ON e.symbol = c.element
    WHERE (
        s.strukturbericht ILIKE 'L12'
        OR s.prototype ILIKE '%L12%'
    )
      AND (
        e.category = 'lanthanide'
        OR e.symbol IN ('Sc', 'Y')
    )
) AS has_rare_earth_l12_compounds;
