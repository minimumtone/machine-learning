WITH target AS (
    SELECT DISTINCT ON (f.entry_id, f.reference_set, cls.sb_class)
        f.entry_id,
        COALESCE(f.formula, me.formula) AS formula,
        me.chemical_system,
        f.reference_set,
        COALESCE(f.prototype, s.prototype) AS prototype,
        cls.sb_class AS strukturbericht,
        COALESCE(f.lattice_a, s.lattice_a) AS lattice_a,
        COALESCE(
            pd.conventional_cell_atoms,
            CASE cls.sb_class
                WHEN 'B2' THEN 2
                WHEN 'L12' THEN 4
            END
        ) AS conventional_cell_atoms,
        f.formation_enthalpy_ev_per_atom,
        f.energy_above_hull,
        f.is_stable
    FROM formation_enthalpy f
    JOIN material_entry me
        ON me.entry_id = f.entry_id
    LEFT JOIN structure s
        ON s.entry_id = f.entry_id
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = COALESCE(s.prototype, f.prototype)
    CROSS JOIN LATERAL (
        SELECT
            regexp_replace(upper(COALESCE(f.strukturbericht, '')), '[^A-Z0-9]', '', 'g') AS f_sb,
            regexp_replace(upper(COALESCE(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') AS s_sb,
            regexp_replace(upper(COALESCE(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') AS pd_sb,
            regexp_replace(upper(COALESCE(f.prototype, '')), '[^A-Z0-9]', '', 'g') AS f_proto,
            regexp_replace(upper(COALESCE(s.prototype, '')), '[^A-Z0-9]', '', 'g') AS s_proto
    ) n
    CROSS JOIN LATERAL (
        SELECT CASE
            WHEN n.f_sb IN ('B2', 'L12') THEN n.f_sb
            WHEN n.s_sb IN ('B2', 'L12') THEN n.s_sb
            WHEN n.pd_sb IN ('B2', 'L12') THEN n.pd_sb
            WHEN n.f_proto IN ('B2', 'L12') THEN n.f_proto
            WHEN n.s_proto IN ('B2', 'L12') THEN n.s_proto
        END AS sb_class
    ) cls
    WHERE cls.sb_class IN ('B2', 'L12')
      AND COALESCE(f.lattice_a, s.lattice_a) IS NOT NULL
    ORDER BY f.entry_id, f.reference_set, cls.sb_class, s.structure_id NULLS LAST
),
vegard AS (
    SELECT
        t.entry_id,
        t.formula,
        t.chemical_system,
        t.reference_set,
        t.prototype,
        t.strukturbericht,
        t.lattice_a,
        t.conventional_cell_atoms,
        power(t.lattice_a, 3) / t.conventional_cell_atoms AS actual_volume_per_atom,
        SUM(c.atomic_fraction::double precision * per.volume_per_atom::double precision) AS vegard_volume_per_atom,
        string_agg(c.element || ':' || to_char(c.atomic_fraction, 'FM0.9999'), ', ' ORDER BY c.element) AS composition,
        t.formation_enthalpy_ev_per_atom,
        t.energy_above_hull,
        t.is_stable
    FROM target t
    JOIN composition c
        ON c.entry_id = t.entry_id
    LEFT JOIN pure_element_reference per
        ON per.element_symbol = c.element
       AND per.reference_set = t.reference_set
    WHERE t.conventional_cell_atoms IS NOT NULL
    GROUP BY
        t.entry_id, t.formula, t.chemical_system, t.reference_set, t.prototype,
        t.strukturbericht, t.lattice_a, t.conventional_cell_atoms,
        t.formation_enthalpy_ev_per_atom, t.energy_above_hull, t.is_stable
    HAVING COUNT(*) = COUNT(per.volume_per_atom)
       AND SUM(c.atomic_fraction::double precision * per.volume_per_atom::double precision) > 0
),
scored AS (
    SELECT
        v.*,
        power(v.conventional_cell_atoms * v.vegard_volume_per_atom, 1.0 / 3.0) AS vegard_lattice_a,
        100.0 * (
            v.lattice_a - power(v.conventional_cell_atoms * v.vegard_volume_per_atom, 1.0 / 3.0)
        ) / power(v.conventional_cell_atoms * v.vegard_volume_per_atom, 1.0 / 3.0) AS vegard_lattice_deviation_pct,
        100.0 * (
            v.actual_volume_per_atom - v.vegard_volume_per_atom
        ) / v.vegard_volume_per_atom AS vegard_volume_deviation_pct
    FROM vegard v
)
SELECT
    entry_id,
    formula,
    chemical_system,
    composition,
    strukturbericht,
    prototype,
    reference_set,
    conventional_cell_atoms,
    ROUND(lattice_a::numeric, 6) AS actual_lattice_a,
    ROUND(vegard_lattice_a::numeric, 6) AS vegard_lattice_a,
    ROUND(vegard_lattice_deviation_pct::numeric, 3) AS vegard_lattice_deviation_pct,
    ROUND(ABS(vegard_lattice_deviation_pct)::numeric, 3) AS abs_vegard_lattice_deviation_pct,
    ROUND(actual_volume_per_atom::numeric, 6) AS actual_volume_per_atom,
    ROUND(vegard_volume_per_atom::numeric, 6) AS vegard_volume_per_atom,
    ROUND(vegard_volume_deviation_pct::numeric, 3) AS vegard_volume_deviation_pct,
    ROUND(formation_enthalpy_ev_per_atom::numeric, 6) AS formation_enthalpy_ev_per_atom,
    ROUND(energy_above_hull::numeric, 6) AS energy_above_hull,
    is_stable
FROM scored
ORDER BY ABS(vegard_lattice_deviation_pct) DESC
LIMIT 50;
