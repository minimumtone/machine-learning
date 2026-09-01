WITH l12_candidates AS (
    SELECT
        me.entry_id,
        me.formula,
        me.reduced_formula,
        s.lattice_a
    FROM material_entry me
    JOIN structure s ON s.entry_id = me.entry_id
    LEFT JOIN prototype_definition pd ON pd.prototype_id = s.prototype
    CROSS JOIN LATERAL (
        VALUES
            (regexp_replace(translate(upper(coalesce(s.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g')),
            (regexp_replace(translate(upper(coalesce(pd.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g')),
            (regexp_replace(translate(upper(coalesce(s.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g')),
            (regexp_replace(translate(upper(coalesce(pd.prototype_id, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g')),
            (regexp_replace(translate(upper(coalesce(pd.prototype_name, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g'))
    ) AS n(norm_val)
    WHERE n.norm_val LIKE '%L12%'

    UNION ALL

    SELECT
        fh.entry_id,
        coalesce(me.formula, fh.formula) AS formula,
        coalesce(me.reduced_formula, fh.reduced_formula) AS reduced_formula,
        fh.lattice_a
    FROM formation_enthalpy fh
    LEFT JOIN material_entry me ON me.entry_id = fh.entry_id
    CROSS JOIN LATERAL (
        VALUES
            (regexp_replace(translate(upper(coalesce(fh.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g')),
            (regexp_replace(translate(upper(coalesce(fh.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g'))
    ) AS n(norm_val)
    WHERE n.norm_val LIKE '%L12%'
),
l12_entries AS (
    SELECT
        entry_id,
        max(formula) AS formula,
        max(reduced_formula) AS reduced_formula,
        avg(lattice_a) AS lattice_a
    FROM l12_candidates
    GROUP BY entry_id
),
stability_source AS (
    SELECT
        entry_id,
        energy_above_hull,
        formation_energy_per_atom AS formation_energy
    FROM phase_stability
    WHERE energy_above_hull IS NOT NULL
      AND formation_energy_per_atom IS NOT NULL

    UNION ALL

    SELECT
        entry_id,
        energy_above_hull,
        formation_enthalpy_ev_per_atom AS formation_energy
    FROM formation_enthalpy
    WHERE energy_above_hull IS NOT NULL
      AND formation_enthalpy_ev_per_atom IS NOT NULL
),
stability AS (
    SELECT DISTINCT ON (entry_id)
        entry_id,
        energy_above_hull,
        formation_energy
    FROM stability_source
    ORDER BY entry_id, energy_above_hull ASC, formation_energy ASC
),
elastic AS (
    SELECT
        c.entry_id,
        avg(et.bulk_modulus_vrh) AS bulk_modulus_vrh
    FROM calculation c
    JOIN elastic_tensor et ON et.calculation_id = c.calculation_id
    WHERE et.bulk_modulus_vrh IS NOT NULL
    GROUP BY c.entry_id
),
scored AS (
    SELECT
        l.entry_id,
        l.formula,
        l.reduced_formula,
        st.energy_above_hull,
        l.lattice_a,
        e.bulk_modulus_vrh,
        st.formation_energy,
        ((1 - least(st.energy_above_hull, 0.05) / 0.05) * 0.30
         + (1 - least(abs(l.lattice_a - 3.57), 0.3) / 0.3) * 0.30
         + (least(e.bulk_modulus_vrh, 300) / 300) * 0.20
         + (-st.formation_energy / 1.0) * 0.20) AS composite_score
    FROM l12_entries l
    JOIN stability st ON st.entry_id = l.entry_id
    JOIN elastic e ON e.entry_id = l.entry_id
    WHERE l.lattice_a IS NOT NULL
)
SELECT
    row_number() OVER (ORDER BY composite_score DESC, entry_id) AS rank,
    entry_id,
    formula,
    reduced_formula,
    energy_above_hull,
    lattice_a,
    bulk_modulus_vrh,
    formation_energy,
    composite_score
FROM scored
ORDER BY composite_score DESC, entry_id
LIMIT 20;
