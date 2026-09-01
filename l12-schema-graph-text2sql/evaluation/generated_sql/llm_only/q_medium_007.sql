WITH l12_compounds AS (
  SELECT
    formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom
  FROM formation_enthalpy
  WHERE formation_enthalpy_ev_per_atom IS NOT NULL
    AND regexp_replace(
          upper(translate(coalesce(strukturbericht, '') || ' ' || coalesce(prototype, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')),
          '[^A-Z0-9]',
          '',
          'g'
        ) LIKE '%L12%'
)
SELECT
  round((floor(formation_energy_ev_per_atom / 0.10) * 0.10)::numeric, 2) AS formation_energy_bin_lower_ev_per_atom,
  round(((floor(formation_energy_ev_per_atom / 0.10) + 1) * 0.10)::numeric, 2) AS formation_energy_bin_upper_ev_per_atom,
  count(*) AS compound_count,
  round(avg(formation_energy_ev_per_atom)::numeric, 4) AS avg_formation_energy_ev_per_atom,
  round(min(formation_energy_ev_per_atom)::numeric, 4) AS min_formation_energy_ev_per_atom,
  round(max(formation_energy_ev_per_atom)::numeric, 4) AS max_formation_energy_ev_per_atom
FROM l12_compounds
GROUP BY floor(formation_energy_ev_per_atom / 0.10)
ORDER BY formation_energy_bin_lower_ev_per_atom;
