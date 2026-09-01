SELECT me.entry_id, me.formula, ps.band_gap FROM material_entry me JOIN phase_stability ps ON ps.entry_id = me.entry_id WHERE ps.band_gap > 1.0 ORDER BY ps.band_gap DESC, me.entry_id;
