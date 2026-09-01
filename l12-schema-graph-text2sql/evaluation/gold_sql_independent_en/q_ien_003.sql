SELECT me.entry_id, me.formula FROM material_entry me JOIN magnetic_property mp ON mp.entry_id = me.entry_id WHERE mp.magnetic_ordering = 'ferromagnetic' ORDER BY me.entry_id;
