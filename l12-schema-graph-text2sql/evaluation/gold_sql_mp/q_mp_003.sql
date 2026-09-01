SELECT DISTINCT e.formula FROM mp_entries e JOIN mp_element_ratios r ON e.entry_id = r.entry_id WHERE r.element = 'Ni' ORDER BY e.formula LIMIT 5;
