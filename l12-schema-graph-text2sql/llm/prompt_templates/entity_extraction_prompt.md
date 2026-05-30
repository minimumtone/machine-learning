You are a materials science NLP entity extractor.
Given a natural language query about L1₂-type intermetallic compounds, extract structured conditions.

Output a JSON object with these optional fields:
- "prototype": string (e.g. "L12")
- "contains_elements": list of element symbols (e.g. ["Ni", "Al"])
- "stability": "stable" or "metastable"
- "properties": list of property names
- "sort_by": column to sort by
- "sort_order": "asc" or "desc"
- "lattice_reference": {"reference_formula": str, "reference_lattice_a": float}

Query:
{query}

JSON:
