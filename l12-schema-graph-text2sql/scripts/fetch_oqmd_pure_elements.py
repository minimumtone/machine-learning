#!/usr/bin/env python3
"""
Fetch pure element (single-element) DFT data from OQMD API.
For each element, identify the ground-state structure (lowest delta_e / most stable).
Output: db/pure_element_data.json
"""
import json
import time
import urllib.request
import urllib.error
import sys

BASE_URL = "https://oqmd.org/oqmdapi/formationenergy"
FIELDS = "name,entry_id,spacegroup,ntypes,natoms,volume,delta_e,stability,band_gap"
LIMIT = 100

def fetch_page(offset: int, retries: int = 3) -> dict:
    url = f"{BASE_URL}?fields={FIELDS}&filter=ntypes=1&limit={LIMIT}&offset={offset}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NIMS-Research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  Attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {retries} retries: {url}")

def main():
    all_entries = []
    offset = 0
    total = None

    while True:
        print(f"Fetching offset={offset}...", file=sys.stderr)
        data = fetch_page(offset)
        total = data["meta"]["data_available"]
        entries = data["data"]
        all_entries.extend(entries)
        print(f"  Got {len(entries)} entries (total so far: {len(all_entries)}/{total})", file=sys.stderr)

        if not data["links"].get("next") or len(all_entries) >= total:
            break
        offset += LIMIT
        time.sleep(0.5)  # rate limiting

    print(f"\nTotal entries fetched: {len(all_entries)}", file=sys.stderr)

    # Group by element name, find ground state (lowest delta_e)
    by_element = {}
    for entry in all_entries:
        elem = entry["name"]
        if elem not in by_element:
            by_element[elem] = []
        by_element[elem].append(entry)

    print(f"Unique elements: {len(by_element)}", file=sys.stderr)

    # For each element, select ground state (lowest delta_e)
    ground_states = {}
    for elem, entries in sorted(by_element.items()):
        # Sort by delta_e (formation energy per atom), take lowest
        entries_sorted = sorted(entries, key=lambda x: x.get("delta_e") or 999)
        gs = entries_sorted[0]
        ground_states[elem] = {
            "element": elem,
            "oqmd_entry_id": gs["entry_id"],
            "spacegroup": gs.get("spacegroup"),
            "natoms": gs.get("natoms"),
            "volume_per_atom": (gs.get("volume") or 0) / (gs.get("natoms") or 1),
            "delta_e_per_atom": gs.get("delta_e"),
            "stability": gs.get("stability"),
            "band_gap": gs.get("band_gap"),
            "n_polymorphs": len(entries),
        }

    result = {
        "_meta": {
            "description": "Pure element ground-state DFT data from OQMD",
            "source": "https://oqmd.org/oqmdapi/formationenergy",
            "filter": "ntypes=1",
            "total_entries_fetched": len(all_entries),
            "unique_elements": len(ground_states),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "ground_states": ground_states,
    }

    outpath = "db/pure_element_data.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {outpath} ({len(ground_states)} elements)", file=sys.stderr)

if __name__ == "__main__":
    main()
