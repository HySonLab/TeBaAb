#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def read_list_file(filepath):
    """Read a .list file containing comma-separated numbers."""
    with open(filepath, "r") as f:
        content = f.read().strip()
        if not content:
            return []
        return [int(x.strip()) for x in content.split(",") if x.strip().isdigit()]

def main():
    if len(sys.argv) < 3:
        print("Usage: python create_ambig_config.py <sasa_ab.list> <sasa_ag.list>", file=sys.stderr)
        sys.exit(1)

    sasa_ab_path = Path(sys.argv[1])
    sasa_ag_path = Path(sys.argv[2])

    if not sasa_ab_path.exists() or not sasa_ag_path.exists():
        print("Error: One or both input files do not exist.", file=sys.stderr)
        sys.exit(1)

    # Read residue indices
    sasa_ab = read_list_file(sasa_ab_path)
    sasa_ag = read_list_file(sasa_ag_path)

    # Build configuration
    ambig_config = [
        {
            "id": 1,
            "chain": "A",
            "active": sasa_ab,
            "passive": [],
            "target": [2],
            "passive_from_active": True
        },
        {
            "id": 2,
            "chain": "B",
            "active": [],
            "passive": sasa_ag,
            "target": [1]
        }
    ]

    # Print JSON to stdout (so user can redirect output)
    print(json.dumps(ambig_config, indent=2))

if __name__ == "__main__":
    main()
