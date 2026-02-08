#!/usr/bin/env python3
import freesasa
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Calculate solvent accessible surface area (SASA) per residue using FreeSASA.")
    parser.add_argument("--input", "-i", required=True, help="Input PDB file path.")
    parser.add_argument("--threshold", "-t", type=float, default=0.4, help="Relative SASA threshold (default: 0.4)")
    parser.add_argument("--output_mode", "-o", choices=["chain_resnum", "resnum_only", "chain_only"], default="resnum_only",
                        help="Output format: 'chain_resnum', 'resnum_only' (default), or 'chain_only'")

    args = parser.parse_args()

    # Load structure
    try:
        structure = freesasa.Structure(args.input)
    except Exception as e:
        print(f"Error loading PDB: {e}", file=sys.stderr)
        sys.exit(1)

    # Run SASA
    result = freesasa.calc(structure)
    residues = result.residueAreas()

    accessible_residues = []

    # Collect accessible residues
    for chain in residues:
        for resnum, resarea in residues[chain].items():
            rel = resarea.relativeTotal
            if rel >= args.threshold:
                accessible_residues.append((chain, resnum))

    # Format results according to user selection
    if args.output_mode == "chain_resnum":
        formatted = [f"{c}:{r}" for c, r in accessible_residues]
    elif args.output_mode == "resnum_only":
        formatted = [str(r) for _, r in accessible_residues]
    elif args.output_mode == "chain_only":
        formatted = [c for c, _ in accessible_residues]
    else:
        formatted = []

    # Join as comma-separated string
    active_residues = ",".join(formatted)
    print(active_residues)

    return accessible_residues


if __name__ == "__main__":
    main()