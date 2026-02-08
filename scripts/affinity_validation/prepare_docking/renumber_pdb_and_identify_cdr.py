# Original source of this code: https://github.com/haddocking/HADDOCK-antibody-antigen/

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020:
#   Francesco Ambrosetti
#

"""
Formats the antibody to fit the HADDOCK requirements with the
specified chain id and returns the list of residues belonging
to the HV loops defined according to the HADDOCK friendly format.

*** The antibody has to be numbered according to the Chothia scheme ***

Usage:
    python haddock-format.py <chothia numbered antibody> <output .pdb file> <chain_id>

Example:
    python 4G6K_ch.pdb 4G6K-HADDOCK.pdb A

Author: {0}
Email: {1}
"""

import argparse
import biopandas.pdb as bp
import copy as cp
import os
import sys
from pathlib import Path

__author__ = "Francesco Ambrosetti"
__email__ = "ambrosetti.francesco@gmail.com"
USAGE = __doc__.format(__author__, __email__)


def check_input():
    """
    Check and collect the script inputs
    Returns:
        args.pdb (str): path to the pdb-file
        args.chain (str): chain id to use for the HADDOCK-formatted structure
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=USAGE,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('pdb', help='Path to the IMGT numbered antibody PDB structure', type=str)
    parser.add_argument('out', help='Path to the output PDB file', type=str)
    parser.add_argument('chain', help='Chain id to use for the HADDOCK-formatted PDB structure', type=str)

    args = parser.parse_args()

    if not os.path.isfile(args.pdb):
        emsg = 'ERROR!! File {0} not found or not readable\n'.format(args.pdb)
        sys.stderr.write(emsg)
        sys.exit(1)

    if not args.pdb.endswith(".pdb"):
        emsg = 'ERROR!! File {0} not recognize as a PDB file\n'.format(args.pdb)
        sys.stderr.write(emsg)
        sys.exit(1)

    return args.pdb, args.out, args.chain


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


class AbHaddockFormat:
    """Class to renumber a IMGT antibody to make it HADDOCK-ready"""

    # L chain loops (IMGT)
    l1 = [f"{i}_L" for i in range(27, 39)]  # 27 - 38
    l2 = [f"{i}_L" for i in range(56, 66)]  # 56 - 65
    l3 = [f"{i}_L" for i in range(105, 118)]  # 105 - 117 (+ possible insertions like 111A_L, 111B_L)
    loops_l = l1 + l2 + l3
    
    # H chain loops (IMGT)
    h1 = [f"{i}_H" for i in range(27, 39)]  # 27 - 38
    h2 = [f"{i}_H" for i in range(56, 66)]  # 56 - 65
    h3 = [f"{i}_H" for i in range(105, 118)]  # 105 - 117 (+ possible insertions like 111A_H, 111B_H)
    loops_h = h1 + h2 + h3

    def __init__(self, pdbfile, chain):
        """
        Constructor for the AbHaddockFormat class
        Args:
            pdbfile (str): path to the antibody .pdb file
            chain (str): chain id to use for the HADDOCK-ready structure
        """
        self.file = pdbfile
        self.pdb = bp.PandasPdb().read_pdb(self.file)
        self.chain = chain

    def check_chain(self):
        """
        Check if the antibody contains the light and heavy chain
        Returns:
            0
        """
        chain_ids = self.pdb.df['ATOM']['chain_id'].values

        if 'H' not in chain_ids:
            emsg = 'ERROR!! File {0} does not contain the heavy chain\n'.format(self.file)
            sys.stderr.write(emsg)
            sys.exit(1)

        elif 'L' not in chain_ids:
            emsg = 'ERROR!! File {0} does not contain the light chain\n'.format(self.file)
            sys.stderr.write(emsg)
            sys.exit(1)

        return 0

    def ab_format(self):
        """
        Renumbers the antibody and extract the HV residues

        Returns:
            hv_list (list): list of the HV residue numbers
            new_pdb (biopandas.pdb.pandas_pdb.PandasPdb): HADDOCK-ready pdb
        """

        # Check antibody chain ids
        self.check_chain()

        # Modify resno to include insertions and chain id
        resno = self.pdb.df['ATOM']['residue_number'].values
        ins = self.pdb.df['ATOM']['insertion'].values
        chain = self.pdb.df['ATOM']['chain_id'].values
        ch_resno = ['{0}{1}_{2}'.format(i, j, c) for i, j, c in zip(resno, ins, chain)]

        # Create new resno
        count = 0
        prev_resid = None
        new_resno = []

        # Renumber
        for r in ch_resno:
            if r != prev_resid:
                count += 1
                new_resno.append(count)
                prev_resid = r
            elif r == prev_resid:
                new_resno.append(count)
                prev_resid = r

        # Update pdb
        new_pdb = cp.deepcopy(self.pdb)
        new_pdb.df['ATOM']['chain_id'] = self.chain
        new_pdb.df['ATOM']['residue_number'] = new_resno
        new_pdb.df['ATOM']['insertion'] = ''  # Remove insertions

        # Create dictionary with old and new numbering
        resno_dict = dict(zip(unique(ch_resno), unique(new_resno)))

        # Collect HV residues with the new numbering
        hv_list = []

        # Heavy chain
        for hv_heavy in self.loops_h:
            if hv_heavy in resno_dict.keys():
                hv_list.append(resno_dict[hv_heavy])
         
        # Light chain
        for hv_light in self.loops_l:
            if hv_light in resno_dict.keys():
                hv_list.append(resno_dict[hv_light])

        hv_list.sort()
        return hv_list, new_pdb


def main(pdb_file: str, out_file: str, chain_id: str, active_sites_file: str = None) -> list:
    # Renumber pdb file and get HV residues
    pdb_format = AbHaddockFormat(pdb_file, chain_id)
    hv_resno, pdb_ren = pdb_format.ab_format()

    # Write pdb into a file
    pdb_ren.to_pdb(path=out_file, records=['ATOM'], append_newline=True)

    # Print HV residues
    active_residues = ','.join(map(str, hv_resno))
    print(active_residues)
    if active_sites_file is not None:
        Path(active_sites_file).write_text(active_residues)
    return hv_resno


if __name__ == '__main__':

    # Get inputs
    pdb_file, out_file, chain_id = check_input()
    # pdb_file = "/kaggle/input/tebaab-result-w-des-wo-des/result/with_description/pdb/antibody_1a2y_1.pdb"
    # out_file = "/kaggle/working/antibody_1a2y_1_formatted.pdb"
    # chain_id = "A"
    main(pdb_file, out_file, chain_id)