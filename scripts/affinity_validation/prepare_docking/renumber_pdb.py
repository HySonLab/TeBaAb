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
    """Class to renumber a  to make it HADDOCK-ready"""

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

        return new_pdb


def main(pdb_file: str, out_file: str, chain_id: str, active_sites_file: str = None) -> list:
    # Renumber pdb file and get HV residues
    pdb_format = AbHaddockFormat(pdb_file, chain_id)
    pdb_ren = pdb_format.ab_format()

    # Write pdb into a file
    pdb_ren.to_pdb(path=out_file, records=['ATOM'], append_newline=True)

if __name__ == '__main__':

    # Get inputs
    pdb_file, out_file, chain_id = check_input()
    main(pdb_file, out_file, chain_id)