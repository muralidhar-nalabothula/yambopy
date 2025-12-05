#!/usr/bin/env python3
import argparse
import numpy as np
import os

from yambopy.dbs.excitondb import YamboExcitonDB
from yambopy.dbs.latticedb import YamboLatticeDB
from yambopy.dbs.wfdb import YamboWFDB


def run_plot_exc_wf_real_space(args):
    parser = argparse.ArgumentParser(description="Generate real-space " +
                                     "exciton wavefunction in Gaussian .cube file.")

    parser.add_argument("--path", type=str, default=".",
                        help="Calculation directory (default: Current working directory.)")
    parser.add_argument("-J","--bse_dir", type=str, default="GW_BSE", metavar="DIR",
                        help="BSE results folder (default: GW_BSE)")
    parser.add_argument("--iqpt", type=int, default=1,help="Q-point index (default: 1)")
    parser.add_argument("--iexe", type=int, required=True,help="Exciton index to plot.")
    # --iexe and --iqpt are not python indexing. i,e 1st item starts from 1.
    parser.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 1],
                        help="Supercell dimensions (default: 1 1 1)")
    parser.add_argument("--wfc_cutoff", type=float, default=-1.0,
                        help="Wavefunction cutoff in Ry (default: −1 → full cutoff)")
    parser.add_argument("--degen_tol", type=float, default=0.01,
                        help="Degeneracy threshold in eV (default: 0.01)")
    # Mutually-exclusive: fix hole OR electron position
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hole", nargs=3, type=float,
                     help="Fix hole position in reduced units.")
    grp.add_argument("--electron", nargs=3, type=float,
                     help="Fix electron position in reduced units.")
    args = parser.parse_args(args)
    #
    calc_path = args.path
    BSE_dir = args.bse_dir
    iqpt = args.iqpt
    #
    if args.hole:
        fix_particle = "h"
        fixed_position = args.hole
    else:
        fix_particle = "e"
        fixed_position = args.electron
    #
    nsfile = os.path.join(calc_path, "SAVE", "ns.db1")
    #
    lattice = YamboLatticeDB.from_db_file(nsfile)
    #
    filename = f"ndb.BS_diago_Q{iqpt}"
    excdb = YamboExcitonDB.from_db_file(lattice, filename=filename,
                                        folder=os.path.join(calc_path, BSE_dir), neigs=-1)
    #
    wfdb = YamboWFDB(path=calc_path,latdb=lattice,
                     bands_range=[np.min(excdb.table[:, 1]) - 1,np.max(excdb.table[:, 2])])
    #
    excdb.real_wf_to_cube(iexe=args.iexe-1,wfdb=wfdb,fixed_postion=fixed_position,
                          supercell=args.supercell,degen_tol=args.degen_tol,
                          wfcCutoffRy=args.wfc_cutoff,fix_particle=fix_particle)



