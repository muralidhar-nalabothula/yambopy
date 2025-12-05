from yambopy.bse.exciton_irreps import compute_exc_rep
import argparse

parser = argparse.ArgumentParser(description="Compute exciton representation from GW-BSE results")

parser.add_argument("--path", type=str, default=".",
    help="Path to the calculation directory (default: .)")

parser.add_argument(
    "-J", type=str, default="SAVE",
    help="BSE Job directory  (default: SAVE)"
)
parser.add_argument(
    "--iqpt", type=int, default=1,
    help="Q-point index (default: 1)"
)
parser.add_argument(
    "--nstates", type=int, default=1,
    help="Number of exciton states (default: 1)"
)
parser.add_argument(
    "--degen_tol", type=float, default=1e-2,
    help="Tolerance for degeneracy (default: 1e-2)"
)

args = parser.parse_args()

compute_exc_rep(
    path=args.path,
    bse_dir=args.J,
    iqpt=args.iqpt,
    nstates=args.nstates,
    degen_tol=args.degen_tol,
)
