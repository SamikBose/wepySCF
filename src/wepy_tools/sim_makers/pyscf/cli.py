"""Command-line helper for PySCFSimMaker user scripts.

Call parse_args() to be able to run the simulation entry point with the --sub-step and --from-branch arguments.
Then, pass the results into the sim maker such as PySCFSimMaker.run(args.sub_step, args.from_branch).
"""

# Standard Library
import argparse


def parse_args(argv=None):
    """Parse --sub-step / --from-branch for resuming or branching a run.

    Pass argv explicitly (e.g. in tests); defaults to sys.argv when None.
    """
    parser = argparse.ArgumentParser(description="Run (or resume/branch) a REVO/PySCF wepy simulation.")
    parser.add_argument(
        "--sub-step",
        type=int,
        default=None,
        help="Resume from a previous sub-step. Defaults to starting a new simulation.",
    )
    parser.add_argument(
        "--from-branch",
        type=int,
        default=None,
        help="Branch of the previous sub-step to resume from. Defaults to the latest.",
    )

    return parser.parse_args(argv)
