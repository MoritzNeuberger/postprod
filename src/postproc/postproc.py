from __future__ import annotations

import argparse
import warnings

from misc import load_inst
from process_manager import process_manager

warnings.filterwarnings("ignore")


def main(infile):
    inst = load_inst(infile)
    pm = process_manager(inst, overwrite=False)
    pm.run_processes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="post_proc", description="A generic post processor for Geant4 simulation"
    )
    parser.add_argument("input_file")
    args = parser.parse_args()
    main(args.input_file)
