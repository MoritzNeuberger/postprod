from __future__ import annotations

import uproot
from data_manager import data_manager
from processor_manager import processor_manager


def run_post_proc(args):
    try:
        infile = args[0]
        outfile = args[1]
        inst = args[2]
        task_id = args[3]

        pm = processor_manager(inst)
        dm = data_manager(inst, infile, outfile, pm, task_id)
        dm.process_data()
        dm.write_output()
    except uproot.exceptions.KeyInFileError:
        pass
