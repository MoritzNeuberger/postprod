from __future__ import annotations

import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
from process import run_post_proc


class process_manager:
    def __init__(self, inst, overwrite=False):
        self.in_folder = inst["io"]["input"]["folder"]
        self.in_format = inst["io"]["input"]["format"]
        self.out = inst["io"]["output"]
        self.overwrite = overwrite
        self.threads = inst["para"]["threads"]
        self.mode = inst["para"].get("mode", "")

        # Get input files and corresponding output files
        self.input_files = Path.glob(self.in_folder + "*." + self.in_format)
        if self.mode != "summarize":
            self.output_files = [
                infile.replace(self.in_folder, self.out).replace(
                    self.in_format, ".hdf5"
                )
                for infile in self.input_files
            ]
        else:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.output_files = [
                infile.replace(self.in_folder, self.tmp_dir.name).replace(
                    self.in_format, ".hdf5"
                )
                for infile in self.input_files
            ]
            # self.output_files = self.out

        # Filter out files that already exist if overwrite is False
        if not self.overwrite:
            mask_doesnt_exist = np.array(
                [not Path.exists(f) for f in self.output_files]
            )
            self.input_files = np.array(self.input_files)[mask_doesnt_exist].tolist()
            self.output_files = np.array(self.output_files)[mask_doesnt_exist].tolist()

        # Create a list of arguments: each is a tuple (input_file, output_file, inst)
        self.args = list(
            zip(
                self.input_files,
                self.output_files,
                [inst] * len(self.input_files),
                np.arange(len(self.input_files)),
            )
        )

    def summarize(self):
        def gen_files():
            for file in self.output_files:
                with h5py.File(file, "r") as f:
                    group = f["awkward"]
                    reconstituted = ak.from_buffers(
                        ak.forms.from_json(group.attrs["form"]),
                        group.attrs["length"],
                        {k: np.asarray(v) for k, v in group.items()},
                    )
                    yield ak.Array(reconstituted)

        with h5py.File(self.out, "w") as f:
            group = f.create_group("awkward")
            form, length, container = ak.to_buffers(
                ak.to_packed(ak.from_iter(gen_files())), container=group
            )
            group.attrs["form"] = form.to_json()
            group.attrs["length"] = length

    def run_processes(self):
        if self.threads > 1:
            # Use multiprocessing to run run_post_proc with the arguments
            with ProcessPoolExecutor(max_workers=self.threads) as executor:
                # Unpack the arguments using *args
                futures = [executor.submit(run_post_proc, arg) for arg in self.args]
                # iterate over all submitted tasks and get results as they are available
                for future in as_completed(futures):
                    # get the result for the next completed task
                    future.result()  # blocks
            # results = executor.map(lambda args: run_post_proc(*args), self.args)
        else:
            for i in range(len(self.args)):
                run_post_proc(self.args[i])

        if self.mode == "summarize":
            self.summarize()
