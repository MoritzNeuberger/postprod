import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import os
from process import run_post_proc

class process_manager:
    def __init__(self, inst, overwrite=False):
        self.in_folder = inst["io"]["input"]
        self.out_folder = inst["io"]["output"]
        self.overwrite = overwrite
        self.threads = inst["para"]["threads"]

        # Get input files and corresponding output files
        self.input_files = glob(self.in_folder + "*.root")
        self.output_files = [infile.replace(self.in_folder, self.out_folder).replace(".root",".h5py") for infile in self.input_files]

        # Filter out files that already exist if overwrite is False
        if not self.overwrite:
            mask_doesnt_exist = np.array([not os.path.exists(f) for f in self.output_files])
            print("skipping:")
            for line in np.array(self.input_files)[~mask_doesnt_exist].tolist():
                print(line)
            self.input_files = np.array(self.input_files)[mask_doesnt_exist].tolist()
            self.output_files = np.array(self.output_files)[mask_doesnt_exist].tolist()

        # Create a list of arguments: each is a tuple (input_file, output_file, inst)
        self.args = list(zip(self.input_files, self.output_files, [inst] * len(self.input_files),np.arange(len(self.input_files))))

    def run_processes(self):
        if self.threads > 1:
            # Use multiprocessing to run run_post_proc with the arguments
            with ProcessPoolExecutor(max_workers=self.threads) as executor:
                
                # Unpack the arguments using *args
                futures = [executor.submit(run_post_proc, arg) for arg in self.args]
                # iterate over all submitted tasks and get results as they are available
                for future in as_completed(futures):
                    # get the result for the next completed task
                    result = future.result() # blocks
            #results = executor.map(lambda args: run_post_proc(*args), self.args)
        else:
            for i in range(len(self.args)):
                run_post_proc(self.args[i])