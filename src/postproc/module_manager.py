from __future__ import annotations

from module import module


class module_manager:
    def __init__(self, inst):
        self.module_list = []
        for p_inst in inst["instr"]:
            p_inst_local = p_inst.copy()
            if "para" in p_inst_local:
                p_inst_local["para"].update(inst["para"])
            else:
                p_inst_local["para"] = inst["para"]
            self.module_list.append(module(p_inst_local))

    def run(self, processing_variables, pbar, task_id):
        for proc in (
            self.module_list
        ):  # tqdm(self.module_list, desc="Processing", unit="proc"):
            # tqdm.write(f"Running: {proc.name}")  # Display the name of the current process
            pbar.set_description(f"{task_id} - {proc.name}")
            proc.run(processing_variables)
