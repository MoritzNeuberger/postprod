from modules import *
from tqdm import tqdm

class processor:

    def __init__(self,inst):
        self.name = inst["name"]
        self.input = inst["input"]
        self.output = inst["output"]
        self.module_name = inst["module"]
        if "para" in inst:
            self.para = inst["para"]
        else:
            self.para = {}
        self.module = self._get_module(self.module_name)

    def _get_module(self,module):
        if module == "group_sensitive_volume":
            return m_group_sensitive_volume
        if module == "window":
            return m_window  
        if module == "active_volume":
            return m_active_volume
        if module == "sum_energy":
            return m_sum_energy
        if module == "r90_estimator":
            return m_r90_estimator

    def run(self,processing_variables):
        self.module(self.para, self.input, self.output, processing_variables)

class processor_manager:

    def __init__(self,inst):
        self.processor_list = list()
        for p_inst in inst["instr"]:
            p_inst_local = p_inst.copy()
            if "para" in p_inst_local:
                p_inst_local["para"].update(inst["para"])
            else:
                p_inst_local["para"] = inst["para"]
            self.processor_list.append(processor(p_inst_local))

    def run(self,processing_variables,pbar,task_id):
        for proc in self.processor_list: #tqdm(self.processor_list, desc="Processing", unit="proc"):
            #tqdm.write(f"Running: {proc.name}")  # Display the name of the current process
            pbar.set_description(f"{task_id} - {proc.name}")
            proc.run(processing_variables)
