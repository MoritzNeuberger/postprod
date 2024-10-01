import json
def load_inst(file):
    with open(file,"r") as f:
        inst = json.load(f)
    return inst
