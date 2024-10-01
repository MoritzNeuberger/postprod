import awkward as ak

def m_sum_energy(para, input, output, pv):

    in_n = {
        "edep": input[0]
    }

    out_n = {
        "edep": output[0]
    }

    pv[out_n["edep"]] = ak.sum(pv[in_n["edep"]],axis=-1)
