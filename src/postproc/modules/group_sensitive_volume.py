from __future__ import annotations

import awkward as ak
import numpy as np
from numba import njit


def generate_mask(vol, group, sensitive_volumes):
    sv_in_group = np.array(sensitive_volumes["sensVolID"])[
        np.array(sensitive_volumes["group"]) == group
    ]
    mask = ak.zeros_like(vol)

    if isinstance(sv_in_group, int):
        mask = mask + (vol == sv_in_group)
    else:
        for sv in sv_in_group:
            mask = mask + (vol == sv)
    return mask > 0


@njit
def get_detector_ids_in_window(v_voln_hw):
    list_indices = []
    for i in range(len(v_voln_hw)):
        if v_voln_hw[i] not in list_indices and v_voln_hw[i] > -1:
            list_indices.append(v_voln_hw[i])
    return list_indices


@njit
def group_window_into_detector_ids(v_voln_hw, v_in):
    list_indices = get_detector_ids_in_window(v_voln_hw)
    output = []
    for i in range(len(list_indices)):
        tmp = []
        for j in range(len(v_in)):
            if list_indices[i] == v_voln_hw[j]:
                tmp.append(v_in[j])
        output.append(tmp)
    return output


def group_all_in_detector_ids(v_voln_hw, v_in):
    @njit
    def _internal(v_voln_hw, v_in):
        output = []
        for i in range(len(v_voln_hw)):
            tmp = []
            for j in range(len(v_voln_hw[i])):
                tmp.append(group_window_into_detector_ids(v_voln_hw[i][j], v_in[i][j]))
            output.append(tmp)
        return output

    return ak.Array(_internal(v_voln_hw, v_in))


def m_group_sensitive_volume(para, input, output, pv):
    in_n = {
        "t": input[0],
        "edep": input[1],
        "vol": input[2],
        "posx": input[3],
        "posy": input[4],
        "posz": input[5],
    }

    out_n = {
        "t": output[0],
        "edep": output[1],
        "vol": output[2],
        "posx": output[3],
        "posy": output[4],
        "posz": output[5],
    }

    if "group" in para:
        mask = generate_mask(pv[in_n["vol"]], para["group"], para["sensitive_volumes"])
        for key, value in out_n.items():
            pv[value] = pv[in_n[key]][mask]

    else:
        for key, value in out_n.items():
            pv[value] = group_all_in_detector_ids(pv[in_n["vol"]], pv[in_n[key]])
