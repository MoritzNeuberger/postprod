from __future__ import annotations

import awkward as ak
import numpy as np


def generate_output(wt_m1, wt_m2, val, para):
    t_min = para["t_min"]
    t_max = para["t_max"]

    # @njit
    def _internal(wt_m1, wt_m2, val, t_min, t_max):
        output = []
        for i in range(len(wt_m1)):  # event
            tmp = []
            if isinstance(wt_m1[i], ak.Array):
                for j in range(len(wt_m1[i]) - 2):  # window 1
                    tmp2 = []
                    for k in range(len(wt_m2[i]) - 2):  # window 2
                        if (wt_m2[i][k] > wt_m1[i][j] + t_min) & (
                            wt_m2[i][k] < wt_m1[i][j] + t_max
                        ):
                            tmp2.append(val[i][k])
                    tmp.append(np.sum(tmp2))
                output.append(tmp)
            else:
                if i >= len(wt_m1) - 1:
                    break
                for k in range(len(wt_m2) - 2):  # window 2
                    if (wt_m2[k] > wt_m1[i] + t_min) & (wt_m2[k] < wt_m1[i] + t_max):
                        tmp.append(val[k])
                output.append(np.sum(tmp))
        return output

    return ak.Array(_internal(wt_m1, wt_m2, val, t_min, t_max))


def m_coincidence_window(para, input, output, pv):
    """
    Coincidence Window module for the postprocessing pipeline.

    Given two lists of windowed data, the module calculates the sum of the values in the second window that fall within a time window of the first window.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        - coincidence_time (float): Time window for coincidence.

    input (list): List of input parameters in the following order:
        - t: Name of times array.
        - val: Name of value array.

    output (list): List of output parameters in the following order:
        - coincident_events: Array of coincident events.

    pv (dict): Dictionary to store the processed values.

    """
    in_n = {
        "w_t_mod1": input[0],
        "w_t_mod2": input[1],
        "val": input[2],
    }

    out_n = {"val": output[0]}

    pv[out_n["val"]] = generate_output(
        pv[in_n["w_t_mod1"]], pv[in_n["w_t_mod2"]], pv[in_n["val"]], para
    )
