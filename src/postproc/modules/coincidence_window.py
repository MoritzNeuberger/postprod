from __future__ import annotations

import awkward as ak
from numba import njit


def generate_output(wt_m1, wt_m2, val, para):
    t_min = para["t_min"]
    t_max = para["t_max"]

    @njit
    def _internal(wt_m1, wt_m2, val, t_min, t_max):
        output = []
        for i in range(len(wt_m1)):  # event
            tmp = []
            for j in range(len(wt_m1[i])):  # window 1
                tmp2 = []
                for k in range(len(wt_m2[i])):  # window 2
                    if (wt_m2[i][k] > wt_m1[i][j] + t_min) & (
                        wt_m2[i][k] < wt_m1[i][j] + t_max
                    ):
                        tmp2.append(val[i][k])
                tmp.append(ak.sum(tmp2))
            output.append(tmp)
        return output

    return ak.Array(_internal(wt_m1, wt_m2, val, t_min, t_max))


def m_coincidence_window(para, input, output, pv):
    in_n = {
        "w_t_mod1": input[0],
        "w_t_mod2": input[1],
        "val": input[2],
    }

    out_n = {"val": output[0]}

    pv[out_n["val"]] = generate_output(
        pv[in_n["w_t_mod1"]], pv[in_n["w_t_mod2"]], pv[in_n["val"]], para
    )
