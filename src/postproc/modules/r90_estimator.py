from __future__ import annotations

import awkward as ak
import numpy as np


def get_R90_per_detector(v_dist, v_edep):
    tot_e = np.sum(v_edep)
    mask_sort = v_dist.argsort()
    v_dist_sort = v_dist[mask_sort]
    v_edep_sort = v_edep[mask_sort]
    v_cumsum_edep_sort = np.cumsum(v_edep_sort)
    if len(v_cumsum_edep_sort):
        pos = np.argmax(v_cumsum_edep_sort >= 0.9 * tot_e)
        return v_dist_sort[pos]
    return 0


def calculate_R90(v_edep_hwd, v_posx_hwd, v_posy_hwd, v_posz_hwd):
    v_posx_edep = v_edep_hwd * v_posx_hwd
    v_posy_edep = v_edep_hwd * v_posy_hwd
    v_posz_edep = v_edep_hwd * v_posz_hwd
    v_edep_e = ak.sum(v_edep_hwd, axis=3)

    v_meanx = ak.sum(v_posx_edep, axis=3) / v_edep_e
    v_meany = ak.sum(v_posy_edep, axis=3) / v_edep_e
    v_meanz = ak.sum(v_posz_edep, axis=3) / v_edep_e

    v_dist = np.sqrt(
        (v_posx_hwd - v_meanx) ** 2
        + (v_posy_hwd - v_meany) ** 2
        + (v_posz_hwd - v_meanz) ** 2
    )

    return ak.Array(
        [
            [
                [
                    get_R90_per_detector(
                        v_dist[i, j, k].to_numpy(), v_edep_hwd[i, j, k].to_numpy()
                    )
                    for k in range(len(v_dist[i][j]))
                ]
                for j in range(len(v_dist[i]))
            ]
            for i in range(len(v_dist))
        ]
    )


def m_r90_estimator(para, input, output, pv):  # noqa: ARG001
    in_n = {"edep": input[0], "x": input[1], "y": input[2], "z": input[3]}

    out_n = {"r90": output[0]}
    pv[out_n["r90"]] = calculate_R90(
        pv[in_n["edep"]], pv[in_n["x"]], pv[in_n["y"]], pv[in_n["z"]]
    )
