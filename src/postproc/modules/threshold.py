from __future__ import annotations


def m_threshold(para, input, output, pv):
    in_n = {
        "val": input[0],
    }

    out_n = {"val": output[0]}

    pv[out_n["val"]] = (pv[in_n["val"]] > para["thr"][0]) & (
        pv[in_n["val"]] < para["thr"][1]
    )
