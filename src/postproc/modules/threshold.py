from __future__ import annotations


def m_threshold(para, input, output, pv):
    """
    Threshold module for the postprocessing pipeline.

    Returns a awkward array of boolean values in the same shape as the input array, indicating whether the input values are within the threshold range.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        - thr (list): List containing two float values representing the lower and upper energy thresholds.

    input (list): List of input parameters in the following order:
        - val: Name of the values array.

    output (list): List of output parameters in the following order:
        - val: Array of boolean values indicating whether the input values are within the threshold range.

    pv (dict): Dictionary to store the processed values.

    """

    in_n = {
        "val": input[0],
    }

    out_n = {"val": output[0]}

    pv[out_n["val"]] = (pv[in_n["val"]] > para["thr"][0]) * (
        pv[in_n["val"]] < para["thr"][1]
    ) > 0
