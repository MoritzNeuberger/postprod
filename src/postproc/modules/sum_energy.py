from __future__ import annotations

import awkward as ak


def m_sum_energy(para, input, output, pv):  # noqa: ARG001
    """
    Sum Energy module for the postprocessing pipeline.

    Calculates the sum of the lowest dimension of the input array.
    Reduces one dimension of the array.

    Parameters:
    para (dict): Dictionary containing parameters for the module.

    input (list): List of input parameters in the following order:
        - edep: Name of energy depositions array.

    output (list): List of output parameters in the following order:
        - total_edep: Total energy deposition.

    pv (dict): Dictionary to store the processed values.

    """
    # Module implementation
    in_n = {"edep": input[0]}

    out_n = {"edep": output[0]}

    pv[out_n["edep"]] = ak.sum(pv[in_n["edep"]], axis=-1)
