from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np
from numba import njit, types
from numba.typed import Dict


def generate_mask_cylinder(x, y, z, para):
    r = np.sqrt(x**2 + y**2)
    mask = ak.ones_like(x)
    mask = mask - (r > para["conditions"]["r"])
    mask = mask - ((z > para["conditions"]["h1"]) + (z < para["conditions"]["h2"]))

    inverse = False
    if "inverse" in para:
        inverse = para["inverse"]

    if inverse:
        return mask < 1
    return mask > 0


@njit
def is_point_inside_polycone(x, y, z, r_values, z_values):
    # Step 1: Convert (x, y, z) to cylindrical coordinates (r, z)
    r_point = np.sqrt(x**2 + y**2)

    # Step 2: Check if the z-coordinate is within the z-range of the polycone
    if z < min(z_values) or z > max(z_values):
        return False  # Point is outside based on z

    # Step 3: Determine which segment of the polycone the z value falls into
    for i in range(len(z_values) - 1):
        z_low, z_high = z_values[i], z_values[i + 1]
        r_low, r_high = r_values[i], r_values[i + 1]

        if z_low <= z <= z_high:
            # Step 4: Interpolate the radius at the point's z value
            t = (z - z_low) / (
                z_high - z_low
            )  # Normalized z position between z_low and z_high
            r_interp = r_low + t * (r_high - r_low)  # Linear interpolation of r

            # Step 5: Check if the point's radial distance is less than the interpolated radius
            return r_point <= r_interp

    return False  # Point is outside


@njit
def is_in_active_volume(x, y, z, vol, dl_input):
    pos = dl_input[vol]["center"]
    r_dl = dl_input[vol]["r_dl"]
    z_dl = dl_input[vol]["z_dl"]
    return is_point_inside_polycone(x - pos[0], y - pos[1], z - pos[2], r_dl, z_dl)


def generate_mask_deadlayer(x, y, z, vol, para):
    with Path.open(para["file"]) as f:
        dl_input = json.load(f)

    def convert_to_numba_dict(py_dict):
        # Create the outer numba.typed.Dict
        nb_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.DictType(types.unicode_type, types.float64[:]),
        )

        for vol_key, vol_value in py_dict.items():
            # Create the inner dictionary for each volume (flatten surface_mesh fields)
            vol_dict = Dict.empty(
                key_type=types.unicode_type, value_type=types.float64[:]
            )

            # Convert "center" to a numpy array
            center_array = np.array(vol_value["center"], dtype=np.float64)
            vol_dict["center"] = center_array

            # Flatten "surface_mesh.orig.r", "surface_mesh.orig.z", "surface_mesh.dl.r", and "surface_mesh.dl.z"
            r_orig_array = np.array(
                vol_value["surface_mesh"]["orig"]["r"], dtype=np.float64
            )
            z_orig_array = np.array(
                vol_value["surface_mesh"]["orig"]["z"], dtype=np.float64
            )
            r_dl_array = np.array(
                vol_value["surface_mesh"]["dl"]["r"], dtype=np.float64
            )
            z_dl_array = np.array(
                vol_value["surface_mesh"]["dl"]["z"], dtype=np.float64
            )

            # Store the flattened arrays into the vol_dict
            vol_dict["r_orig"] = r_orig_array
            vol_dict["z_orig"] = z_orig_array
            vol_dict["r_dl"] = r_dl_array
            vol_dict["z_dl"] = z_dl_array

            # Store vol_dict inside the nb_dict with the integer key
            nb_dict[int(vol_key)] = vol_dict

        return nb_dict

    dl_input_numba = convert_to_numba_dict(dl_input)

    @njit
    def _internal(x, y, z, vol, dl_input):
        output = []
        for i in range(len(x)):  # event
            tmp = []
            for j in range(len(x[i])):  # window
                tmp2 = []
                for k in range(len(x[i][j])):  # detector
                    tmp3 = []
                    for m in range(len(x[i][j][k])):  # hit
                        if vol[i][j][k][m] > 1000000:
                            tmp3.append(
                                is_in_active_volume(
                                    x[i][j][k][m],
                                    y[i][j][k][m],
                                    z[i][j][k][m],
                                    vol[i][j][k][m],
                                    dl_input,
                                )
                            )
                    tmp2.append(tmp3)
                tmp.append(tmp2)
            output.append(tmp)
        return output

    return ak.Array(_internal(x, y, z, vol, dl_input_numba))


def m_active_volume(para, input, output, pv):
    """
    Active Volume module for the postprocessing pipeline.

    Given a condition on the location, the module filters the input data based on the condition.

    Parameters:
    para (dict): Dictionary containing parameters for the module.
        - active_threshold (float): Threshold for active volume.

    input (list): List of input parameters in the following order:
        - t: Name of times array.
        - edep: Name of energy depositions array.
        - vol: Name of volumes array.
        - posx: Name of x positions array.
        - posy: Name of y positions array.
        - posz: Name of z positions array.

    output (list): List of output parameters in the following order:
        - t: Array of times.
        - edep: Array of energy depositions.
        - vol: Array of volumes.
        - posx: Array of x positions.
        - posy: Array of y positions.
        - posz: Array of z positions.
        - vol_red: Array of reduced volumes.

    pv (dict): Dictionary to store the processed values.

    """
    if len(output) > 2:
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
            "vol_red": output[6],
        }

    else:
        in_n = {
            "w_t": input[0],
            "edep": input[1],
            "vol": input[2],
            "posx": input[3],
            "posy": input[4],
            "posz": input[5],
        }
        out_n = {"w_t": output[0], "edep": output[1]}

    if para["type"] == "cylinder":
        mask = generate_mask_cylinder(
            pv[in_n["posx"]], pv[in_n["posy"]], pv[in_n["posz"]], para
        )
        tmp = pv[in_n["edep"]][mask]
        mask_tmp = ak.Array([[True for arr2 in arr1 if len(arr2)] for arr1 in tmp])
        pv[out_n["edep"]] = tmp[mask_tmp]
        pv[out_n["w_t"]] = pv[in_n["w_t"]][ak.any(mask, axis=-1)]

    if para["type"] == "deadlayer":
        mask = generate_mask_deadlayer(
            pv[in_n["posx"]], pv[in_n["posy"]], pv[in_n["posz"]], pv[in_n["vol"]], para
        )
        pv[out_n["vol_red"]] = ak.firsts(pv[in_n["vol"]], axis=-1)

        for key in in_n:
            tmp = pv[in_n[key]][mask]
            mask_tmp = ak.Array([[True for arr2 in arr1 if len(arr2)] for arr1 in tmp])
            pv[out_n[key]] = tmp[mask_tmp]
