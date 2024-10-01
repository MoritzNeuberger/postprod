
import awkward as ak
from numba import njit

def subtract_smallest_time(t_sv,t_all):
    return t_sv - ak.min(t_all,axis=1)


def define_windows(t_sub,dT = 1e4,t_max=1e8):
    t_sub_sort = ak.sort(t_sub,axis=1)
    
    @njit
    def create_windows(vec,wdl,mtd):
        output = list()
        for x in vec:
            tmp_list = [0]
            for y in x:
                if y > tmp_list[-1] + wdl and y < mtd:
                    tmp_list.append(y)
            tmp_list.append(mtd)
            tmp_list.append(1e20)
            output.append(tmp_list)
        return output
    
    
    return ak.Array(create_windows(t_sub_sort,dT,t_max))


def generate_map(t_sub,w_t):

    @njit
    def _internal(t_sub,w_t):
        output = list()
        for i in range(len(t_sub)):
            tmp_list = list()
            for k in range(len(t_sub[i])):
                for j in range(len(w_t[i]) - 1):
                    if t_sub[i][k] >= w_t[i][j] and t_sub[i][k] < w_t[i][j+1]:
                        tmp_list.append(j)
            output.append(tmp_list)
        return output
    return ak.Array(_internal(t_sub,w_t))

@njit
def generate_windowed_hit_list(mapping,v_in,index):
    output = list()
    for i in range(len(mapping)):
        if mapping[i] == index:
            output.append(v_in[i])
    if len(output) == 0:
        output.append(0)
    return output

@njit
def generate_all_windowed_hit_lists(mapping,v_in,i):
    list_in_indices = list()
    output = list()
    for i in range(len(mapping)):
        if mapping[i] not in list_in_indices:
            list_in_indices.append(mapping[i])
    for i in range(len(list_in_indices)):
        output.append(generate_windowed_hit_list(mapping,v_in,i))

    return output

def generate_all_windowed_hits_for_all_events(mapping,v_in):

    @njit
    def _internal(mapping,v_in):
        output = list()
        for i in range(len(v_in)):
            output.append(generate_all_windowed_hit_lists(mapping[i],v_in[i],i))
        return output
    
    return ak.Array(_internal(mapping,v_in))


def m_window(para, input, output, pv):

    in_n = {
        "t_all": input[0],
        "t": input[1],
        "edep": input[2],
        "vol": input[3],
        "posx": input[4],
        "posy": input[5],
        "posz": input[6]
    }

    out_n = {
        "w_t": output[0],
        "t_sub": output[1],
        "edep": output[2],
        "vol": output[3],
        "posx": output[4],
        "posy": output[5],
        "posz": output[6]
    }

    t_sub = subtract_smallest_time(pv[in_n["t"]],pv[in_n["t_all"]])
    w_t = define_windows(t_sub,para["dT"],para["t_max"])
    map = generate_map(t_sub,w_t)

    pv[out_n["t_sub"]] = generate_all_windowed_hits_for_all_events(map,t_sub) #t_sub
    pv[out_n["w_t"]] = w_t

    for key in list(out_n.keys())[2:]:
        pv[out_n[key]] = generate_all_windowed_hits_for_all_events(map,pv[in_n[key]])



    