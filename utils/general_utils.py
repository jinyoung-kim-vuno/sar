import numpy as np

def pad_both_sides(dimension, vol, pad, pad_values) :
    pad_func = lambda vol, pad : np.pad(vol, pad, 'constant', constant_values=pad_values)

    if dimension == 2 :
        pad = (0, ) + pad

    padding = ((pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2]))

    if len(vol.shape) == 3 :
        return pad_func(vol, padding)
    else :
        return pad_func(vol, ((0, 0),) + padding)