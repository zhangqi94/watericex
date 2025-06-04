

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp



####################################################################################################
def jax_linear_interpolation(d_points, s_points):
    def interpolate(d):
        idx = jnp.searchsorted(d_points, d, side="left") - 1
        idx = jnp.clip(idx, 0, len(d_points) - 2)
        
        d0, d1 = d_points[idx], d_points[idx + 1]
        s0, s1 = s_points[idx], s_points[idx + 1]
        return s0 + (s1 - s0) * (d - d0) / (d1 - d0)
    return interpolate


def load_txt_data(file_path, print_type = 1):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    data = np.array(data, dtype=np.float64)
    if print_type == 1:
        print(data.shape)
    return data


def load_energy_data(file_path):
    data = load_txt_data(file_path, print_type=0)
    x, y = data[:,1], data[:,2]
    y = y - y[0]
    x = np.concatenate([-np.flip(x), x])
    y = np.concatenate([ np.flip(y), y]) / 16
    return x, y


