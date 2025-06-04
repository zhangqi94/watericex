import pickle
import os
import re
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

####################################################################################################
def ckpt_filename(epoch, path):
    return os.path.join(path, "epoch_%06d.pkl" % epoch)

def save_pkl_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_pkl_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def load_txt_data(file_path, print_type = 1):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    data = jnp.array(data, dtype=jnp.float64)
    if print_type == 1:
        print(data.shape)
    return data

####################################################################################################
def find_ckpt_filename(path_or_file):
    if os.path.isfile(path_or_file):
        epoch = int(re.search('epoch_([0-9]*).pkl', path_or_file).group(1))
        return path_or_file, epoch
    files = [f for f in os.listdir(path_or_file) if ('pkl' in f)]
    for f in sorted(files, reverse=True):
        fname = os.path.join(path_or_file, f)
        try:
            with open(fname, "rb") as f:
                pickle.load(f)
            epoch = int(re.search('epoch_([0-9]*).pkl', fname).group(1))
            return fname, epoch
        except (OSError, EOFError):
            print('Error loading checkpoint. Trying next checkpoint...', fname)
    return None, 0

