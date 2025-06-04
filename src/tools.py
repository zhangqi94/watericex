import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pickle
import os
import numpy as np
from importlib.metadata import version

####################################################################################################
# if version('jax') > '0.4.25':
#     jaxtreemap = jax.tree.map
# else:
#     jaxtreemap = jax.tree_map
    
def get_jaxtreemap():
    if version('jax') > '0.4.25':
        jaxtreemap = jax.tree.map
    else:
        jaxtreemap = jax.tree_map
    return jaxtreemap

jaxtreemap = get_jaxtreemap()

def get_jaxkey():
    if version('jax') > '0.4.25':
        jaxrandomkey = jax.random.key
    else:
        jaxrandomkey = jax.random.PRNGKey
    return jaxrandomkey

jaxrandomkey = get_jaxkey()

####################################################################################################
def shard(x):
    return x

def replicate(pytree, num_devices):
    dummy_input = jnp.empty(num_devices)
    return jax.pmap(lambda _: pytree)(dummy_input)

####################################################################################################

def automatic_mcstddev(mc_stddev, accept_rate, target_acc=0.4):
    if accept_rate > (target_acc+0.100):
        mc_stddev *= 1.2
    elif (target_acc+0.025) < accept_rate <= (target_acc+0.100):
        mc_stddev *= 1.05
    elif (target_acc-0.025) < accept_rate <= (target_acc+0.025):
        mc_stddev *= 1.0
    elif (target_acc-0.100) < accept_rate <= (target_acc-0.025):
        mc_stddev *= 0.95
    elif accept_rate <= (target_acc-0.100):
        mc_stddev *= 0.8
    return mc_stddev

