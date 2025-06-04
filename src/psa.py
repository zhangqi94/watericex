import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn

####################################################################################################

class ProductSpectraAnsatz(nn.Module):
    """
        Product spectrum ansatz.
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.100.032107
    """
    num_levels: int
    indices_group: int
    sequence_length: int

    def setup(self):
        self.prob_params = self.param('params_psa', nn.initializers.zeros,  
                        (self.sequence_length, 
                         self.num_levels**self.indices_group
                         )
                        )

    def __call__(self):
        return self.prob_params


####################################################################################################

def make_product_spectra_ansatz(num_levels, indices_group, sequence_length):
    van = ProductSpectraAnsatz(num_levels, indices_group, sequence_length)
    return van


####################################################################################################
if __name__ == '__main__':
   
    key = jax.random.key(42) 
    num_levels = 10
    indices_group = 1
    sequence_length = 12
    
    van = make_product_spectra_ansatz(num_levels, indices_group, sequence_length)
    params = van.init(key)
    logits = van.apply(params)
    print("logits.shape", logits.shape)
    print("logits:", logits)
    logp = jax.nn.log_softmax(logits, axis=-1)
    print("logp (after log_softmax):", logp)
    
    state_indices = jax.random.randint(key, (sequence_length,), 0, num_levels**indices_group)
    print("state_indices:", state_indices)
    state_onehot = jax.nn.one_hot(state_indices, num_levels**indices_group)
    print("state_onehot:", state_onehot)
    logp = jnp.sum(logp * state_onehot)
    print("logp (after summing over state_onehot):", logp)

