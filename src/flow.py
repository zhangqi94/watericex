import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn

####################################################################################################
class RealNVP(nn.Module):
    """
        Real-valued non-volume preserving (real NVP) transform.
        The implementation follows the paper "arXiv:1605.08803."
    """
    flow_mask: list
    flow_layers: int
    flow_width: int
    flow_depth: int
    event_size: int

    def setup(self):
        # MLP (Multi-Layer Perceptron) layers for the real NVP.
        self.mlp = [self.build_mlp(self.flow_width, self.flow_depth, self.event_size)
                        for _ in range(self.flow_layers)]
        self.zoom = self.param('zoom', nn.initializers.ones, (self.flow_layers, self.event_size)
                               )
        
        factor_scale_initializer = nn.initializers.normal(stddev=0.001)
        # factor_scale_initializer = nn.initializers.normal(stddev=0.0)
        self.factor_s = self.param('factor_scale', 
                                lambda key, shape: 1.0 + factor_scale_initializer(key, shape), 
                                (self.event_size, ))
        self.factor_t = self.param('factor_shift', 
                                lambda key, shape: 0.0 + factor_scale_initializer(key, shape), 
                                (self.event_size, ))


    def build_mlp(self, flow_width, flow_depth, event_size):
        layers = []
        for _ in range(flow_depth):
            layers.append(nn.Dense((flow_width), dtype=jnp.float64))
            layers.append(nn.tanh)
        layers.append(nn.Dense(event_size * 2, 
                                kernel_init=nn.initializers.truncated_normal(stddev=0.0001), 
                                bias_init=nn.initializers.zeros,
                                dtype=jnp.float64))
        return nn.Sequential(layers)

    def coupling_forward(self, x1, x2, l):
        # get shift and log(scale) from x1
        shift_and_logscale = self.mlp[l](x1)
        shift, logscale = jnp.split(shift_and_logscale, 2, axis=-1)
        logscale = jnp.where(self.flow_mask[l], 0, jnp.tanh(logscale) * self.zoom[l])
        
        # transform: y2 = x2 * scale + shift
        y2 = x2 * jnp.exp(logscale) + shift
        # calculate: logjacdet for each layer
        sum_logscale = jnp.sum(logscale)
        return y2, sum_logscale

    def __call__(self, x):
        # Real NVP (forward)
        # x.shape should be: d1 = num_modes, d2 = 1
        d1, d2 = x.shape  
        
        # initial x and logjacdet
        x_flatten = x.flatten()
        
        x_flatten = self.factor_s * x_flatten + self.factor_t
        logjacdet = jnp.sum(jnp.log(self.factor_s))
        
        for l in range(self.flow_layers):
            # split x into two parts: x1, x2
            x1 = jnp.where(self.flow_mask[l], x_flatten, 0)
            x2 = jnp.where(self.flow_mask[l], 0, x_flatten)
            
            # get y2 from fc(x1), and calculate logjacdet = sum_l log(scale_l)
            y2, sum_logscale = self.coupling_forward(x1, x2, l)
            logjacdet += sum_logscale

            # update: [x1, x2] -> [x1, y2]
            x_flatten = jnp.where(self.flow_mask[l], x_flatten, y2)
            
        x = jnp.reshape(x_flatten, (d1, d2))
        
        return x, logjacdet 

####################################################################################
def make_flow_mask(flow_layers, event_size):
    mask1 = jnp.arange(0, jnp.prod(event_size)) % 2 == 0
    mask1 = mask1.reshape(event_size).astype(bool)
    mask2 = ~mask1

    flow_mask = [mask1 if i % 2 == 0 else mask2 for i in range(flow_layers)]
    flow_mask = jnp.array(flow_mask)
    return flow_mask

####################################################################################################
## phonon
def make_flow_model(flow_layers, flow_width, flow_depth, num_modes):

    event_size = num_modes
    flow_mask = make_flow_mask(flow_layers, 
                               event_size
                               )
    flow = RealNVP(flow_mask, 
                   flow_layers, 
                   flow_width, 
                   flow_depth, 
                   event_size, 
                   )
    
    return flow

####################################################################################################
if __name__ == "__main__":
    
    print("\n========== Test Real NVP ==========")
    import time
    from jax.flatten_util import ravel_pytree
    key = jax.random.key(42)
    
    flow_layers = 4
    flow_width = 32
    flow_depth = 2

    flow_layers = 0
    flow_width = 32
    flow_depth = 2
    
    #========== make flow model with scaling ==========
    if 0:
        num_modes = 12
        flow = make_flow_model(flow_layers, flow_width, flow_depth, num_modes)
        x = jax.random.uniform(key, (num_modes, 1), dtype=jnp.float64)
        params = flow.init(key, x)
        raveled_params, _ = ravel_pytree(params)
        print("#parameters in the flow model: %d" % raveled_params.size, flush=True)

        t1 = time.time()  
        z, logjacdet = flow.apply(params, x)
        t2 = time.time()
        print("direct logjacdet:", logjacdet, ",  time used:", t2-t1)

        t1 = time.time()
        x_flatten = x.reshape(-1)
        flow_flatten = lambda x: flow.apply(params,x.reshape(num_modes, 1))[0].reshape(-1)
        jac = jax.jacfwd(flow_flatten)(x_flatten)
        _, logjacdet = jnp.linalg.slogdet(jac)
        t2 = time.time()
        print("jacfwd logjacdet:", logjacdet, ",  time used:", t2-t1)
        
        print("x:", x.shape, x.flatten())
        print("z:", z.shape, z.flatten())
    
    #========== test atomic coordinates ==========
    if 1:        
        num_atoms, dim = 4, 3
        flow = make_flow_model(flow_layers, flow_width, flow_depth, num_atoms*dim)
        x = jax.random.uniform(key, (num_atoms, dim), dtype=jnp.float64)
        params = flow.init(key, x)
        raveled_params, _ = ravel_pytree(params)
        print("#parameters in the flow model: %d" % raveled_params.size, flush=True)
        
        t1 = time.time()  
        z, logjacdet = flow.apply(params, x)
        t2 = time.time()
        print("direct logjacdet:", logjacdet, ",  time used:", t2-t1)
        
        t1 = time.time()
        x_flatten = x.reshape(-1)
        flow_flatten = lambda x: flow.apply(params,x.reshape(num_atoms, dim))[0].reshape(-1)
        jac = jax.jacfwd(flow_flatten)(x_flatten)
        _, logjacdet = jnp.linalg.slogdet(jac)
        t2 = time.time()
        print("jacfwd logjacdet:", logjacdet, ",  time used:", t2-t1)
        
        print("x:", x.shape, x)
        print("z:", z.shape, z)

