import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

####################################################################################################
"""
    Computes the value of the Hermite polynomial H_n(x) using the formula:
        hermite(n, x) = (1/sqrt(2^n * n!)) / pi^(1/4) * H_n(x)
    
    This code is adapted from the following source: 
        https://code.itp.ac.cn/-/snippets/20
"""

def hermite0(index, x):
    h0 = 1. / jnp.pi**(1/4)
    h1 = jnp.sqrt(2.)*x / jnp.pi**(1/4)

    def body_fun(i, val):
        valm2, valm1 = val
        return valm1, jnp.sqrt(2./i)*x*valm1 - jnp.sqrt((i-1)/i)*valm2
    _, hn = jax.lax.fori_loop(2, index+1, body_fun, (h0, h1))

    return jax.lax.cond(index>0, lambda: hn, lambda: h0)

hermite = jax.custom_jvp(hermite0, nondiff_argnums=(0,))

@hermite.defjvp
def hermite_jvp(index, primal, tangent):
    x, = primal
    dx, = tangent
    hn = hermite(index, x)
    dhn = jnp.sqrt(2*index) * hermite((index-1)*(index>0), x) * dx
    primal_out, tangent_out = hn, dhn
    return primal_out, tangent_out

####################################################################################################

def make_orbitals_1d(m=1.0, hbar=1.0):
    """
    Generate the wavefunctions and energy eigenvalues for a 1D quantum harmonic oscillator.

    The wavefunction `psi_n(x)` for a 1D harmonic oscillator is given by:
        psi_n(x) = ((m * w)^(1/4)) * exp(-0.5 * m * w * x^2) * H_n(sqrt(m * w) * x)
    where:
        - `H_n` is the Hermite polynomial of order `n`,
        - `w` is the oscillator frequency (which is a parameter in the function),
        - `x` is the position.

    The energy levels `E_n` are given by:
        E_n = (n + 1/2) * w

    Returns:
    --------
    fn_wavefunctions : function
        A JAX vectorized function that computes the wavefunction for the harmonic oscillator
        at given indices (quantum number), position (`x`), and frequency (`w`).

    fn_energies : function
        A JAX vectorized function that computes the energy eigenvalues for the harmonic oscillator
        given the quantum numbers (`indices`) and the frequency (`w`).
    """

    @jax.vmap
    def fn_wavefunctions(indices, w, x):
        return jnp.exp( - 0.5 * (m*w/hbar)* (x**2)) * hermite(indices, jnp.sqrt(m*w/hbar) * x)

    @jax.vmap
    def fn_energies(indices, w):
        return hbar * w * ( indices + 0.5 )

    return fn_wavefunctions, fn_energies

####################################################################################################

def logphi_base(fn_wavefunctions, state_indices, wfreqs, coords):
    """
    Compute the logarithm of the absolute value of the wavefunction for a given quantum state 
    and position, and return the sum of these logarithmic values.
    
    Args:
    --------
    fn_wavefunctions : function
        A function that computes the wavefunction for a given set of quantum numbers, position, 
        and oscillator frequency.
    state_indices : int or ndarray
        The quantum state indices (or quantum numbers) for which the wavefunction should be evaluated.
    wfreqs : float
        The frequency parameter of the system (e.g., frequency of the oscillator).
    coords : ndarray
        The positions at which to evaluate the wavefunction.

    Returns:
    --------
    logphi_value : float
        The sum of the logarithms of the absolute values of the wavefunction evaluated at positions `x`.
    """
    
    phi_values = fn_wavefunctions(state_indices, wfreqs, coords)
    logphi_value = jnp.sum(jnp.log(jnp.abs(jnp.array(phi_values))))
    
    return logphi_value


####################################################################################################
if __name__ == '__main__':
    
    key = jax.random.key(42)
    
    num_modes = 10
    state_indices = jax.random.randint(key, shape=(num_modes, ), minval=0, maxval=2)
    wfreqs = jax.random.uniform(key, shape=(num_modes, ))
    coords = jax.random.uniform(key, shape=(num_modes, ))
    
    fn_wavefunctions, fn_energies = make_orbitals_1d()
    logphi_value = logphi_base(fn_wavefunctions, state_indices, wfreqs, coords)
    print("logphi_value:", logphi_value)

    energies = fn_energies(jnp.zeros(num_modes, dtype=jnp.int64), wfreqs)
    print("energies:", jnp.sum(energies), energies)

