'''Jump proposals for MCMC.'''


import numpy as np
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import wave_gen as wg
import likelihood as l



# Fisher jumps
class Fisher:

    def __init__(self, x0):
        self.x0 = x0  # initial state where to compute Fisher
        self.get_Fisher_func = l.get_Fisher  # function to get Fisher

        # store initial Fisher
        self.Fisher = self.get_Fisher_func(self.x0)
        self.vals, self.vecs = jnp.linalg.eigh(self.Fisher)

        # vectorize Fisher jump across chains
        self.fast_Fisher_jump = jit(self.Fisher_jump)
        self.vectorized_Fisher_jump = jit(vmap(self.fast_Fisher_jump, in_axes=(0, None, 0, 0)))

    
    # jump along eigenvectors of Fisher
    def Fisher_jump(self, state, iteration, temperature, key):
        keys = jr.split(key, 2)
        # select direction to jump along
        direction = jr.choice(keys[0], state.shape[0])
        # jump along eigenvector of Fisher scaled by eigenvalue
        jump = 1. / jnp.sqrt(jnp.abs(self.vals[direction])) * self.vecs[:, direction]
        jump *= jr.normal(keys[1]) * jnp.sqrt(temperature)
        return state + jump
    
    

# Differential evolution
class DifferentialEvolution:

    def __init__(self, len_history):
        self.len_history = len_history  # how many samples in adaptive history
        self.x_min = wg.x_mins
        self.x_max = wg.x_maxs
        self.ndim = wg.ndim
        self.jump_weight = 2.38 / jnp.sqrt(2. * self.ndim)

        # initialize adaptive history
        self.history = jr.uniform(jr.PRNGKey(22), minval=self.x_min, maxval=self.x_max,
                                  shape=(self.len_history, self.ndim))
        
        self.fast_DE_jump = jit(self.DE_jump)

        # vectorize jump over chains
        self.vectorized_DE_jump = jit(vmap(self.fast_DE_jump, in_axes=(0, None, 0, 0)))
        
    def DE_jump(self, state, iteration, temperature, key):
        # split random keys
        draw1_key, draw2_key, weight_key, epsilon_key, replacement_key = jr.split(key, 5)
        # get jump
        jump = jr.choice(draw1_key, self.history) - jr.choice(draw2_key, self.history)
        jump *= jr.normal(weight_key) * self.jump_weight
        jump += jr.normal(epsilon_key, shape=(self.ndim,)) * 1.e-4
        # move to new state
        new_state = state + jump
        # update history
        self.history = self.history.at[jr.choice(replacement_key, self.len_history)].set(jnp.copy(state))
        return new_state


