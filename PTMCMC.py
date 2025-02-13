'''Parallel tempering MCMC using Fisher and differential evolution jumps.'''


from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import wave_gen as wg
import likelihood as l


# function to decide jump proposal acceptance / rejection
def accept_reject(new_state, new_lnpost, accept_prob, prev_state, prev_lnpost, key):
    accept = jr.uniform(key) < accept_prob
    final_state = jnp.where(accept, jnp.copy(new_state), jnp.copy(prev_state))
    final_lnpost = jnp.where(accept, new_lnpost, prev_lnpost)
    return final_state, final_lnpost, accept

# vectorized acceptance / rejection
vectorized_accept_reject = jit(vmap(accept_reject, in_axes=(0, 0, 0, 0, 0, 0)))


# Parallel tempering swap
def PT_swap(num_chains,
            chain_ndx,
            temp_ladder,
            iteration,
            lnpost_func,
            jump_accept_counts,
            jump_reject_counts,
            samples,
            lnposts,
            keys):

    # track swaps
    swap_map = list(np.copy(chain_ndx))

    # store current states and likelihood values
    states = samples[:, iteration]
    lnlikes = lnposts[:, iteration] * temp_ladder

    # loop through and propose a swap at each chain (starting from hottest chain and going down in T)
    # and keep track of results in swap_map
    for j, swap_chain in enumerate(reversed(range(num_chains - 1))):
        log_acc_ratio = -lnlikes[swap_map[swap_chain]] / temp_ladder[swap_chain]
        log_acc_ratio += -lnlikes[swap_map[swap_chain + 1]] / temp_ladder[swap_chain + 1]
        log_acc_ratio += lnlikes[swap_map[swap_chain + 1]] / temp_ladder[swap_chain]
        log_acc_ratio += lnlikes[swap_map[swap_chain]] / temp_ladder[swap_chain + 1]
        acc_ratio = np.exp(log_acc_ratio)
        
        # accept or reject swap
        if jr.uniform(keys[j]) <= acc_ratio:  # accept
            swap_map[swap_chain], swap_map[swap_chain + 1] = swap_map[swap_chain + 1], swap_map[swap_chain]
            jump_accept_counts[-1, swap_chain] += 1
        else:  # reject
            jump_reject_counts[-1, swap_chain] += 1

    # record final states after all swaps
    final_states = np.array([states[swap_map_ndx] for swap_map_ndx in swap_map])
    final_lnposts = np.array([lnpost_func(state, temp) for state, temp in zip(final_states, temp_ladder)])
    samples[chain_ndx, iteration + 1] = final_states
    lnposts[chain_ndx, iteration + 1] = final_lnposts
    
    return



def PTMCMC(num_samples,
           num_chains,
           x0,
           ln_posterior_func,
           jump_proposals,
           PT_swap_weight=20):
    
    # temperature ladder with geometric spacing
    chain_ndxs = np.arange(num_chains)
    temp_ladder = 1.3 ** chain_ndxs

    # initialize samples and posterior values
    ndim = x0.shape[0]
    samples = np.zeros((num_chains, num_samples, ndim))
    lnposts = np.zeros((num_chains, num_samples))

    # all chains start at x0
    samples[:, 0] = np.tile(x0, (num_chains, 1))
    lnposts[:, 0] = np.array([ln_posterior_func(samp, temp)
                              for samp, temp in zip(samples[:, 0], temp_ladder)])
    
    # organize jump proposals
    num_jump_types = len(jump_proposals)
    jump_functions = []
    jump_names = []
    jump_weights = []
    for proposal in jump_proposals:
        jump_function, weight = proposal
        jump_functions.append(jump_function)
        jump_names.append(jump_function.__name__)
        jump_weights.append(weight)
    # add PT jump proposal
    num_jump_types += 1
    jump_names.append('PT_swap')
    jump_weights.append(PT_swap_weight)

    # make jump choices
    jump_selections = np.random.choice(num_jump_types, num_samples, p=jump_weights/np.sum(jump_weights))

    # track jump proposal accept and reject counts
    jump_accept_counts = np.zeros((num_jump_types, num_chains))
    jump_reject_counts = np.zeros((num_jump_types, num_chains))

    # main MCMC loop
    for i in range(num_samples - 1):

        # update progress ocassionally
        if i % (num_samples // 1000) == 0:
            print(f'{round(i / num_samples * 100, 3)}%', end='\r')

        # index of jump method
        jump_ndx = jump_selections[i]

        # independent random keys for chain updates
        keys = jr.split(jr.PRNGKey(i), num_chains)

        if jump_ndx == num_jump_types - 1:  # parallel tempering swap
            PT_swap(num_chains=num_chains,
                    chain_ndx=chain_ndxs,
                    temp_ladder=temp_ladder,
                    iteration=i,
                    lnpost_func=ln_posterior_func,
                    jump_accept_counts=jump_accept_counts,
                    jump_reject_counts=jump_reject_counts,
                    samples=samples,
                    lnposts=lnposts,
                    keys=keys)
            
        else:  # intra-chain updates

            # which jump proposal method
            vectorized_jump_function = jump_functions[jump_ndx]
            
            # propose jumps
            new_states = vectorized_jump_function(samples[chain_ndxs, i],
                                                  i,
                                                  temp_ladder,
                                                  keys)
            
            # evaluate posterior at new points
            new_states = np.array(new_states)
            new_lnposts = jnp.array([ln_posterior_func(state, temp) for state, temp in zip(new_states, temp_ladder)])

            # acceptance probabilities
            accept_probs = jnp.exp(new_lnposts - lnposts[chain_ndxs, i])

            # accept or reject proposal
            final_states, final_lnposts, accepted = vectorized_accept_reject(new_states,
                                                                             new_lnposts,
                                                                             accept_probs,
                                                                             samples[chain_ndxs, i],
                                                                             lnposts[chain_ndxs, i],
                                                                             keys)

            # convert updates to numpy arrays
            samples[chain_ndxs, i + 1] = jnp.asarray(final_states)
            lnposts[chain_ndxs, i + 1] = jnp.asarray(final_lnposts)

            # update acceptance / rejection counts
            jump_accept_counts[jump_ndx, chain_ndxs] += jnp.asarray(accepted, dtype=int)
            jump_reject_counts[jump_ndx, chain_ndxs] += jnp.asarray(1 - accepted, dtype=int)
        

    # compute jump acceptance rates
    jump_reject_counts[-1, -1] += 1  # hottest chain doesn't swap with hotter chain, prevents NaN
    accept_rates = jump_accept_counts / (jump_accept_counts + jump_reject_counts)
    print('Jump acceptance rates')
    for name, rate in zip(jump_names, accept_rates):
        print(f'{name}: {rate}')
    
    return samples, lnposts, temp_ladder
                        
