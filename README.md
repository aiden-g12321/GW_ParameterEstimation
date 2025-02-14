# GW_ParameterEstimation
Gravitational wave (GW) modeling and parameter-estimation (PE) with parallel tempering Markov Chain Monte Carlo (PTMCMC).
The l = 2, m = 2 mode is modeled in the frequency-domain with IMRPhenomD. The noise weighted inner product uses
the aLIGO theoretical sensitivity curve.
The PTMCMC algorithm uses jump proposals along the eigenvectors of the Fisher matrix 
and differential evolution jumps.

The MCMC samples over parameters: [$M_1$, $M_2$, $\chi_1$, $\chi_2$, $\log D_L$, $\phi_c$].
The masses are measured in units of solar masses, luminosity distance is in meters,
other parameters are dimensionless.
