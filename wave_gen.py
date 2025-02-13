'''Script to generate gravitational waveforms using IMRPhenomD.

IMPORTANT: the input parameters and those sampled with MCMC are (in order)
[mass 1, mass 2, spin 1, spin 2, log(luminosity distance), phase at coalesence].
Masses are measured in units of solar mass and luminosity distance is in meters.
All other parameters are dimensionless.'''


import numpy as np
import IMRPhenomD.IMRPhenomD_const as imrc
from IMRPhenomD.IMRPhenomD import AmpPhaseFDWaveform, IMRPhenomDGenerateh22FDAmpPhase



# parameter attributes
x_mins = np.array([2.5, 2.5, -0.97, -0.97, np.log(1.e7 * imrc.PC_SI), 0.])
x_maxs = np.array([100.0, 100.0, 0.97, 0.97, np.log(1.e9 * imrc.PC_SI), np.pi])
ndim = x_mins.shape[0]
x_labels = [r'$m_1\;(M_\odot)$', r'$m_2\;(M_\odot)$', r'$\chi_1$', r'$\chi_2$', r'$\ln(D_L/1m)$', r'$\phi_c$']


# frequency bins
f_min = 10.
f_max = 1024.
# Nf = 2**9 + 1
Nf = 2**12 + 1
f = np.linspace(f_min, f_max, Nf)
df = f[1] - f[0]


# get waveform object (for l=2, m=2 mode)
# input parameters as array in order described above
def get_h22(x):
            
    # mass in solar masses
    # distance in ln(luminosity distance / meter)
    m1, m2, chi1, chi2, log_dist, phic = x
    
    # convert masses in kg
    m1_SI =  m1*imrc.MSUN_SI
    m2_SI =  m2*imrc.MSUN_SI
    
    # distance in meters
    distance = np.exp(log_dist)

    # initialize amplitudes and times
    amp_imr = np.zeros(Nf)
    phase_imr = np.zeros(Nf)
    time_imr = np.zeros(Nf)
    timep_imr = np.zeros(Nf)

    # use peak reference frequency
    MfRef_in = 0

    #the first evaluation of the amplitudes and phase will always be much slower, because it must compile everything
    h22 = AmpPhaseFDWaveform(Nf, f, amp_imr, phase_imr, time_imr, timep_imr)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22, f, phic, MfRef_in, m1_SI, m2_SI, chi1, chi2, distance)
    
    return h22


# get frequency-domain waveform
def FD_waveform(x):
    h22 = get_h22(x)
    return h22.amp * np.exp(-1.j * h22.phase)


# compute partial derivative of frequency-domain waveform
# (used for Fisher evaluation)
epsilon = 1.e-6
def partial_FD_waveform(x, deriv_ndx):
    # central finite differencing
    delta_x = np.zeros(ndim)
    delta_x[deriv_ndx] = epsilon
    return (FD_waveform(x + delta_x) - FD_waveform(x - delta_x)) / (2. * epsilon)


