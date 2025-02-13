'''Script define injected parameters, data, and associated objects.'''


import numpy as np
import wave_gen as wg


# injected parameters
m1_inj = 60.
m2_inj = 34.
chi1_inj = -0.11
chi2_inj = 0.63
log_dist_inj = 58.07
phic_inj = np.pi / 1.2
x_inj = np.array([m1_inj, m2_inj, chi1_inj, chi2_inj, log_dist_inj, phic_inj])

# make data waveform object
data_h22 = wg.get_h22(x_inj)

# amplitude and phase of data in frequency-domain
data_amp, data_phase = np.array(data_h22.amp), np.array(data_h22.phase)

data_FD_waveform = wg.FD_waveform(x_inj)
