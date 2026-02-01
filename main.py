# rewrite it based on what Jiani will complete and what's already available

from quantum.bfdcqo import quantum_enhanced_mts

# GLOBAL VARIABLES

theta_cutoff = 0.01
bf_dcqo_iter = 3
pop_size = 50
mts_iter = 1000
alpha = 0.1
kappa = 5
n_iter = 3
T = 1.0
N = 15

result = quantum_enhanced_mts(N=N, pop_size=pop_size, bf_dcqo_iter=bf_dcqo_iter, 
                              mts_iter=mts_iter, alpha=alpha, kappa=kappa, 
                              T=T,
                              quantum_shots=1000)
