delta_pos = normal(0, sigma, size=3*N_samples)
delta_pos = delta_pos.reshape(3,N_samples)
pos = cumsum(delta_pos, axis=-1)

