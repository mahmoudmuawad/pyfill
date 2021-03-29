import numpy as np
from pc_functions import *

        
def predict_update(x, std, u, pre_model, z, sense_std, n_particles, sensor_range):
    # sampling particles 
    particles = particles_sample(x, std, n_particles)

    # sampling weights 
    weights = weights_sample(n_particles)
    
    # appply prediction model on particles 
    particles_pred = pre_particles(particles, pre_model, u)

    # adding noise to particles 
    pcs_pred_noise = add_noise(particles_pred, std)               
    
    # update weights according to probability measurement 
    weights = measure_prob(weights ,pcs_pred_noise ,z , sense_std)
    weights_norm = np.divide(weights, np.sum(weights))
    resampled_particles= resample(weights_norm, pcs_pred_noise)
    best_particle=resampled_particles[:, np.argmax(weights_norm)]
    return best_particle
        
  

