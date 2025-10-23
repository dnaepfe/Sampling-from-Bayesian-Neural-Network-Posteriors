import numpy as np
import matplotlib.pyplot as plt

"""
This code provides an example for sampling from a Gaussian posterior distrubtion 
using SGLD, SGLD-CV and ULA. Part of the code is from the example ULA vs MALA.
"""

rng = np.random.default_rng(42)

#simulate data according to gaussian distribution
def simulate_data(N_data,var,rng):
    true_theta = rng.normal()
    data = rng.normal(true_theta,np.sqrt(var),size=N_data)
    return true_theta, data

#compute closed form parameters of posterior distribution
def true_posterior(data,var):
    y = np.asarray(data, dtype=float)
    N = y.size
    sigma_N = 1 / (1 + (N / var))
    mu_N = sigma_N * (y.sum() / var)
    return mu_N, sigma_N


#-----------------ULA---------------------------------------

#gradient of log posterior
def grad_log_pi(theta,var,data):
    y = np.asarray(data,dtype=float)
    N = y.size
    A = 1 + N/var
    b = y.sum() / var
    return -A*theta + b

#ULA
def ULA(theta_0,n_iter,step_size,data,var,rng):
    samples = np.empty(n_iter)
    samples[0] = theta_0
    
    for k in range(0,n_iter-1):
        Z = rng.normal()
        samples[k+1] = samples[k] + 0.5*step_size*grad_log_pi(samples[k],var,data) + np.sqrt(step_size)*Z
    
    return samples


#-----------------SGLD--------------------------------------

#stochastic gradient of log posterior
def stoch_grad_log_pi(theta,var,data,m,rng):
    y = np.asarray(data,dtype=float)
    N = y.size
    minibatch_ind = rng.choice(N,size=m,replace=False)
    y_batch = y[minibatch_ind]
    
    A = 1 + (N/var)
    b = (N/m) * (y_batch.sum() / var)
    
    return -A*theta + b

#SGLD
def SGLD(theta_0,n_iter,step_size,data,m,var,rng):
    samples = np.empty(n_iter)
    samples[0] = theta_0
    
    for k in range(0,n_iter-1):
        Z = rng.normal()
        samples[k+1] = samples[k] + 0.5*step_size*stoch_grad_log_pi(samples[k], var, data, m, rng) + np.sqrt(step_size)*Z
        
    return samples


#------------------SGLD-CV---------------------------------

#find mode using SGD
def find_mode(data, var):
    mu_N, _ = true_posterior(data, var)
    
    return mu_N

def stoch_cv_grad_log_pi(theta,var,data,m,mode,full_grad_at_mode,rng):
    y = np.asarray(data,dtype=float)
    N = y.size
    minibatch_ind = rng.choice(N,size=m,replace=False)
    y_batch = y[minibatch_ind]
    
    A = 1 + (N/var)
    b = (N / m) * (y_batch.sum() / var)
    
    grad_theta = -A*theta + b
    grad_mode = -A*mode + b
    
    return full_grad_at_mode + (grad_theta - grad_mode)

#SGLD-CV
def SGLD_CV(theta_0,n_iter,step_size,data,m,var,rng):
    samples = np.empty(n_iter)
    samples[0] = theta_0
    
    mode = find_mode(data,var)
    full_grad_at_mode = grad_log_pi(mode,var,data)
    
    for k in range(0,n_iter-1):
        Z = rng.normal()
        grad_cv = stoch_cv_grad_log_pi(samples[k], var, data, m, mode, full_grad_at_mode, rng)
        samples[k+1] = samples[k] + 0.5*step_size*grad_cv + np.sqrt(step_size)*Z
    
    return samples

#------------------Simulation------------------------------

#data generation
N_data = 1000
var = 4

true_theta, data = simulate_data(N_data, var, rng)
mu_N, sigma_N = true_posterior(data,var)

#approximation algorithms 
n_iter = 10000
step_size = 1/1000
theta_0 = 0
m = 10

sample_ULA = ULA(theta_0,n_iter,step_size,data,var,rng)
sample_SGLD = SGLD(theta_0,n_iter,step_size,data,m,var,rng)
sample_SGLD_CV = SGLD_CV(theta_0, n_iter, step_size, data, m, var, rng)

#plot 
# Common grid and true posterior
xgrid = np.linspace(mu_N - 4*np.sqrt(sigma_N), mu_N + 4*np.sqrt(sigma_N), 400)
true_pdf = (1/np.sqrt(2*np.pi*sigma_N)) * np.exp(-0.5*((xgrid - mu_N)**2 / sigma_N))

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

sampler_data = [
    ("ULA", sample_ULA),
    ("SGLD", sample_SGLD),
    ("SGLD-CV", sample_SGLD_CV)
]

for ax, (name, samples) in zip(axes, sampler_data):
    ax.hist(samples, bins=60, density=True, alpha=0.5, color="grey",edgecolor="black", linewidth = 0.5)
    ax.plot(xgrid, true_pdf, color="steelblue", lw=2, label="True posterior")
    ax.set_title(name)
    ax.set_xlabel(r"$\theta$")
    ax.legend()

axes[0].set_ylabel("Density")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("SGLD_samplers", dpi=300, bbox_inches='tight')
plt.show()


    


