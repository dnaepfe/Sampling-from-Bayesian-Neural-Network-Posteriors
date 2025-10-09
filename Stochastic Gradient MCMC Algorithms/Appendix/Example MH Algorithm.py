####################################################
#### Example of the Metropolis-Hasting algorithm ###
####################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


"""
We want to model the number of heads of a (possibly) fair coin toss. We 
assume a binomial distribution for the likelihood and a beta distribution with
parameters r,s=2 as a prior. Assume we observed x=7 heads out of n=10 trials.
Using the Metropolis-Hasting algorithm, we derive a sample of the posterior 
distribition. 
"""

#prior parameters
r = 2
s = 2

#likelihood parameters (observed data)
n = 10
x = 7

#unnormalized posterior: prior * likelihood
def target_density(theta):
    
    #reject invalid parameters
    if theta <= 0 or theta >= 1:
        return 0 
    
    likelihood = theta**x * (1 - theta)**(n-x)
    prior = theta**(r-1) * (1 - theta)**(s-1)
    
    return likelihood * prior

#Metropolis-Hasting sampler
def metropolis_hastings(num_samples, std):
    
    samples = []
    
    #initial value
    theta = 0.5
    
    for _ in range(num_samples):
        
        # Propose new theta
        theta_proposal = np.random.normal(theta, std)
        
        # Compute acceptance probability
        alpha = min(1, target_density(theta_proposal) / target_density(theta))
        
        # Accept or reject
        if np.random.rand() < alpha:
            theta = theta_proposal
        
        samples.append(theta)
    
    return np.array(samples)

# Run the sampler
samples = metropolis_hastings(10000,0.1)

# True posterior
theta_vals = np.linspace(0, 1, 200)
true_pdf = beta.pdf(theta_vals, r + x, s + n - x)

#burn in
burn_in = 1000
post = samples[burn_in:]

#Plots

#histogram and true posterior
plt.hist(post,bins=50,density=True,alpha=0.6,label="MH samples after burn in")
plt.plot(theta_vals,true_pdf,lw=2,label='True Beta posterior')
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
plt.legend()
plt.show()

#trace plot
plt.plot(samples, linewidth=0.8)
plt.ylabel(r'$\theta$')
plt.xlabel('Iteration')
plt.show()


