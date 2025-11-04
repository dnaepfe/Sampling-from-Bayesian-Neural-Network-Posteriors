import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

'''
This is an introductory example of approximating a standard Gaussian posterior distribution using the
Unadjusted Langevin Algorithm.

We assume that the true posterior distribution is proportional to exp(-1/2 * theta^2)
'''
rng = np.random.default_rng(11)

#step size
delta = 0.1 

#initial value
theta_0 = 0.0

#number of iterations
n=10000

#initialize approximation sample
theta = np.zeros(n)
theta[0] = theta_0

for k in range(0,n-1):
    Z = rng.normal()
    theta[k+1] = theta[k] - 0.5*delta*theta[k] + np.sqrt(delta)*Z

#plot
x = np.linspace(-3,3,200)
plt.figure(figsize=(6,4))
plt.hist(theta,bins=50,density=True,alpha=0.6,label = "ULA samples",color="grey",edgecolor="black", linewidth = 0.5)
plt.plot(x,norm.pdf(x,0,1),color="steelblue", lw=2,label="True standard Gaussian")
plt.xlabel(r"$\theta$")
plt.ylabel("Density")
plt.legend()
plt.savefig("ula_example", dpi=300, bbox_inches='tight')
plt.show()
