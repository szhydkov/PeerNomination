from scipy.optimize import root_scalar
import scipy
import numpy as np

def prob_sample(n, m, x, r):

    t = scipy.special.comb(r-1, x-1, exact=True)*scipy.special.comb(n-r, m-x, exact=True)/scipy.special.comb(n-1, m-1, exact=True)

    return t

"Probability of being accepted by the algorithm given position r in the ranking."
def prob_acc(n, m, k, r, eps=0):
    quota = k*m/n + eps
    q = sum([prob_sample(n, m, i, r) for i in range(1, int(np.floor(quota))+1)]) + (quota-np.floor(quota))*prob_sample(n, m, np.floor(quota)+1, r)
    return sum([scipy.special.comb(m, i)*(q**i)*(1-q)**(m-i) for i in range(int(np.ceil(m/2)), m+1)])

"Expected accuracy of the algorithm."
# exp_acc(n, m, k, eps=0) = sum([prob_acc(n, m, k, r, eps) for r in 1:k])/k
def exp_size(n, m, k, eps=0):
    return sum([prob_acc(n, m, k, r, eps) for r in range(1, n+1)])

# "Varance of true positives."
# exp_tp(n, m, k, eps=0) = sum([prob_acc(n, m, k, r, eps) for r in 1:k])
# var_tp(n, m, k, eps=0) = sum([prob_acc(n, m, k, r, eps)*(1-prob_acc(n, m, k, r, eps)) for r in 1:k])

"Estimate epsilon given n, m, k to produce the correct expected size."
def estimate_eps(n, m, k):
    f = lambda x : exp_size(n, m, k, x) - k
    return root_scalar(f, bracket=[-1,1]).root
