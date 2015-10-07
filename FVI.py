import csv
import math
import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from optparse import OptionParser
import time

def FVI(fn, sars, gamma = 0.5, tol = 0.01):
    """
        Does fixed value iteration using the input fn approximator on the input data (expected to be in SARS format)
    """
    # From blackboard calculation
    Br = max([r for s, a,r, s_prime in sars])
    # Lower bound for n_iters
    n_iters = (int)(math.log((tol*((1-gamma)**2))/(2.0*Br))/math.log(gamma))

    maxiters = 20
    if (n_iters > maxiters): n_iters = maxiters
    
    # Initialize the weights. Do one iteration to get things started
    Xs = []
    ys = []
    for s, a, r, s_prime in sars:
        y = r
        Xs.append(np.append(s, a))
        ys.append(y)
    fn.fit(Xs, ys)

    for i in range(n_iters+1):
        ys = []
        for _, _, r, s_prime in sars:
            y = r + gamma * max(fn.predict(np.append(s_prime, 0.0)), fn.predict(np.append(s_prime, 1.0)))
            ys.append(y[0])
        fn.fit(Xs, ys)
        if (i%2==0): print "At iter, ",i
    return fn
