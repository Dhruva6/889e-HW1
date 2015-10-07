import csv
import math
import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from optparse import OptionParser
import time

def FVI(fn, sars, numFeat, gamma = 0.5, tol = 0.01):
    """
        Does fixed value iteration using the input fn approximator on the input data (expected to be in SARS format)
    """
    k = 2 * (numFeat)
    # indices for various operations
    sIdx = 0
    aIdx = sIdx + numFeat
    rIdx = aIdx + 1
    spIdx = rIdx + 1
    spIdxEnd = spIdx + k

    # From blackboard calculation
    Br = max([r for r in sars[:, rIdx]])

    # Lower bound for n_iters
    n_iters = (int)(math.log((tol*((1-gamma)**2))/(2.0*Br))/math.log(gamma))

    maxiters = 20
    if (n_iters > maxiters): n_iters = maxiters

    # console log
    print "Running for ", n_iters, "iterations"

    # Initialize the weights. Do one iteration to get things started
    Xs = []
    ys = []
    for idx in range(len(sars)):
        r = sars[idx, rIdx]
        y = r
        phi_s = np.zeros((k, 1))
        
        offset = 0 if sars[idx,aIdx] == -1.0 else numFeat
        phi_s[offset:offset+numFeat,0] = sars[idx,0:aIdx]

        Xs.append(phi_s.flatten())
        ys.append(y)
    fn.fit(Xs, ys)

    for i in range(n_iters+1):
        ys = []
        for idx in range(len(sars)):
            phi_s_prime_0 = np.zeros((k, 1))
            phi_s_prime_1 = np.zeros((k, 1))
            
            phi_s_prime_0[0:0+numFeat, 0] = sars[idx, spIdx:spIdxEnd]
            phi_s_prime_1[numFeat:numFeat+numFeat, 0] = sars[idx, spIdx:spIdxEnd]
            r = sars[idx, rIdx]
            y = r + gamma * max(fn.predict(phi_s_prime_0.flatten()), fn.predict(phi_s_prime_1.flatten()))
            ys.append(y[0])
        fn.fit(Xs, ys)
        if (i%2==0): print "At iter, ",i
    return fn
