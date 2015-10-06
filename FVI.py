import csv
import math
import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from Util import generate_sars
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
    #n_iters = math.log(((tol*(1-gamma**2))/Br) - gamma)
    
    # Initialize the weights. Do one iteration to get things started
    Xs = []
    ys = []
    for s, a, r, s_prime in sars:
        y = r
        Xs.append(np.append(s, a))
        ys.append(y)
    fn.fit(Xs, ys)
    for i in range(n_iters+1):
        Xs = []
        ys = []
        for s, a, r, s_prime in sars:
            print "Timing comp:"
            start = time.time()
            y = r + gamma * max(fn.predict(np.append(s_prime, 0.0)), fn.predict(np.append(s_prime, 1.0)))
            end = time.time()
            print end-start
            Xs.append(np.append(s, a))
            ys.append(y[0])
        fn.fit(Xs, ys)
        if (i%2==0): print "At iter, ",i
    return fn

if __name__=="__main__":
    with open('generated_episodes_3000.csv') as csv_file:
        data = np.array(list(csv.reader(csv_file))[1:])
    sars = generate_sars(data)
       
    (options, args) = parser.parse_args()
    if options.model == "lstsq":
        fn = linear_model.LinearRegression()
    else:
        n_neighbors = options.nn
        fn = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        
    FVI(fn, sars)
