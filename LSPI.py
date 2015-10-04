
# coding: utf-8

# Solution to the first homework of Real life reinforcement learning
import csv
import numpy as np
from sklearn import linear_model
from Util import generate_sarsa

# the Least Squares Temporal Difference Q learning algorithm
# Input: sars         - the nx4 array of (s,a,r,s') elements
#        current_pi   - the current policy at s' (nx1)
#        gamma        - the discount factor (defaults to 0.9)
#
# Output: w_pi - the estimated weight for the linear model
def LSTDQ(sars, current_pi, gamma = 0.9):

    # at present, our basis is N^d - where d is 9 (features) + 1 (action)
    k = 10

    # to avoid singularities, we start off A with a small delta along the diagonal
    delta = 1e-03

    # the "A" Matrix (kxk) 
    A = delta*np.eye(k)
    
    # the b vector
    b = np.zeros([k, 1], dtype=float)

    # iterate through and build A and b
    for idx in range(0, len(sars)):
        
        # In future implementations, we can have a different basis such as
        # RBF or a polynomial for our features
        # In the current implementation, we take the given features as the basis 
        phi_s = np.reshape(np.append(sars[idx,0], sars[idx,1]), (k,1))
        phi_s_prime = np.reshape(np.append(sars[idx,3], current_pi[idx,0]), (k,1))
        
        # Update A - here, we add to A, a Rank 1 matrix formed by
        # the vectors phi(s) and (phi(s) - gamma*phi(s'))
        A = A + np.outer(phi_s, phi_s - gamma*phi_s_prime)

        # update B - we add to B, the feature vector scaled by the reward
        b = b + sars[idx,2]*phi_s

    # compute the weights for the policy pi - solve the system A*w_pi = b
    w_pi,_,_,_ = np.linalg.lstsq(A, b)

    return w_pi

# Policy Improvement
def ImprovePolicy(s, w_pi):

    # the new policy
    policy = np.zeros((len(s),1))
    
    # iterate through every state, 
    for idx in range(0, len(s)):
        # the state-action value for action 0.0
        q0 = np.dot(np.append(s[idx],0.0), w_pi)

        # the state-action value for action 1.0
        q1 = np.dot(np.append(s[idx],1.0), w_pi)

        # update the policy as argmax(action = {0.0, 1.0}) Q^
        policy[idx] = 1.0 if q0 < q1 else 0.0

        # to the next state
        idx = idx+1    

    return policy

# prompt
print "Reading file: generated_episodes_3000.csv"

# First read in all the data from the file.
with open('generated_episodes_3000.csv') as csv_file:
    data = np.array(list(csv.reader(csv_file))[1:])

# Uncomment the next statement if you want to see what a row of the data looks like.
#print data[0]

# Generate the (s,a,r,s',a') tuple from data
sarsa = np.array(generate_sarsa(data))

# pull out the (s,a,r,s) tuple from sarsa
sars = sarsa[:,0:4]

#  the initial policy executed at s'
current_pi = np.reshape(sarsa[:,4], (len(sars),1))

# Instantiation of the Ordinary Least Squares class
ols = linear_model.LinearRegression()

#
# LSPI loop
#

# the maximum number of iterations to run
maxIter = 1

# the current loop counter
iter = 0

# loop
while iter < maxIter:

    # Estimate the State-Action VF Approximation using LSTDQ
    w_pi = LSTDQ(sars, current_pi)

    # improve the policy
    new_pi = ImprovePolicy(sars[:,0], w_pi)

    # termination condition
    # we can do a sum of per element diff between current_pi and new_pi
    # if this sum is smaller than a TBD number, we conclude that the
    # policy cannot be improved any further.

    # update current_pi
    current_pi = new_pi
    
    # update iter
    iter = iter + 1

