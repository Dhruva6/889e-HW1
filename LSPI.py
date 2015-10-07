
# coding: utf-8

# Solution to the first homework of Real life reinforcement learning
import csv
import math
import random
import pickle
import numpy as np
from sklearn import linear_model
from Util import generate_sarsa
from Util import generate_test_states
from optparse import OptionParser

# the Least Squares Temporal Difference Q learning algorithm
# Input: sars         - the nx4 array of (s,a,r,s') elements
#        current_pi   - the current policy at s' (nx1)
#        gamma        - the discount factor (defaults to 0.9)
#
# Output: w_pi - the estimated weight for the linear model
def LSTDQ(sars, current_pi, gamma = 0.9, useRBFKernel = False, k=10):
    # if configured to use the RBF Kernel
    if useRBFKernel == True:
        phi = computePhiRBF(sars[0,0], 0.0)
        k = len(phi)
        

    # to avoid singularities, we start off A with a small delta along the diagonal
    delta = 1e-09

    # the "A" Matrix (kxk) 
    A = delta*np.eye(k)
    
    # the b vector
    b = np.zeros([k, 1], dtype=float)

    # iterate through and build A and b
    for idx in range(0, len(sars)):
                
        if useRBFKernel == True:
            phi_s = computePhiRBF(sars[idx,0], sars[idx,1])
            phi_s_prime = computePhiRBF(sars[idx,3], current_pi[idx,0])
        else:
            phi_s = np.reshape(np.append(sars[idx,0], sars[idx,1]), (k,1))
            phi_s_prime = np.reshape(np.append(sars[idx,3], current_pi[idx,0]), (k,1))
        
        # Update A - here, we add to A, a Rank 1 matrix formed by
        # the vectors phi(s) and (phi(s) - gamma*phi(s'))
        A = A + np.outer(phi_s, phi_s - gamma*phi_s_prime)
        #A = A + phi_s * (phi_s - gamma * phi_s_prime).T
        
        # update B - we add to B, the feature vector scaled by the reward
        b = b + sars[idx,2]*phi_s

    # compute the weights for the policy pi - solve the system A*w_pi = b
    w_pi,_,_,_ = np.linalg.lstsq(A, b)

    return w_pi

# Policy Improvement
def ImprovePolicy(s, w_pi, useRBFKernel = False):

    # the new policy
    policy = np.zeros((len(s),1))

    # the value of the improved policy
    value = np.zeros((len(s),1))

    # iterate through every state, 
    for idx in range(len(s)):

        # State-Action value function for actions 0.0 and 1.0
        if useRBFKernel == True:
            q0 = np.dot(computePhiRBF(s[idx], 0.0).T, w_pi)
            q1 = np.dot(computePhiRBF(s[idx], 1.0).T, w_pi)
        else:
            q0 = np.dot(np.append(s[idx],0.0), w_pi)
            q1 = np.dot(np.append(s[idx],1.0), w_pi)
        

        # update the policy as argmax(action = {0.0, 1.0}) Q^
        policy[idx] = 1.0 if q1 > q0 else 0.0

        # update the value
        value[idx] = max(q0, q1)
        
    return (policy, value)

def LSPI(sars, current_pi, gamma, useRBFKernel = False, k = 10):
    # the maximum number of iterations to run
    maxIter = 5

    # the current loop counter
    iter = 1

    # epsilon tolerance to terminate the policy improvement
    eps = 1e-02;

    # the initial weight vector
    if useRBFKernel == True:
        phi = computePhiRBF(sars[0,0], 0.0)
        w_pi = np.zeros((len(phi),1))
    else:
        w_pi = np.zeros((k,1))

    # the current value for all state-action pairs 
    current_value = np.zeros((len(sars),1))

    # loop
    while iter < maxIter:

        if 0 == iter%2:
            print "Now at policy iteration #{}".format(iter)
            
        # Estimate the State-Action VF Approximation using LSTDQ
        new_w_pi = LSTDQ(sars, current_pi, gamma, useRBFKernel, k=k)

        # improve the policy
        new_pi, new_value = ImprovePolicy(sars[:,0], new_w_pi, useRBFKernel)

        # termination condition
        if np.linalg.norm(new_w_pi - w_pi) < eps:
            print "PI converged at iteration # {}".format(iter)
            break
            
        # update current_pi
        current_pi = new_pi

        # update w_pi
        w_pi = new_w_pi

        # update current_value
        current_value = new_value
            
        # update iter
        iter = iter + 1

    return (current_pi, w_pi, current_value)

def computePhiRBF(s, a):

    # 3 kernels per dimension, centered at mu
    mu_kernels = [0.25, 0.5, 0.75]

    # get the state in column major format
    if len(s) == 1:
        s = s.T

    # the phi
    phi = np.zeros((2*(len(s) * len(mu_kernels) + 1), 1))

    # start counter
    idx = 0 if a == 0.0 else len(s)*len(mu_kernels)+1
    
    # constant basis
    phi[idx] = 1.0

    # to the next entry
    idx = idx+1

    # for each dimension in the state 
    for dim in range(len(s)):

        # for each mu in the kernel
        for mu in mu_kernels:

            phi[idx] = math.exp(-0.5 * (s[dim] - mu) **2)
            
            # to the next entry
            idx = idx+1

    return phi
