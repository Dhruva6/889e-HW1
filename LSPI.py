
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
def LSTDQ(sars, current_pi, gamma = 0.9):

    # at present, our basis is R^d - where d is 9 (features) + 1 (action)
    k = 10

    # to avoid singularities, we start off A with a small delta along the diagonal
    delta = 1e-09

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
        #A = A + np.outer(phi_s, phi_s - gamma*phi_s_prime)
        A = A + phi_s * (phi_s - gamma * phi_s_prime).T
        
        # update B - we add to B, the feature vector scaled by the reward
        b = b + sars[idx,2]*phi_s

    # compute the weights for the policy pi - solve the system A*w_pi = b
    w_pi,_,_,_ = np.linalg.lstsq(A, b)

    return w_pi

# Policy Improvement
def ImprovePolicy(s, w_pi):

    # the new policy
    policy = np.zeros((len(s),1))

    # the value of the improved policy
    value = np.zeros((len(s),1)) 
    
    # iterate through every state, 
    for idx in range(0, len(s)):
        # the state-action value for action 0.0
        q0 = np.dot(np.append(s[idx],0.0), w_pi)

        # the state-action value for action 1.0
        q1 = np.dot(np.append(s[idx],1.0), w_pi)

        # update the policy as argmax(action = {0.0, 1.0}) Q^
        policy[idx] = 1.0 if q1 > q0 else 0.0

        # update the value
        value[idx] = max(q0, q1)
        
    return (policy, value)

# Policy Evaluation at the given states
def EvaluatePolicy(s, w_pi):
  
    # the value of the improved policy
    value = np.zeros((len(s),1))

    # the new policy
    policy = [False] * len(s)

    # iterate through every state, 
    for idx in range(0, len(s)):

        # the state-action value for action 0.0
        q0 = np.dot(np.append(s[idx,0],0.0), w_pi)

        # the state-action value for action 1.0
        q1 = np.dot(np.append(s[idx,0],1.0), w_pi)

        # update the value
        value[idx] = max(q0, q1)

        # update the policy
        policy[idx] = True if q1 > q0 else False
        
    return (policy, value)

def LSPI(sars, current_pi, gamma):
    # the maximum number of iterations to run
    maxIter = 50

    # the current loop counter
    iter = 1

    # epsilon tolerance to terminate the policy improvement
    eps = 1e-02;

    # the initial weight vector
    w_pi = np.zeros((10,1))

    # the current value for all state-action pairs 
    current_value = np.zeros((len(sars),1))

    # loop
    while iter < maxIter:

        if 0 == iter%2:
            print "Now at policy iteration #{}".format(iter)
            
        # Estimate the State-Action VF Approximation using LSTDQ
        new_w_pi = LSTDQ(sars, current_pi, gamma)

        # improve the policy
        new_pi, new_value = ImprovePolicy(sars[:,0], new_w_pi)

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

    


