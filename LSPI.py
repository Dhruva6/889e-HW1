
# coding: utf-8

# Solution to the first homework of Real life reinforcement learning
import math
import numpy as np
from sklearn import linear_model

# the Least Squares Temporal Difference Q learning algorithm
# Input: sars         - the nx4 array of (s,a,r,s') elements
#        current_pi   - the current policy at s' (nx1)
#        gamma        - the discount factor (defaults to 0.9)
#
# Output: w_pi - the estimated weight for the linear model
def LSTDQ(sars, current_pi, numFeat, kernelMu, gamma = 0.9, useRBFKernel = False):

    # at present, our basis is R^d - where d is 9 (features) + 1 (action)
    k = 2*(numFeat)

    # if configured to use the RBF Kernel
    if useRBFKernel == True:
        phi = computePhiRBF(kernelMu, numFeat, sars[0,0:numFeat], -1)
        k = len(phi)
        
    # to avoid singularities, we start off A with a small delta along the diagonal
    delta = 1e-09

    # the "A" Matrix (kxk) 
    A = delta*np.eye(k)
    
    # the b vector
    b = np.zeros([k, 1], dtype=float)

    # indices for various operations
    sIdx = 0
    aIdx = sIdx + numFeat
    rIdx = aIdx + 1
    spIdx = rIdx + 1
    spIdxEnd = spIdx + k

    # iterate through and build A and b
    for idx in range(len(sars)):

        # allocate
        phi_s = np.zeros((k,1))
        phi_s_prime = np.zeros((k,1))
                
        if useRBFKernel == True:
            phi_s = computePhiRBF(kernelMu, numFeat, sars[idx,0:aIdx], sars[idx,aIdx])
            phi_s_prime = computePhiRBF(kernelMu, numFeat, sars[idx,spIdx:spIdxEnd],current_pi[idx] )
        else:

            # where to place our features for phi
            offset = 0 if sars[idx,aIdx] == -1.0 else numFeat
            phi_s[offset:offset+numFeat,0] = sars[idx,0:aIdx]

            offset = 0 if current_pi[idx] == -1.0 else numFeat
            phi_s_prime[offset:offset+numFeat,0] = sars[idx,spIdx:spIdxEnd]
                        
        # Update A - here, we add to A, a Rank 1 matrix formed by
        # the vectors phi(s) and (phi(s) - gamma*phi(s'))
        A = A + np.outer(phi_s, phi_s - gamma*phi_s_prime)

        # update B - we add to B, the feature vector scaled by the reward
        b = b + sars[idx,rIdx:rIdx+1]*phi_s

    # compute the weights for the policy pi - solve the system A*w_pi = b
    w_pi,_,_,_ = np.linalg.lstsq(A, b)

    return w_pi

# Policy Improvement
def ImprovePolicy(s, w_pi, numFeat, kernelMu, useRBFKernel = False):

    # the new policy
    policy = np.zeros((len(s),1))

    # the value of the improved policy
    value = np.zeros((len(s),1))

    # allocate phi and phi_s_prime
    phi_s = np.zeros(2*numFeat)
    phi_s_prime = np.zeros(2*numFeat)
    
    # iterate through every state, 
    for idx in range(len(s)):
        
        # State-Action value function for actions -1.0 and 1.0
        if useRBFKernel == True:
            phi_s = computePhiRBF(kernelMu, numFeat, s[idx,0:numFeat], -1).T
            phi_s_prime = computePhiRBF(kernelMu, numFeat, s[idx,0:numFeat], 1).T
        else:
            phi_s[0:numFeat] = s[idx,0:numFeat]
            phi_s_prime[numFeat:2*numFeat] = s[idx,0:numFeat]

        # compute q0 and q1
        q0 = np.dot(phi_s, w_pi)
        q1 = np.dot(phi_s_prime, w_pi)

        # update the policy as argmax(action = {-1.0, 1.0}) Q^
        policy[idx] = 1.0 if q1 > q0 else -1.0

        # update the value
        value[idx] = max(q0, q1)
        
    return (policy, value)

def LSPI(sars, current_pi, numFeat, gamma, kernelMu, useRBFKernel = False):
    # the maximum number of iterations to run
    maxIter = 5

    # the current loop counter
    iter = 1

    # epsilon tolerance to terminate the policy improvement
    eps = 1e-02;

    # the initial weight vector
    if useRBFKernel == True:        
        phi = computePhiRBF(kernelMu, numFeat, sars[0,0:numFeat], -1.0)
        w_pi = np.zeros((len(phi),1))
    else:
        w_pi = np.zeros((2*numFeat,1))

    # the current value for all state-action pairs 
    current_value = np.zeros((len(sars),1))

    # loop
    while iter < maxIter:

        if 0 == iter%2:
            print "Now at policy iteration #{}".format(iter)
            
        # Estimate the State-Action VF Approximation using LSTDQ
        new_w_pi = LSTDQ(sars, current_pi, numFeat, kernelMu, gamma, useRBFKernel)
        
        # improve the policy
        new_pi, new_value = ImprovePolicy(sars[:,0:numFeat], new_w_pi, numFeat, kernelMu, useRBFKernel)

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

def computePhiRBF(kernelMu, numFeat, s, a):

    # the phi
    phi = np.zeros((2*(len(s) * kernelMu.shape[1] + 1), 1))

    # start counter
    idx = 0 if a == -1.0 else len(s)*kernelMu.shape[1]+1

    # constant basis
    phi[idx] = 1.0

    # to the next entry
    idx = idx+1

    # for each dimension in the state 
    for dim in range(numFeat):

        # for each mu in the kernel
        for mu in kernelMu[dim]:

            phi[idx] = math.exp(-0.5 * (s[dim] - mu) **2)
            
            # to the next entry
            idx = idx+1
            
    return phi
