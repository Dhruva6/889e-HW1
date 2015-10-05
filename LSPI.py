
# coding: utf-8

# Solution to the first homework of Real life reinforcement learning
import csv
import math
import random
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

    # the value of the improved policy
    value = np.zeros((len(s),1)) 
    
    # iterate through every state, 
    for idx in range(0, len(s)):
        # the state-action value for action 0.0
        q0 = np.dot(np.append(s[idx],0.0), w_pi)

        # the state-action value for action 1.0
        q1 = np.dot(np.append(s[idx],1.0), w_pi)

        # update the policy as argmax(action = {0.0, 1.0}) Q^
        policy[idx] = 1.0 if q0 < q1 else 0.0

        # update the value
        value[idx] = max(q0, q1)
        
        # to the next state
        idx = idx+1    

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
        policy[idx] = True if q0 < q1 else False
        
        # to the next state
        idx = idx+1    

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

    # loop
    while iter < maxIter:

        if 0 == iter%2:
            print "Now at policy iteration #{}".format(iter)
            
        # Estimate the State-Action VF Approximation using LSTDQ
        new_w_pi = LSTDQ(sars, current_pi, gamma)

        # improve the policy
        new_pi, new_value = ImprovePolicy(sars[:,0], w_pi)

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

# prompt
print "Reading file: generated_episodes_3000.csv"

# First read in all the data from the file.
with open('generated_episodes_3000.csv') as csv_file:
    data = np.array(list(csv.reader(csv_file))[1:])

# Generate the (s,a,r,s',a') tuple from data
sarsa = np.array(generate_sarsa(data))

# pull out the (s,a,r,s) tuple from sarsa
sars = sarsa[:,0:4]

# should we perform cross validation on gamma?
crossValidateGamma = True
#gamma = np.linspace(0.0, 1.0, 50, True)
gamma = np.linspace(0.95, 1.0, 20, False)

# cross-validate if requested
if crossValidateGamma:

    # the number of times to run the cross validation for a given gamma
    maxCVTimes  = 5

    # the number of folds
    numFolds  = 10

    # number of test elements
    numTestElements = len(sars)/numFolds

    # number of training elements
    numTrainElements = len(sars) - numTestElements

    print "Train Elements {}, Test Elements {}".format(numTrainElements, numTestElements)

    # the mean values of each of the policy
    mean_policy_values = np.zeros((len(gamma),1))

    # index for g
    gIdx = 0
    
    # iterate through all the elements of gamma
    for g in gamma:

        print "Cross validating for gamma: {0:.3f}".format(g)
    
        # the current loop counter
        cvTimes = 0

        # iterate 
        while cvTimes < maxCVTimes:

            # get the training set rows
            trainRows = random.sample(range(0,len(sars)), numTrainElements)
        
            # the test set rows
            testRows = list(set(range(0,len(sars))) - set(trainRows))
        
            #  the initial policy executed at s'
            current_pi = np.reshape(sarsa[:,4], (len(sars),1))

            # LSPI
            _, w_pi,_ = LSPI(sars[trainRows,:], current_pi, g)
        
            # evaluate the policy at sars[testRows,:]
            _,values = EvaluatePolicy(sars[testRows,0:1], w_pi)

            # update the mean_policy_values for the current gamma
            mean_policy_values[gIdx] = mean_policy_values[gIdx] + np.mean(values)
                
            # tick over the counter
            cvTimes = cvTimes + 1

        # average over all the cross-validation times
        mean_policy_values[gIdx,0] = mean_policy_values[gIdx,0]/float(maxCVTimes)
        
        # console log
        print "Mean policy value for test set: {0:.2f}".format(mean_policy_values[gIdx,0])
        
        # tick over gIdx
        gIdx = gIdx + 1

    # write the gamma values to the csv file
    with open("LSPI_gamma_CV.csv", "w") as out_file:
        out_file.write("# Gamma, Mean Policy Value\n")
        for i in range(len(gamma)):
            out_string = "{0:.5f},{1:.5f}\n".format(gamma[i],mean_policy_values[i,0])
            out_file.write(out_string)

else: # just pick the gamma with the largest return

    # gamma (from cross validation)
    gamma = 0.9975

    #  the initial policy executed at s'
    current_pi = np.reshape(sarsa[:,4], (len(sars),1))

    # LSPI
    current_pi, w_pi, current_value = LSPI(sars, current_pi, gamma)

    print "gamma: {0:.5f}, Mean Policy Value: {1:.5f}".format(gamma, np.mean(current_value))
