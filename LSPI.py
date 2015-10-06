
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
def LSTDQ(sars, current_pi, gamma = 0.9, useRBFKernel = False):

    # at present, our basis is R^d - where d is 9 (features) + 1 (action)
    k = 10

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

# Policy Evaluation at the given states
def EvaluatePolicy(s, w_pi, useRBFKernel = False):
  
    # the value of the improved policy
    value = np.zeros((len(s),1))

    # the new policy
    policy = [False] * len(s)

    # iterate through every state, 
    for idx in range(len(s)):

        # State-Action value function for actions 0.0 and 1.0
        if useRBFKernel == True:
            q0 = np.dot(computePhiRBF(s[idx], 0.0).T, w_pi)
            q1 = np.dot(computePhiRBF(s[idx], 1.0).T, w_pi)
        else:
            q0 = np.dot(np.append(s[idx],0.0), w_pi)
            q1 = np.dot(np.append(s[idx],1.0), w_pi)

        # update the value
        value[idx] = max(q0, q1)

        # update the policy
        policy[idx] = True if q1 > q0 else False
        
    return (policy, value)

def LSPI(sars, current_pi, gamma, useRBFKernel = False):
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
        w_pi = np.zeros((10,1))

    # the current value for all state-action pairs 
    current_value = np.zeros((len(sars),1))

    # loop
    while iter < maxIter:

        if 0 == iter%2:
            print "Now at policy iteration #{}".format(iter)
            
        # Estimate the State-Action VF Approximation using LSTDQ
        new_w_pi = LSTDQ(sars, current_pi, gamma, useRBFKernel)

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
        
#
# command line options
#
parser = OptionParser()

# add the options
parser.add_option("-v", action="store_true", dest="crossValidateGamma", default=False, help="Run Cross Validation on gamma[default=False]")
#parser.add_option("-l", action="store_true", dest="loadWeights", help="Load weights from training[default=False]", default=False)
parser.add_option("-t", action="store_true", dest="testData", help="Test on given data[default=False]", default=False)
parser.add_option("-k", action="store_true", dest="useRBFKernel", help="Use RBF Kernel[default=False]", default=False)

parser.add_option("-f", action="store", type="string", dest="trainFile", help="CSV Training data file name[default=generated_episodes_3000.csv]", default="generated_episodes_3000.csv")
parser.add_option("-p", action="store", type="string", dest="paramsFile", help="File with parameters from training[default=params.pk1]", default="params.pk1")
parser.add_option("-s", action="store", type="string", dest="testFile", help="CSV Test data file name[default=testData.csv]", default="testData.csv")

# parse the options 
(options, args) = parser.parse_args()

#
#
#
# if configured to NOT test
if options.testData == False:
    
    # prompt
    print "Reading file: " + options.trainFile

    # First read in all the data from the file.
    with open(options.trainFile) as csv_file:
        data = np.array(list(csv.reader(csv_file))[1:])

    # Generate the (s,a,r,s',a') tuple from data
    sarsa = np.array(generate_sarsa(data))

    # pull out the (s,a,r,s) tuple from sarsa
    sars = sarsa[:,0:4]

    # should we perform cross validation on gamma?
    gamma = np.linspace(0.95, 1.0, 20, False)

    # cross-validate if requested
    if options.crossValidateGamma == True:

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
        
        # iterate through all the elements of gamma
        for gIdx, g in enumerate(gamma):

            print "Cross validating for gamma: {0:.3f}".format(g)
            
            # the current loop counter
            cvTimes = 0

            # iterate 
            while cvTimes < maxCVTimes:

                print "now performing CV # {}".format(cvTimes+1)
                
                # get the training set rows
                trainRows = random.sample(range(0,len(sars)), numTrainElements)
                
                # the test set rows
                testRows = list(set(range(0,len(sars))) - set(trainRows))
                
                #  the initial policy executed at s'
                current_pi = np.reshape(sarsa[:,4], (len(sars),1))

                # LSPI
                _, w_pi,_ = LSPI(sars[trainRows,:], current_pi, g, options.useRBFKernel)

                # evaluate the policy at sars[testRows,:]
                _,values = EvaluatePolicy(sars[testRows,0:1], w_pi, options.useRBFKernel)

                # update the mean_policy_values for the current gamma
                mean_policy_values[gIdx] = mean_policy_values[gIdx] + np.mean(values)
                
                # tick over the counter
                cvTimes = cvTimes + 1

            # average over all the cross-validation times
            mean_policy_values[gIdx,0] = mean_policy_values[gIdx,0]/float(maxCVTimes)
                
            # console log
            print "Mean policy value for test set: {0:.2f}".format(mean_policy_values[gIdx,0])
                

        # write the gamma values to the csv file
        with open("LSPI_gamma_CV.csv", "w") as out_file:
            out_file.write("# Gamma, Mean Policy Value\n")
            for i in range(len(gamma)):
                out_string = "{0:.5f},{1:.5f}\n".format(gamma[i],mean_policy_values[i,0])
                out_file.write(out_string)
        
    else: # use gamma that was picked using the cross validation

        # gamma (from cross validation)
        gamma = 0.9975

        # console log
        print "Running LSPI on *ALL* the training data with gamma {0:.3f}".format(gamma)      
        #  the initial policy executed at s'
        current_pi = np.reshape(sarsa[:,4], (len(sars),1))

        # LSPI
        current_pi, w_pi, current_value = LSPI(sars, current_pi, gamma, options.useRBFKernel)

        # console log
        print "Saving gamma and weights to file: " + options.paramsFile

        # dump the weight and the gamma to disk
        paramsOut = open(options.paramsFile, 'wb')
        pickle.dump(gamma, paramsOut, -1)
        pickle.dump(w_pi, paramsOut, -1)
        paramsOut.close()

        # # save a sample of test data
        # writer = csv.writer(open(options.testFile, 'w'))
        # for row in range(0, len(sars)):
        #     writer.writerow(sars[row,0])
else :

    print "Loading gamma and w_pi from file: " + options.paramsFile
    
    # load the gamma and the weights from paramsFile
    paramsIn = open(options.paramsFile, 'rb')
    gamma = pickle.load(paramsIn)
    w_pi = pickle.load(paramsIn)
    paramsIn.close()
    
    # load the scaler
    scalerIn = open('Scaler.pk1', 'rb')
    scaler = pickle.load(scalerIn)
    scalerIn.close()
   
    # load the file with test data
    print "Loading test data from file: " + options.testFile
    with open(options.testFile) as csv_file:
        data = np.array(list(csv.reader(csv_file))[1:])

    # generate the test states from data
    test_s = np.array(generate_test_states(data, scaler))

    # evaluate the policy
    policy, value = EvaluatePolicy(test_s, w_pi, options.useRBFKernel)
