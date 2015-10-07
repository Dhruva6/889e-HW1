# A collection of Utility functions - copied from Real\ life\ RL\ -\ HW1.py

# Import
import pickle
import numpy as np
from sklearn import preprocessing
import random
            
def getValidStates(data, testData = False):
    """
        Returns just the states that we know, i.e. states without the 'NA' in the data fields.
    """

    # the number of features describing the state
    numFeat = 9

    # the total number of fields associated with a state (numFeat + 1 reward + 1 action)
    sarLength = numFeat + 2
    
    # the total number of state, action, rewards per trajectory
    numSARPerEpisode = 24 if testData == False else 1

    for episode in data:
        # Start at the beginning and keep looking at a net length of len(s) + len(a) + len(r) + len(s') points
        # Each time, we increment our start position by s+a+r = 11 points
        sarCount = 0
        
        while sarCount < numSARPerEpisode:
            startIdx = sarCount * sarLength
            endIdx = startIdx + (2 * sarLength) if testData == False else startIdx+numFeat
            datum = episode[startIdx:endIdx]
            
            try:
                s = np.array(datum[:numFeat].astype(np.float))

                if testData == True:
                    yield s, numFeat
                else:
                    a = 1.0 if datum[9:10] == 'true' else -1.0
                    r = datum[10:11].astype(np.float)
                    s_p = np.array(datum[11:20].astype(np.float))
                    a_p = 1.0 if datum[20:21] == 'true' else -1.0

                    # prepare to pass the value                 
                    yield s, a, r, s_p, a_p, numFeat
            
            # we have a 'NA' in the current state or the next state
            except ValueError:

                if testData == True:
                    # we fill in the missing value as 0.0 (the mean value)
                    s = np.array([elem if elem!='NA' else 0.0 for i, elem in enumerate(datum[:9])]).astype(np.float)
                    yield s, numFeat
                else:
                    s = np.array([elem if elem!='NA' else 0.0 for i, elem in enumerate(datum[:9])]).astype(np.float)
                    s_p = np.array([elem if elem!='NA' else 0.0 for i, elem in enumerate(datum[:9])]).astype(np.float)
                    yield s, a, r, s_p, a_p, numFeat
            
            # to the next state, action, reward
            sarCount += 1
            

def generateSARSASamples(data, testData = False):

    # if parsing training data
    if testData == False:
    
        # state normalizer
        stateNormalizer = preprocessing.StandardScaler()

        # rewards normalizer
        rewardNormalizer = preprocessing.MinMaxScaler()

        # iterate through and aggregate the data
        aggState = []
        rewards = []
        currAction = []
        nextAction = []
        
        for s,a,r,s_p, a_p, numFeat in getValidStates(data):

            # aggregate s and s_p
            aggState.append(s)
            aggState.append(s_p)

            # append r to rewards 
            rewards.append(r)

            # append a to currAction
            currAction.append(a)

            # append a_p to nextAction
            nextAction.append(a) 
            
        #  normalize the aggregated states
        stateNormalizer.fit(aggState)
        aggState = stateNormalizer.transform(aggState)
        
        #
        # Serialize scaler
        #
        paramsOut = open("Scaler.pk1", 'wb')
        pickle.dump(stateNormalizer, paramsOut, -1)
        paramsOut.close()

        
        # (Min-Max) normalize the rewards 
        r = np.array(rewardNormalizer.fit_transform(rewards))
       # r = np.array(rewards)

        # allocate SARSA
        sarsa = np.zeros((len(r), 2*numFeat+3))

        #
        # populate SARSA
        #

        # populate the state related fields
        for i,idx in enumerate(range(0,len(aggState),2)):
            sarsa[i,0:numFeat] = aggState[idx]
            sarsa[i,numFeat+2:-1] = aggState[idx+1]

        # populate the rewards and actions
        sarsa[:,numFeat] = np.array(currAction)
        sarsa[:,numFeat+1:numFeat+2] = r
        sarsa[:,-1] = np.array(nextAction)

        #
        # here, we compute the kernels as well
        #
        offset = numFeat+2
        minS = sarsa[:,0:numFeat].min(axis=0).reshape((1,numFeat))
        minSp = sarsa[:,offset:offset+numFeat].min(axis=0).reshape((1,numFeat))
        minRange = np.vstack((minS, minSp)).min(axis=0).reshape((1,numFeat))
    
        maxS = sarsa[:,0:numFeat].max(axis=0).reshape((1,numFeat))
        maxSp = sarsa[:,offset:offset+numFeat].max(axis=0).reshape((1,numFeat))
        maxRange = np.vstack((maxS, maxSp)).max(axis=0).reshape((1,numFeat))

        # the location of the kernels will vary by the minRange and maxRange
        kernelRanges = np.vstack((minRange, maxRange)).reshape((numFeat,2))
        
        # the number of kernels per dimension
        numKernels = 3

        # the mean value for kernels
        kernelMu = np.zeros((numFeat, numKernels))

        # force the kernels to lie half way between min range and 0
        # at 0 and one half way between 0 and max range
        for idx in range(0,numFeat):
            kernelMu[idx, 0] = kernelRanges[idx,0]/2.0
            kernelMu[idx, 1] = 0.0
            kernelMu[idx, 2] = kernelRanges[idx,1]/2.0

        #
        # serialize the kernel means
        #

        paramsOut = open("RBFKernel.pk1", 'wb')
        pickle.dump(kernelMu, paramsOut, -1)
        paramsOut.close()
        
        
        return sarsa, numFeat, kernelMu
    else:

        #
        # De-Serialize scaler
        #
        scalerIn = open('Scaler.pk1', 'rb')
        stateNormalizer = pickle.load(scalerIn)
        scalerIn.close()

        #
        # De-serialize kernelMu
        #
        kernelIn = open('RBFKernel.pk1')
        kernelMu = pickle.load(kernelIn)
        kernelIn.close()

        states = []
        for s, numFeat in getValidStates(data, testData):
            states.append(s)

        testStates = np.array(stateNormalizer.transform(states))
        #testStates = np.array(states)
        return testStates, numFeat
    


# Policy Evaluation at the given states
def EvaluatePolicy(s, w_pi, numFeat, useRBFKernel = False):
  
    # the value of the improved policy
    value = np.zeros((len(s),1))

    # the new policy
    policy = [False] * len(s)

    # allocate phi and phi_s_prime
    phi_s = np.zeros(2*numFeat)
    phi_s_prime = np.zeros(2*numFeat)

    # iterate through every state, 
    for idx in range(len(s)):

        phi_s[0:numFeat] = s[idx,0:numFeat]
        phi_s_prime[numFeat:2*numFeat] = s[idx,0:numFeat]

        # State-Action value function for actions 0.0 and 1.0
        if useRBFKernel == True:
            assert False
        else:
            q0 = np.dot(phi_s, w_pi)
            q1 = np.dot(phi_s_prime, w_pi)

        # update the value
        value[idx] = max(q0, q1)

        # update the policy
        policy[idx] = True if q1 > q0 else False
        
    return (policy, value)

def CrossValidate(model, model_name, numFeat, gamma, sars, sarsa=[], current_pi=[], fn=None):
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
                
            # LSPI
            if model_name == "lspi":
                _, w_pi,_ = model(sars[trainRows,:], current_pi, numFeat, g)
            # FVI
            else:
                w_pi = (model(fn, sars[trainRows,:], numFeat, gamma=g)).coef_
                
            # evaluate the policy at sars[testRows,:]
            _,values = EvaluatePolicy(sars[testRows,0:numFeat], w_pi, numFeat)
                
            # update the mean_policy_values for the current gamma
            mean_policy_values[gIdx] = mean_policy_values[gIdx] + np.mean(values)
                
            # tick over the counter
            cvTimes = cvTimes + 1
        
        # average over all the cross-validation times
        mean_policy_values[gIdx,0] = mean_policy_values[gIdx,0]/float(maxCVTimes)
        
        # console log
        print "Mean policy value for test set: {0:.2f}".format(mean_policy_values[gIdx,0])

        # write the gamma values to the csv file
        with open(model_name+"_gamma_CV.csv", "w") as out_file:
            out_file.write("# Gamma, Mean Policy Value\n")
            for i in range(len(gamma)):
                out_string = "{0:.5f},{1:.5f}\n".format(gamma[i],mean_policy_values[i,0])
                out_file.write(out_string)
