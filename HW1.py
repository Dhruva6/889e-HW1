### Main file for HW1 of 15-889e: Real life reinforcement Learning ###
from optparse import OptionParser

import LSPI
import FVI
import OMP
import numpy as np
import csv

from sklearn import linear_model
from sklearn import neighbors
from Util import *

#
# command line options
#
parser = OptionParser()

# add the options
parser.add_option("-m", action="store", type="string", dest="model", default="lspi", help="Model to use: lspi or fvi [default=lspi]")
parser.add_option("-v", action="store_true", dest="crossValidateGamma", default=False, help="Run Cross Validation on gamma[default=False]")
parser.add_option("-w", action="store_true", dest="writeTestData", help="write test data[default=False]", default=False)
parser.add_option("-t", action="store_true", dest="testData", help="Test on given data[default=False]", default=False)

parser.add_option("-f", action="store", type="string", dest="trainFile", help="CSV Training data file name[default=generated_episodes_3000.csv]", default="generated_episodes_3000.csv")
parser.add_option("-p", action="store", type="string", dest="paramsFile", help="File with parameters from training[default=params.pk1]", default="params.pk1")
parser.add_option("-s", action="store", type="string", dest="testFile", help="CSV Test data file name[default=testData.csv]", default="testData.csv")
parser.add_option("-o", action="store_true", dest="OMP", help="Run feature selection with OMP-TD[default=False]", default=False)

#
# FVI options
#
# add the option for which function approximator to use
parser.add_option("--fn", action="store", type="string", dest="fn", default="lstsq", help="Function approximator to use: knn or lstsq [default=lstsq]")
parser.add_option("--nn", action="store", type="int", dest="nn", default="5", help="The number of neighbours for K-NN [default=5]")

# parse the options 
(options, args) = parser.parse_args()

import sys

if options.model=="lspi":
    model = LSPI.LSPI
else:
    model = FVI.FVI

if options.fn=="lstsq": 
    fn = linear_model.LinearRegression()
else:
    print "Using KNN for FVI"
    n_neighbours= options.nn
    fn = neighbors.KNeighborsRegressor(n_neighbours, weights="distance")

# OMPTD - hacked in.  This should come from calling the OMP-TD method
OMPTDFeatS = [0, 1, 4, 6, 7]

# if configured to NOT test
if options.testData == False:
    # prompt
    print "Reading file: " + options.trainFile

    # First read in all the data from the file.
    with open(options.trainFile) as csv_file:
        data = np.array(list(csv.reader(csv_file))[1:])

    # SARSA samples 
    sarsa, numFeat, kernelMu = generateSARSASamples(data)

    # Update SARS
    sars = sarsa[:,0:-1]

    if options.OMP:
        OMPTDFeatS =  OMP.OMP_TD(sars, 10000, 0.95)
        
        #  features to pull out from s'
        OMPTDFeatSPrime = [x+numFeat+2 for x in OMPTDFeatS] 
    
        # create a boolean mask
        mask = np.zeros((numFeat*2)+3, dtype=bool)
        mask[OMPTDFeatS] = True
        mask[numFeat] = True
        mask[numFeat+1] = True
        mask[OMPTDFeatSPrime] = True
        mask[-1] = True
        
        print mask
    
        # mask out the features that are not relelvant
        sarsa = sarsa[:, mask]
        sars = sarsa[:, 0:-1]

        # Modify numFeat 
        numFeat = len(OMPTDFeatS)

    # should we perform cross validation on gamma?
    if options.model=="lspi":
        gamma = np.linspace(0.1, 1.0, 20, False)
    else:
        gamma = np.linspace(0.8, 1.0, 10, False)

    # cross-validate if requested
    if options.crossValidateGamma == True:
        #  the initial policy executed at s'
        current_pi = np.reshape(sarsa[:,4], (len(sars),1))
        CrossValidate(model, options.model, numFeat, gamma, sars, kernelMu, sarsa=sarsa, current_pi=current_pi, fn=fn)
        
    else: # use gamma that was picked using the cross validation

        # gamma (from cross validation)
        gamma = 0.955

        if options.model == "lspi":
            # LSPI
            #  the initial policy executed at s'
            current_pi = sarsa[:,-1]
            current_pi, w_pi, current_value = model(sars, current_pi, numFeat, gamma, kernelMu)
        else:
            print "Running FVI One *ALL* the training data with gamma {0:.3f}".format(gamma)
            w_pi = (model(fn, sars, numFeat)).coef_
            current_pi = []
            current_value = []
            current_pi, current_value = EvaluatePolicy(sars[:, 0:numFeat], w_pi, numFeat, kernelMu)
            
        # console log
        print "Saving gamma and weights to file: " + options.paramsFile
        print "num samples {}, num true {}".format(len(sars), sum(current_pi))

        # dump the weight and the gamma to disk
        paramsOut = open(options.paramsFile, 'wb')
        pickle.dump(gamma, paramsOut, -1)
        pickle.dump(w_pi, paramsOut, -1)
        paramsOut.close()

        if options.writeTestData:
            # generate a random set of rows to write out
            testRows = random.sample(range(len(sars)), int(0.8*len(sars)))
            
            # save a sample of test data
            writer = csv.writer(open(options.testFile, 'w'))
            for row in testRows:
                writer.writerow(sars[row,0:numFeat])
else :

    print "Loading gamma and w_pi from file: " + options.paramsFile
    
    # load the gamma and the weights from paramsFile
    paramsIn = open(options.paramsFile, 'rb')
    gamma = pickle.load(paramsIn)
    w_pi = pickle.load(paramsIn)
    paramsIn.close()

    # load the file with test data
    print "Loading test data from file: " + options.testFile
    with open(options.testFile) as csv_file:
        data = np.array(list(csv.reader(csv_file))[1:])

    # generate the test states from data
    test_s, numFeat, kernelMu = generateSARSASamples(data, True)
    
    # evaluate the policy
    policy, value = EvaluatePolicy(test_s, w_pi, numFeat, kernelMu)

    print "Num states: {}, num true: {}, value: {}".format(len(test_s), sum(policy), np.mean(value))

