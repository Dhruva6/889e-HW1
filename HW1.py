### Main file for HW1 of 15-889e: Real life reinforcement Learning ###
from optparse import OptionParser

import LSPI
import FVI
import Util
import OMP
import numpy as np
import csv
from sklearn import linear_model
from sklearn import neighbors
import pickle

#
# command line options
#
parser = OptionParser()

# add the options
parser.add_option("-m", action="store", type="string", dest="model", default="lspi", help="Model to use: lspi or fvi [default=lspi]")
parser.add_option("-v", action="store_true", dest="crossValidateGamma", default=False, help="Run Cross Validation on gamma[default=False]")
#parser.add_option("-l", action="store_true", dest="loadWeights", help="Load weights from training[default=False]", default=False)
parser.add_option("-t", action="store_true", dest="testData", help="Test on given data[default=False]", default=False)
parser.add_option("-o", action="store_true", dest="OMP", help="Run OMP_TD[default=False", default=False)

parser.add_option("-f", action="store", type="string", dest="trainFile", help="CSV Training data file name[default=generated_episodes_3000.csv]", default="generated_episodes_3000.csv")
parser.add_option("-p", action="store", type="string", dest="paramsFile", help="File with parameters from training[default=params.pk1]", default="params.pk1")
parser.add_option("-s", action="store", type="string", dest="testFile", help="CSV Test data file name[default=testData.csv]", default="testData.csv")
parser.add_option("-k", action="store_true", dest="rbf", help="Use the RBFKernel", default=False)

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

# if configured to NOT test
if options.testData == False:
    # prompt
    print "Reading file: " + options.trainFile

    # First read in all the data from the file.
    with open(options.trainFile) as csv_file:
        data = np.array(list(csv.reader(csv_file))[1:])
    
    # Generate the (s,a,r,s',a') tuple from data
    sarsa = np.array(Util.generate_sarsa(data))

    # pull out the (s,a,r,s) tuple from sarsa
    sars = sarsa[:,0:4]
    
    # should we perform cross validation on gamma?
    if options.model=="lspi":
        gamma = np.linspace(0.95, 1.0, 20, False)
    else:
        gamma = np.linspace(0.8, 1.0, 10, False)

    # cross-validate if requested
    if options.crossValidateGamma == True:
        #  the initial policy executed at s'
        current_pi = np.reshape(sarsa[:,4], (len(sars),1))
        Util.CrossValidate(model, options.model, gamma, sars, sarsa=sarsa, current_pi=current_pi, fn=fn, useRBF=options.rbf)
        
    else: # use gamma that was picked using the cross validation

        # gamma (from cross validation)
        gamma = 0.9975
        k = 10
        if options.OMP:
            print "Running OMP-TD and extracting 5 features"
            elemList = OMP.OMP_TD(sars, 100000, gamma)
            new_sarsa = []
            for s, a, r, s_prime, a_prime in sarsa:
                new_s = s[0][elemList]
                new_s_prime = s_prime[0][elemList]
                new_sarsa.append([new_s, a, r, new_s_prime, a_prime])
            sarsa = np.array(new_sarsa)
            sars = sarsa[:, 0:4]
            k = 6

        if options.model == "lspi":
            # LSPI
            # console log
            #  the initial policy executed at s'
            current_pi = np.reshape(sarsa[:,4], (len(sars),1))
            current_pi, w_pi, current_value = model(sars, current_pi, gamma, useRBFKernel=options.rbf, k=k)
        else:
            print "Running FVI One *ALL* the training data with gamma {0:.3f}".format(gamma)
            w_pi = (model(fn, sars)).coef_
            
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
    test_s = np.array(Util.generate_test_states(data, scaler))

    # evaluate the policy
    policy, value = Util.EvaluatePolicy(test_s, w_pi)    
