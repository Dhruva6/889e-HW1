### Main file for HW1 of 15-889e: Real life reinforcement Learning ###
from optparse import OptionParser

from LSPI import *
from FVI import *
from Util import *
from OMP import *

#
# command line options
#
parser = OptionParser()

# add the options
parser.add_option("-m", action="store", type="string", dest="model", default="lspi", help="Model to use: lspi or fvi [default=lspi]")
parser.add_option("-v", action="store_true", dest="crossValidateGamma", default=False, help="Run Cross Validation on gamma[default=False]")
#parser.add_option("-l", action="store_true", dest="loadWeights", help="Load weights from training[default=False]", default=False)
parser.add_option("-t", action="store_true", dest="testData", help="Test on given data[default=False]", default=False)

parser.add_option("-f", action="store", type="string", dest="trainFile", help="CSV Training data file name[default=generated_episodes_3000.csv]", default="generated_episodes_3000.csv")
parser.add_option("-p", action="store", type="string", dest="paramsFile", help="File with parameters from training[default=params.pk1]", default="params.pk1")
parser.add_option("-s", action="store", type="string", dest="testFile", help="CSV Test data file name[default=testData.csv]", default="testData.csv")

#
# FVI options
#
# add the option for which function approximator to use
parser.add_option("--fn", action="store", type="string", dest="fn", default="lstsq", help="Function approximator to use: knn or lstsq [default=lstsq]")
parser.add_option("--nn", action="store", type="int", dest="nn", default="5", help="The number of neighbours for K-NN [default=5]")

# parse the options 
(options, args) = parser.parse_args()

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
#    elem_list = OMP_TD(sars, 6)
    if options.fn=="lstsq": 
        fn = linear_model.LinearRegression()
    else:
        n_neighbours= options.nn
        fn = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")

    # should we perform cross validation on gamma?
    if options.model=="lspi":
        gamma = np.linspace(0.95, 1.0, 20, False)
    else:
        gamma = np.linspace(0.5, 1.0, 10, False)

    # cross-validate if requested
    if options.crossValidateGamma == True:
        #  the initial policy executed at s'
        current_pi = np.reshape(sarsa[:,4], (len(sars),1))
        if options.model=="lspi":
            CrossValidate(LSPI, "lspi", gamma, sars, sarsa=sarsa, current_pi=current_pi)
        else:
            CrossValidate(FVI, "fvi", gamma, sars, fn=fn)
        
    else: # use gamma that was picked using the cross validation

        # gamma (from cross validation)
        gamma = 0.9975

        if options.model == "lspi":
            # LSPI
            # console log
            #  the initial policy executed at s'
            current_pi = np.reshape(sarsa[:,4], (len(sars),1))
            current_pi, w_pi, current_value = LSPI(sars, current_pi, gamma)
        else:
            print "Running FVI One *ALL* the training data with gamma {0:.3f}".format(gamma)
            w_pi = (FVI(fn, sars)).coef_
            
            #fvi_curr_pi = [EvaluatePolicy(s, w_pi)[0] for s in sarsa[:, 0]]

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
    policy, value = EvaluatePolicy(test_s, w_pi)
