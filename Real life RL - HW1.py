
# coding: utf-8

## 18-889e: Home work 1 Solution

# Note for Venkat: If you can open this, you have successfully managed to install ipython and the ipython notebook. Basically this forms a neat way for me to view the progress of my ideas so I tend to use it for homeworks. We can switch to scripts if you prefer that instead as well. Also, I've imported the 'numpy' package into python. The syntax of using that is close to Matlab so I thought it would make things more comfortable for you. 
# The whole program is divided into cells that must be run in the order they appear. Basically you have to keep hitting 'Shift+Enter' to run the whole program. You can also do that on this cell.

# In[114]:

# Solution to the first homework of Real life reinforcement learning
import csv
import math
import numpy as np
from sklearn import linear_model
from sklearn import neighbors

# First read in all the data from the file.
with open('generated_episodes_3000.csv') as csv_file:
    data = np.array(list(csv.reader(csv_file))[1:])

# At this point all the data has been loaded into 'data'. I've ignored the first row because those are just labels that are 
# useless to us.


# In[115]:

# Uncomment the next statement if you want to see what a row of the data looks like.
# data[0]
# data[0].shape
#print data[0]


# In[116]:

from sklearn import preprocessing
def get_known_states(data):
    """
        Returns just the states that we know, i.e. states without the 'NA' in the data fields.
    """
    event_length = 9 + 9 + 2
    state_length = 9 + 2
    num_states = 24
    for episode in data:
        # Start at the beginning and keep looking at a net length of len(s) + len(a) + len(r) + len(s') points
        # Each time, we increment our start position by s+a+r = 11 points
        curr_state = 0
        while curr_state < num_states:
            start_idx = curr_state * state_length
            end_idx = start_idx + event_length
            datum = episode[start_idx:end_idx]
            try:
                s = datum[:9].astype(np.float)
                yield s
            # There's a problem if a data field is 'NA' - Not entirely sure what do in that case so for now I'm just ignoring
            # those data points
            except ValueError:
                pass    
            curr_state += 1
            
def generate_sars(data):
    """
        Function that returns the next (s, a, r, s') pair from the input data one by one every time you call it. Requires a 
        scaler to have been computed so that we can approximate the vaues for the 'NA' pairs in the data.
    """
    # Compute the known states and then compute a 'scaler' which stores the means and variances that will be used for standardization.
    known_states = [state for state in get_known_states(data)]
    scaler = preprocessing.StandardScaler().fit(known_states)
    event_length = 9 + 9 + 2
    state_length = 9 + 2
    num_states = 24
    sars = []
    for episode in data:
        # Start at the beginning and keep looking at a net length of len(s) + len(a) + len(r) + len(s') points
        # Each time, we increment our start position by s+a+r = 11 points
        curr_state = 0
        while curr_state < num_states:
            start_idx = curr_state * state_length
            end_idx = start_idx + event_length
            datum = episode[start_idx:end_idx]
            # If its normal data without 'NA', proceed as before except we 'scale' the values to mean-0 and variance-1
            a = 1.0 if datum[9:10]=='true' else 0.0
            r = np.asscalar(datum[10:11].astype(np.float))
            try:
                s = datum[:9].astype(np.float)
                scaler.transform(s)
                s_prime = datum[11:].astype(np.float)       
                scaler.transform(s_prime)
                sars.append([s, a, r, s_prime])
            # IF there was a value error it means there was a 'NA' field somewhere. 
            except ValueError:
                # ONLY S AND S' have these 'NA' fields (I've confirmed). Therefore we go through them and replace any
                # fields that have 'NA' with the mean of the corresponding feature, and then apply the scaler.
                s = np.array([elem if elem!='NA' else scaler.mean_[i].astype(np.float) for i, elem in enumerate(datum[:9])]).astype(np.float)
                scaler.transform(s)
                s_prime = np.array([elem if elem!='NA' else scaler.mean_[i].astype(np.float) for i, elem in enumerate(datum[:9])]).astype(np.float)
                scaler.transform(s_prime).astype(np.float)
                sars.append([s, a, r, s_prime])
            curr_state += 1
    return sars


# In[121]:

def FVI(fn, sars, gamma = 0.999, tol = 0.1):
    """
        Does fixed value iteration using the input fn approximator on the input data (expected to be in SARS format)
    """
    # From blackboard calculation
    Br = max([r for s, a,r, s_prime in sars])
    n_iters = math.log(((tol*(1-gamma**2))/Br) - gamma)
    print n_iters
    # Initialize the weights. Do one iteration to get things started
    Xs = []
    ys = []
    for s, a, r, s_prime in sars:
        y = r
        Xs.append(np.append(s, a))
        ys.append(y)
    fn.fit(Xs, ys)
    for i in range(iters):
        Xs = []
        ys = []
        for s, a, r, s_prime in sars:
            y = r + gamma * max(fn.predict(np.append(s_prime, 0.0)), fn.predict(np.append(s_prime, 1.0)))
            Xs.append(np.append(s, a))
            ys.append(y[0])
        fn.fit(Xs, ys)
    return fn


#### With that the main loop of our FVI code is pretty simply defined as below:

# In[122]:

# A key difference here is that we're just precomputing and storing all the s, a, r, s_prime pairs ahead of time. No generator
# functions here.
sars = generate_sars(data)

# And now we'll have to do the following in a loop - Currently this is one iteration of proper FVI
fn = linear_model.Lasso(alpha = 0.1)

# Uncomment for K-NN
#n_neighbors = 5
#fn = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")

FVI(fn, sars)


# TODO:
# 
# -> Figure out- How many iterations do we have to do that last cell for? When do we say we've converged
# 
# -> OMP-TD for feature selection + possible feature set expansion
# 
# -> LSPI (using LSTDQ already implemented)

# In[109]:




# In[ ]:



