
# coding: utf-8

## 18-889e: Home work 1 Solution

# Note for Venkat: If you can open this, you have successfully managed to install ipython and the ipython notebook. Basically this forms a neat way for me to view the progress of my ideas so I tend to use it for homeworks. We can switch to scripts if you prefer that instead as well. Also, I've imported the 'numpy' package into python. The syntax of using that is close to Matlab so I thought it would make things more comfortable for you. 
# The whole program is divided into cells that must be run in the order they appear. Basically you have to keep hitting 'Shift+Enter' to run the whole program. You can also do that on this cell.

# In[54]:

# Solution to the first homework of Real life reinforcement learning
import csv
import numpy as np
from sklearn import linear_model
from sklearn import neighbors

# First read in all the data from the file.
with open('generated_episodes_3000.csv') as csv_file:
    data = np.array(list(csv.reader(csv_file))[1:])

# At this point all the data has been loaded into 'data'. I've ignored the first row because those are just labels that are 
# useless to us.


# In[55]:

# Uncomment the next statement if you want to see what a row of the data looks like.
# data[0]
# data[0].shape
#print data[0]


# In[85]:

# Fitted V/Q Iteration
# Helper functions

def create_event_iterator(data):
    """
        Function that returns the next (s, a, r, s') pair from the input data one by one every time you call it.
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
                a = 1.0 if datum[9:10]=='true' else 0.0
                r = np.asscalar(datum[10:11].astype(np.float))
                s_prime = datum[11:].astype(np.float)                
                yield s, a, r, s_prime
            # There's a problem if a data field is 'NA' - Not entirely sure what do in that case so for now I'm just ignoring
            # those data points
            except ValueError:
                pass                
            curr_state += 1

def compute_Q(weights, s, a):
    """
        Simple helper to compute the Q value given an input weight vector and the (s, a) pair
    """
    # Multiply the weights with the array of (s, a)
    return np.dot(np.append(s, a), weights)


# In[89]:

# Initialize the weights. Do one iteration to get things started
event_iterator = create_event_iterator(data)
Xs = []
ys = []
for s, a, r, s_prime in event_iterator:
    y = r
    Xs.append(np.append(s, a))
    ys.append(y)


# In[90]:

# And now we'll have to do the following in a loop - Currently this is one iteration of proper FVI
fn = linear_model.Lasso(alpha = 0.1)

# Uncomment for K-NN
#n_neighbors = 5
#fn = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")

fn.fit(Xs, ys)
event_iterator = create_event_iterator(data)
Xs = []
ys = []
for s, a, r, s_prime in event_iterator:
    y = r + max(fn.predict(np.append(s_prime, 0.0)), fn.predict(np.append(s_prime, 1.0)))
    Xs.append(np.append(s, a))
    ys.append(y[0])
fn.fit(Xs, ys)


# TODO:
# 
# -> Figure out- How many iterations do we have to do that last cell for? When do we say we've converged
# 
# -> What to do about the 'NA' data - Currently ignoring it but thats a very temporary solution
# 
# -> Try other function approximators

# In[91]:




# In[ ]:



