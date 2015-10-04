
# coding: utf-8

# In[54]:

# Solution to the first homework of Real life reinforcement learning
import csv
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

# prompt
print "Reading file: generated_episodes_3000.csv"

# First read in all the data from the file.
with open('generated_episodes_3000.csv') as csv_file:
    data = np.array(list(csv.reader(csv_file))[1:])

# Uncomment the next statement if you want to see what a row of the data looks like.
#print data[0]

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

# Instantiation of the Ordinary Least Squares class
ols = linear_model.LinearRegression()

# at present, our basis is N^d - where d is 9
k = 9

# the "A" Matrix (kxk)
A = np.zeros([k, k], dtype=float)

# the b vector
b = np.zeros([k, 1], dtype=float)

# the discount factor gamma
gamma = 0.9

# generate the object that will produces us the (s,a,r,s') tuple
event_iterator = create_event_iterator(data)

print "Building the matrix A and the vector b"

# iterate through the tuple
for s, a, r, s_prime in event_iterator:
    
    # In future implementations, we can have a different basis such as RBF or a polynomial for our features
    # In the current implementation, we take the given features as the basis 
    phi_s = np.reshape(s, (k,1))
    phi_s_prime = np.reshape(s_prime, (k,1))
    
    # Update A - here, we add to A, a Rank 1 matrix formed by phi(s) and (phi(s) - gamma*phi(s'))
    A = A + np.outer(phi_s, phi_s - gamma*phi_s_prime)

    # update B - we add to B, the feature vector scaled by 
    b = b + r*phi_s

print "Solving for the weights"

# compute the weights for the policy pi - this is done by solving the linear system A*w_pi = b
w_pi,_,_,_ = np.linalg.lstsq(A, b)

print w_pi



