
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
        Function that returns the next (s, a, r, s', a(s')) pair from the input data one by one every time you call it.
    """
    event_length = 9 + 9 + 2 + 1
    state_length = 9 + 2
    num_states = 24
    for episode in data:
        # Start at the beginning and keep looking at a net length of len(s) + len(a) + len(r) + len(s') + len(a(s')) points
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
                s_prime = datum[11:20].astype(np.float)
                pi_s_prime =  1.0 if datum[20:21]=='true' else 0.0               
                yield s, a, r, s_prime, pi_s_prime
            # There's a problem if a data field is 'NA' - Not entirely sure what do in that case so for now I'm just ignoring
            # those data points
            except ValueError:
                pass                
            curr_state += 1

# generate the object that will produces us the (s,a,r,s', pi_s_prime) tuple
event_iterator = create_event_iterator(data)

# current policy 
policy = []

# iterate
for _,a,_,_,_ in event_iterator:
    policy.append(a)

## console debug
#print len(pi_current)
 
# Instantiation of the Ordinary Least Squares class
ols = linear_model.LinearRegression()

# at present, our basis is N^d - where d is 9 (features) + 1 (action)
k = 10

# to avoid singularities, we start off A with a small delta along the diagonal
delta = 1e-03

# the "A" Matrix (kxk) 
A = delta*np.eye(k)

# the b vector
b = np.zeros([k, 1], dtype=float)

# the discount factor gamma
gamma = 0.9

# generate the object that will produces us the (s,a,r,s', pi_s_prime) tuple
event_iterator = create_event_iterator(data)

print "Building the matrix A and the vector b"

# iterate through the tuple
for s, a, r, s_prime, pi_s_prime in event_iterator:
    
    # In future implementations, we can have a different basis such as RBF or a polynomial for our features
    # In the current implementation, we take the given features as the basis 
    phi_s = np.reshape(np.append(s,a), (k,1))
    phi_s_prime = np.reshape(np.append(s_prime, pi_s_prime), (k,1))
    
    # Update A - here, we add to A, a Rank 1 matrix formed by phi(s) and (phi(s) - gamma*phi(s'))
    A = A + np.outer(phi_s, phi_s - gamma*phi_s_prime)

    # update B - we add to B, the feature vector scaled by 
    b = b + r*phi_s

print "Solving for the weights"

# compute the weights for the policy pi - this is done by solving the linear system A*w_pi = b
w_pi,_,_,_ = np.linalg.lstsq(A, b)

print w_pi

print "Improving Policy..."
## One Time (Greedy) Policy Improvement
idx = 0
for s,_,_,_,_ in event_iterator:

    # the state-action value for action 0.0
    q0 = np.dot(np.append(s,0.0), w_pi)

    # the state-action value for action 1.0
    q1 = np.dot(np.append(s,1.0), w_pi)

    # update the policy as argmax(action = {0.0, 1.0}) Q^
    policy[idx] = 1.0 if q0 < q1 else 0.0

    # to the next state
    idx = idx+1    


