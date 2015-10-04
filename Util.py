# A collection of Utility functions - copied from Real\ life\ RL\ -\ HW1.py

# Import
import numpy as np
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

def generate_sarsa(data):
    """
        Function that returns the next (s, a, r, s', a') pair from the input data one by one every time you call it. Requires a 
        scaler to have been computed so that we can approximate the vaues for the 'NA' pairs in the data.
    """
    # Compute the known states and then compute a 'scaler' which stores the means and variances that will be used for standardization.
    known_states = [state for state in get_known_states(data)]
    scaler = preprocessing.StandardScaler().fit(known_states)
    event_length = 9 + 9 + 2 + 1
    state_length = 9 + 2
    num_states = 24
    sarsa = []
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
                s_prime = datum[11:20].astype(np.float)
                a_prime = 1.0 if datum[20:21] == 'true' else 0.0       
                scaler.transform(s_prime)
                sarsa.append([s, a, r, s_prime, a_prime])
            # IF there was a value error it means there was a 'NA' field somewhere. 
            except ValueError:
                # ONLY S AND S' have these 'NA' fields (I've confirmed). Therefore we go through them and replace any
                # fields that have 'NA' with the mean of the corresponding feature, and then apply the scaler.
                s = np.array([elem if elem!='NA' else scaler.mean_[i].astype(np.float) for i, elem in enumerate(datum[:9])]).astype(np.float)
                scaler.transform(s)
                s_prime = np.array([elem if elem!='NA' else scaler.mean_[i].astype(np.float) for i, elem in enumerate(datum[:9])]).astype(np.float)
                scaler.transform(s_prime).astype(np.float)
                sarsa.append([s, a, r, s_prime, a_prime])
            curr_state += 1
    return sarsa
