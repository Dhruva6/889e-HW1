import numpy as np

def OMP_TD(sars, beta, gamma = 0.99):
    k = 9
    
    I = set()
    w_pi = np.zeros((k, 1))
    c = [beta for beta in range(k)]
    j = 0
    n = len(sars)
    phi_s = np.column_stack(sars[:, 0]).T
    phi_s_prime = np.column_stack(sars[:, 3]).T
    R = sars[:, 2]
    while(len(I)!=n and c[j] <= beta):
        c = np.linalg.norm(np.dot(phi_s.T, (R + gamma * np.dot(phi_s_prime, w_pi) - np.dot(phi_s, w_pi))))/n
        j = np.argmax(c)
        if c[j] > beta:
            I.add(j)
        w_pi = np.linalg.inverse(np.dot(phi_s[I].T, phi_s[I]) - gamma* np.dot(phi_s[I].T, phi_s_prime[I]))phi_s.T*R
    return w_pi
            
        
