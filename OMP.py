import numpy as np

def OMP_TD(sars, beta, gamma = 0.99):
    print "Started OMP-TD"
    k = 9
    I = set()
    w_pi = np.zeros((k, 1))
    c = [beta for beta in range(k)]
    j = 0
    n = len(sars)
    
    phi_s = np.array([np.array(elem) for elem in sars[:,0]])
    phi_s = np.reshape(phi_s, (n, k))

    phi_s_prime = np.array([np.array(elem) for elem in sars[:,3]])
    phi_s_prime = np.reshape(phi_s_prime, (n, k))

    R = sars[:, 2]
    while(len(I)!=n and c[j] <= beta):
        print "HERE"
        c = np.linalg.norm(np.dot(phi_s.T, (R + gamma * np.dot(phi_s_prime, w_pi) - np.dot(phi_s, w_pi))))/n
        print c.shape, " AND HERE"
        j = np.argmax(c)
        if c[j] > beta:
            I.add(j)
        print phi_s[I].shape
        w_pi = np.linalg.inverse(np.dot(phi_s[I].T, phi_s[I]) - gamma* np.dot(phi_s[I].T, phi_s_prime[I]))
        w_pi = np.dot(w_pi, np.dot(phi_s.T, R))
    return w_pi
            
        
