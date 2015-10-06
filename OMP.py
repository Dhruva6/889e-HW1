import numpy as np

def OMP_TD(sars, beta, gamma = 0.99):
    print "Started OMP-TD"
    k = 9
    numFeat = 5
    I = set()
    w_pi = np.zeros((k, 1))
    c = [beta+1 for i in range(k)]
    j = 0
    n = len(sars)
    
    phi_s = np.array([np.array(elem) for elem in sars[:,0]])
    phi_s = np.reshape(phi_s, (n, k))

    phi_s_prime = np.array([np.array(elem) for elem in sars[:,3]])
    phi_s_prime = np.reshape(phi_s_prime, (n, k))

    R = np.array(sars[:, 2])
    beta = 10**6
    c[0] = beta+1
    while(len(I) < numFeat):# or c[j] > beta):
        #c = np.linalg.norm(np.dot(phi_s.T, (R + gamma * np.dot(phi_s_prime, w_pi) - np.dot(phi_s, w_pi))))/n
        temp = (gamma * np.dot(phi_s_prime, w_pi) - np.dot(phi_s, w_pi))
        c = np.array([R[i]+temp[i] for i in range(len(R))])
        c = abs(np.dot(phi_s.T, c))/n
        minC = min(c)
        c = [elem if idx not in I else minC-1 for idx, elem in enumerate(list(c))]
        j = np.argmax(c)        
        #if c[j] > beta:
        I.add(j)

        lI = list(I)
        w_pi = np.linalg.pinv(np.dot(phi_s[:,  lI].T, phi_s[:, lI]) - gamma* np.dot(phi_s[:, lI].T, phi_s_prime[:, lI]))
        w_pi = np.dot(w_pi, np.dot(phi_s[:, lI].T, R))
        new_w_pi = np.zeros((k, 1))
        for i, idx in enumerate(lI):
            new_w_pi[idx] = w_pi[i]
        w_pi = new_w_pi
    print "Finished OMP-TD"
    elemList = []
    for idx, elem in enumerate(w_pi):
        if elem > 0:
            elemList.append(idx)
    print elemList
    return elemList
            
        
