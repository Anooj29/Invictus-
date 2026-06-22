import numpy as np 
from numpy import * 


def dh_matrix(theta, d, a, alpha):

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

    return T


theta = np.radians(45)
alpha = np.radians(90)

T = dh_matrix(theta, 10, 5, alpha)

print(T)