import numpy as np 

def dh_parameter(theta, d, r, alpha):
    ct = np.cos(theta)
    st = np.sin(theta)
    
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([[ct, -st*ca, st*sa, r*ct],
                  [st, ct*ca, -ct*sa, r*st],
                  [0, sa, ca, d],
                  [0, 0, 0, 1]])
    
    return T 

theta = np.radians(45)
alpha = np.radians(90)

X = dh_parameter(theta, 10, 5, alpha)
print(X)
    