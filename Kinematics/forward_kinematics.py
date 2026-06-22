import numpy as np
from numpy import *

# length of the links 
a1 = 6.2 
a2 = 5.2 
a3 = 0 
a4 = 7 

# motor angles 
theta_1 = 30 
theta_2 = 60 

theta_1 = (theta_1/180)*pi 
theta_2 = (theta_2/180)*pi 

r0_1 = [[cos(theta_1), -sin(theta_1), 0],
        [sin(theta_1), cos(theta_1), 0],
        [0, 0, 1]]

r1_2 = [[cos(theta_2), -sin(theta_2), 0],
        [sin(theta_2), cos(theta_2), 0],
        [0, 0, 1]]

r0_2 = dot(r0_1, r1_2)

# Displacement vector 
d0_1 = [[a2*cos(theta_1)], [a2*sin(theta_1)], [a1]]
d1_2 = [[a4*cos(theta_1)], [a4*sin(theta_2)], [a3]]

# Homogeneous transformation matrix from joint 0 to 1 

H0_1 = concatenate((r0_2, d0_1), 1)
H0_1 = concatenate((H0_1, [[0, 0, 0, 1]]), 0)

print(matrix(H0_1))
print('\n')

# Homogeneous transformation matrix from joint 0 to 1 

H1_2 = concatenate((r1_2, d1_2), 1)
H1_2 = concatenate((H1_2, [[0, 0, 0, 1]]), 0)

H0_2 = dot(H0_1, H1_2)

print(matrix(H0_2))