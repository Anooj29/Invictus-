import numpy as np 
from numpy import * 

a1 = 6.2 
a2 = 5.3 
a3 = 0 
a4 = 7 

x = -7 
y = 9 

r1 = sqrt(x**2+y**2)
phi_1 = arccos((a4**2-a2**2-r1**2)/(-2*a2*r1))
phi_2 = arctan2(y, x)
theta_1 = rad2deg(phi_2-phi_1)

phi_3 = arccos((r1**2-a2**2-a4**2)/(-2*a2*a4))
theta_2 = 180-rad2deg(phi_3)

print('theta one: ', theta_1)
print('theta two: ', theta_2)