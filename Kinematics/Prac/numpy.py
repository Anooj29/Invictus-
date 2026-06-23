import numpy as np 
from numpy import * 
import matplotlib.pyplot as plt 

def get_input():
    
    link1_lenght = float(input("Enter length of first link: "))
    link2_length = float(input("Enter lenght of second link: "))
    
    theta1 = np.radians(float(input("Enter joint 1 angle (deg): ")))
    theta2 = np.radians(float(input("Enter joint 2 angle (deg): ")))
    
    return link1_lenght, link2_length, theta1, theta2