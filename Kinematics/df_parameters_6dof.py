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

np.set_printoptions(precision=4, suppress=True)

dh_table = [ 
            [np.radians(30), 4, 0, np.radians(90)],
            [np.radians(45), 0, 5, np.radians(90)],
            [np.radians(60), 0, 3, np.radians(90)],
            [np.radians(90), 2, 0,np.radians(90)],
            [np.radians(45), 0, 0, np.radians(90)],
            [np.radians(0), 1, 0,np.radians(90)]
            ]

T_matrices = []

for row in dh_table:
    theta, d, r, alpha = row
    T = dh_parameter(theta, d, r, alpha)
    T_matrices.append(T)
    
print("\n>> Individual Transformation Matrice\n")

for i, T in enumerate(T_matrices):
    print(f"T{i}{i+1} =")
    print(T)
    print()
    
T_final = np.eye(4)
for T in T_matrices:
    T_final = T_final @ T
    
print("\n>> Final Transformation Matrix\n")
print(np.round(T_final, 4))

position = T_final[0:3, 3]

print("\n>> End Effector Position\n")
print(f"x = {position[0]:.4f}")
print(f"y = {position[1]:.4f}")
print(f"z = {position[2]:.4f}") 

rotation = T_final[0:3, 0:3]   

print("\n>> Rotation Matrix\n")
print(np.round(rotation, 4))