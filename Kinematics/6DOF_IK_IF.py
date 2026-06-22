import numpy as np


def dh(theta, d, a, alpha):

    ct = np.cos(theta)
    st = np.sin(theta)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(q, DH):

    T = np.eye(4)

    for i in range(6):

        theta = q[i] + DH[i][0]
        d     = DH[i][1]
        a     = DH[i][2]
        alpha = DH[i][3]

        T = T @ dh(theta, d, a, alpha)

    return T


def jacobian(q, DH):

    T = np.eye(4)

    origins = []
    z_axes = []

    origins.append(np.array([0, 0, 0]))
    z_axes.append(np.array([0, 0, 1]))

    for i in range(6):

        theta = q[i] + DH[i][0]
        d     = DH[i][1]
        a     = DH[i][2]
        alpha = DH[i][3]

        T = T @ dh(theta, d, a, alpha)

        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    pe = origins[-1]

    J = np.zeros((6, 6))

    for i in range(6):

        zi = z_axes[i]
        oi = origins[i]

        Jv = np.cross(zi, pe - oi)
        Jw = zi

        J[0:3, i] = Jv
        J[3:6, i] = Jw

    return J


def rotation_error(Rd, Rc):

    Re = Rd @ Rc.T

    return 0.5 * np.array([
        Re[2,1] - Re[1,2],
        Re[0,2] - Re[2,0],
        Re[1,0] - Re[0,1]
    ])


def inverse_kinematics(
        target,
        q0,
        DH,
        max_iter=1000,
        tol=1e-4,
        damping=0.05):

    q = np.array(q0, dtype=float)

    for k in range(max_iter):

        T = forward_kinematics(q, DH)

        pos_current = T[:3, 3]
        pos_target = target[:3, 3]

        R_current = T[:3, :3]
        R_target = target[:3, :3]

        ep = pos_target - pos_current
        eo = rotation_error(R_target, R_current)

        error = np.concatenate((ep, eo))

        if np.linalg.norm(error) < tol:

            print("Converged")
            print("Iterations:", k)

            return q

        J = jacobian(q, DH)

        JT = J.T

        dq = JT @ np.linalg.inv(
            J @ JT +
            (damping ** 2) * np.eye(6)
        ) @ error

        q = q + dq

    print("Did not converge")
    return q

# Replace with your actual robot dimensions

DH = np.array([

    [0, 0.40, 0.00, np.pi/2],
    [0, 0.00, 0.35, 0],
    [0, 0.00, 0.30, 0],
    [0, 0.20, 0.00, np.pi/2],
    [0, 0.00, 0.00, -np.pi/2],
    [0, 0.10, 0.00, 0]

])

target = np.eye(4)

target[0,3] = 0.40
target[1,3] = 0.25
target[2,3] = 0.50


q0 = np.deg2rad([0, 20, 30, 0, 45, 0])


solution = inverse_kinematics(
    target,
    q0,
    DH
)

print("\nJoint Angles (rad)")
print(solution)

print("\nJoint Angles (deg)")
print(np.rad2deg(solution))

print("\nVerification FK")

T = forward_kinematics(solution, DH)

print(T)
