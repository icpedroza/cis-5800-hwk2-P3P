import numpy as np


def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)
        Pw = R@Pc + t
    """

    ##### STUDENT CODE START #####
    ones_column = np.ones((Pc.shape[0], 1), dtype=Pc.dtype)
    Pc = np.concatenate((Pc, ones_column), axis=1)
    K_inv = np.linalg.inv(K)
    Pc_cal = (K_inv @ Pc.T).T

    a = np.linalg.norm(Pw[1] - Pw[2])
    b = np.linalg.norm(Pw[0] - Pw[2])
    c = np.linalg.norm(Pw[0] - Pw[1])

    scale_factor = np.linalg.norm(Pc_cal[:3, :], axis=1, keepdims=True)
    Pc_cal_norm = Pc_cal[:3, :] / scale_factor
    dp23 = np.dot(Pc_cal_norm[1], Pc_cal_norm[2])
    dp13 = np.dot(Pc_cal_norm[0], Pc_cal_norm[2])
    dp12 = np.dot(Pc_cal_norm[0], Pc_cal_norm[1])
    alpha = np.arccos(dp23)
    beta = np.arccos(dp13)
    gamma = np.arccos(dp12)

    A_4 = ((a**2 - c**2) / b**2 - 1)**2 - (4 * c**2 / b**2) * np.cos(alpha)**2
    A_3 = 4 * (((a**2 - c**2) / b**2) * (1 - ((a**2 - c**2) / b**2)) * np.cos(beta)
            - (1 - ((a**2 + c**2) / b**2)) * np.cos(alpha) * np.cos(gamma)
            + 2 * (c**2 / b**2) * np.cos(alpha)**2 * np.cos(beta))
    A_2 = 2 * (((a**2 - c**2) / b**2)**2 - 1 
            + 2 * ((a**2 - c**2) / b**2)**2 * np.cos(beta)**2
            + 2 * ((b**2 - c**2) / b**2) * np.cos(alpha)**2
            - 4 * ((a**2 + c**2) / b**2) * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
            + 2 * ((b**2 - a**2) / b**2) * np.cos(gamma)**2)
    A_1 = 4 * (-((a**2 - c**2) / b**2) * (1 + ((a**2 - c**2) / b**2)) * np.cos(beta)
            + 2 * (a**2 / b**2) * np.cos(gamma)**2 * np.cos(beta)
            - (1 - ((a**2 + c**2) / b**2)) * np.cos(alpha) * np.cos(gamma))
    A_0 = (1 + ((a**2 - c**2) / b**2))**2 - 4 * (a**2 / b**2) * np.cos(gamma)**2
    
    coefficients = np.array([A_4, A_3, A_2, A_1, A_0])
    coefficients = coefficients.astype(float)
    roots = np.roots(coefficients)
    v = roots[np.isclose(roots.imag, 0)].real

    u = ((-1 + ((a**2 - c**2) / b**2)) * v**2
         - 2 * ((a**2 - c**2) / b**2) * np.cos(beta) * v
         + 1 + ((a**2 - c**2) / b**2)) / (2 * (np.cos(gamma) - v * np.cos(alpha)))
    
    d1 = np.sqrt(a**2 / (u**2 + v**2 - 2 * u * v * np.cos(alpha)))
    d2 = u * d1
    d3 = v * d1

    cam_coordinate_combos = np.zeros((len(d1), Pc_cal_norm.shape[0], Pc_cal_norm.shape[1]))

    for i in range(len(d1)):
        cam_coordinate_combos[i] = Pc_cal_norm * np.array([d1[i], d2[i], d3[i]])[:, np.newaxis]

    best_R = None
    best_t = None
    best_diff = float('inf')
    for i in range(len(cam_coordinate_combos)):
        test = cam_coordinate_combos[i]
        R_wc, t_wc = Procrustes(cam_coordinate_combos[i], Pw[:3, :])
        # R_cw = R.T
        # t_cw = -R.T@t
        fth_pt_c = R_wc.T @ (Pw[3] - t_wc)
        fth_pt_i = K @ fth_pt_c
        fth_pt = fth_pt_i / fth_pt_i[-1]
        diff = fth_pt[:2] - Pc[3, :2]
        magnitude = np.linalg.norm(diff)
        if magnitude < best_diff:
            best_diff = magnitude
            best_R = R_wc
            best_t = t_wc

    ##### STUDENT CODE END #####

    return best_R, best_t


def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    A_centroid = np.mean(X, axis=0)
    B_centroid = np.mean(Y, axis=0)
    A = X - A_centroid
    B = Y - B_centroid

    R_0 = A.T @ B
    [U, S, VT] = np.linalg.svd(R_0)

    UV_T = U @ VT
    det_UV_T = np.linalg.det(UV_T)
    D = np.diag([1, 1, det_UV_T])
    # VU_T = VT.T @ U.T
    # det_VU_T = np.linalg.det(VU_T)
    # D = np.diag([1, 1, det_VU_T])
    R_cw = U @ (D @ VT)
    t_cw = A_centroid - (R_cw @ B_centroid)
    R = R_cw.T
    t = -R_cw.T@t_cw
    ##### STUDENT CODE END #####

    return R, t

if __name__ == "__main__":
    K = np.array([[823.8, 0.0, 304.8],
                [0.0, 822.8, 236.3],
                [0.0, 0.0, 1.0]])
    Pc = np.random.rand(4,2)
    Pw = np.random.rand(4,3)

    X = np.random.rand(10,3)
    Y = np.random.rand(10,3)
    # Procrustes(X,Y)
    Pc = np.array(
        [[304.28405762, 346.36758423],
        [449.04196167, 308.92901611],
        [363.24179077, 240.77729797],
        [232.29425049, 266.60055542]]
    )
    Pw = np.array([
        [-0.07, - 0.07,  0.],
        [0.07, - 0.07,  0.],
        [0.07, 0.07, 0.],
        [-0.07,  0.07,  0.]
    ])

    P3P(Pc,Pw,K)

