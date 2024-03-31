from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)
        R@Pc + t = Pw
    """

    ##### STUDENT CODE START #####
    H = est_homography(Pw[:, :2], Pc)
    K_inv = np.linalg.inv(K)
    result = K_inv @ H

    h1 = result[:, 0]
    h2 = result[:, 1]
    h3 = result[:, 2]
    h1xh2 = np.cross(h1, h2)
    R_0 = np.column_stack((h1, h2, h1xh2))

    [U, S, VT] = np.linalg.svd(R_0)

    UV_T = U @ VT
    det_UV_T = np.linalg.det(UV_T)
    D = np.diag([1, 1, det_UV_T])
    R = U @ (D @ VT)
    # R_2 = U @ V.T
    # r1xr2 = np.cross(R_2[:, 0], R_2[:, 1])
    # R_0 = np.column_stack((R_2, r1xr2))
    # s1 = S[0][0]
    # s2 = S[1][1]
    # lamb = (s1 + s2) / 2
    t = h3 / np.linalg.norm(h1)
    ##### STUDENT CODE END ##### 

    return R.T, -R.T@t


if __name__ == "__main__":
    K = np.array([[823.8, 0.0, 304.8],
                [0.0, 822.8, 236.3],
                [0.0, 0.0, 1.0]])
