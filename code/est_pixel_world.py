import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    ones_column = np.ones((pixels.shape[0], 1), dtype=pixels.dtype)
    pixels = np.concatenate((pixels, ones_column), axis=1)
    K_inv = np.linalg.inv(K)
    pixels_cal = np.dot(K_inv, pixels.T).T
    lamb = np.zeros(pixels.shape[0])
    for i in range(pixels.shape[0]):
        lamb[i] = -t_wc[2] / (R_wc[2,0] * pixels_cal[i,0] + R_wc[2,1] * pixels_cal[i,1] + R_wc[2,2] * pixels_cal[i,2])
    Pw = lamb[:, np.newaxis] * (R_wc @ pixels_cal.T).T + t_wc

    ##### STUDENT CODE END #####
    return Pw

if __name__ == "__main__":
    # Real world data
    K = np.array([[823.8, 0.0, 304.8],
                  [0.0, 822.8, 236.3],
                  [0.0, 0.0, 1.0]])
    t_wc = np.array( [-0.34344175, - 0.52469404,  0.34557451])
    R_wc = np.array([
        [ 0.876204,   -0.15612653,  0.45595071],
        [-0.47769752, -0.40661165,  0.77876315],
        [ 0.06380928, -0.90016191, -0.43085602]] )

    Pixels = np.array([[220,  330],
                       [550 , 260]])

    print(est_pixel_world(Pixels,R_wc,t_wc,K))
