import numpy as np
import cv2
from scipy.interpolate import interpn
from skimage import io

def g(x, sigma):
    return np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (2 * np.pi * (sigma ** 2))

def get_difference(I1, I2):
    d = I1 - I2
    Id = (d - np.min(d)) / (np.max(d) - np.min(d))
    io.imsave('outputs/difference.png', Id)


def bilateral_filtering(I, ksize, sigma_s, sigma_r):
    YCrCb = cv2.cvtColor(I, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(YCrCb)

    # padding the edges
    half_ksize = (ksize - 1) // 2
    Y_padded = np.pad(Y, half_ksize, 'edge')
    
    # calculate G_sigma_s
    points = np.array(np.meshgrid(
                np.arange(- half_ksize, half_ksize + 1), 
                np.arange(- half_ksize, half_ksize + 1))
            ).T.reshape(-1, 2).reshape(ksize, ksize, 2)
    l2_norm = np.linalg.norm(points - np.array([0, 0]), axis = 2)
    G_s = g(l2_norm, sigma_s)

    def compute_filtered_value(p_i, p_j):
        '''
        p_i and p_j are on the padded image
        '''
        if p_i < half_ksize or p_i + half_ksize + 1 > Y_padded.shape[0]:
            return 0.001
        if p_j < half_ksize or p_j + half_ksize + 1 > Y_padded.shape[1]:
            return 0.001
        
        # qs is the coordinates on the original unpadded I
        q_xv, q_yv = np.meshgrid(np.arange(p_i - half_ksize, p_i + half_ksize + 1), np.arange(p_j - half_ksize, p_j + half_ksize + 1))

        # calculate G_sigma_r
        range_difference = Y_padded[q_xv, q_yv] - Y_padded[p_i, p_j]
        normalized_range_difference = range_difference / np.sum(range_difference)
    
        G_r = g(normalized_range_difference, sigma_r)
        G_r = G_r / np.sum(G_r)
        
        # calculate joint G (w)
        w = G_s * G_r
        normalized_w = w / np.sum(w)

        # apply gaussian filter to point p_i, p_j and get updated value for position p
        return np.sum(normalized_w * Y_padded[q_xv, q_yv])


    vectorized_compute_filtered_value = np.vectorize(compute_filtered_value)
    xv, yv = np.meshgrid(np.arange(Y_padded.shape[0]), np.arange(Y_padded.shape[1]))

    Y_filtered = vectorized_compute_filtered_value(xv, yv)
    Y_filtered = Y_filtered.T
    Y_filtered = Y_filtered[half_ksize: -half_ksize, half_ksize: -half_ksize]

    # Y_filtered = gamma_decoding(Y_filtered)
    I_filtered = cv2.cvtColor(cv2.merge([Y_filtered.astype(np.uint8), Cr, Cb]), cv2.COLOR_YCrCb2RGB)
    return I_filtered
    


# set the print format of numpy array
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})

KERNEL = 5
SIGMA_S = 1
SIGMA_R = 0.5

# I = np.load("data/one_tiff.npy")
I = io.imread("test_data/small_2.jpg")
# io.imsave('I.png', I)
# print('# of 0 in I: {}'.format(np.count_nonzero(I == 0)))
I_filtered = bilateral_filtering(I, KERNEL, SIGMA_S, SIGMA_R)

io.imsave('outputs/bilateral_filtered.png', I_filtered)
get_difference(I_filtered, I)
