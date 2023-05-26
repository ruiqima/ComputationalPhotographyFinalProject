import numpy as np
import cv2
from skimage import io

class Debug:
    def __init__(self, debug_println_number = 20, is_debugging = True):
        self.debug_println_count = 0
        self.debug_println_number = debug_println_number
        self.is_debugging = is_debugging
    
    def print_start(self, I):
        if self.is_debugging:
            print('\n')
            print('START BILATERAL FILTERING ON IMG, SHAPE ({}, {}, {}) **********'.format(I.shape[0], I.shape[1], I.shape[2]))
    
    def print_G_S(self, G_s):
        if self.is_debugging:
            print('Constant G_s for this Gaussian kernel is:')
            print(G_s)
    
    def print_progress(self, I):
        h, w, _ = I.shape
        self.debug_println_count += 1
        interval =  (h * w) // self.debug_println_number
        if self.debug_println_count % interval == 0:
            print('in progress...  {} trail  {}% complete'.format(self.debug_println_count, self.debug_println_count * 100 // (h * w) ))


'''
main function to export!
Call this function to run the function of this file.
'''
def run_self_implemented_bilateral_filtering(I, kernel, sigma_s, sigma_r, debugger, output_folder):
    I_filtered = bilateral_filtering(I, kernel, sigma_s, sigma_r, debugger)
    I_diff = get_difference(I_filtered, I)

    io.imsave('{}/I_filtered.png'.format(output_folder), I_filtered)
    io.imsave('{}/I_difference.png'.format(output_folder), I_diff)
    side_by_side = np.zeros((I_filtered.shape[0], I_filtered.shape[1] * 2 + 10, I_filtered.shape[2]))
    side_by_side[:, :I_filtered.shape[1], :] = I_filtered
    side_by_side[:, I_filtered.shape[1] : I_filtered.shape[1] + 10, 0] = 255
    side_by_side[:, I_filtered.shape[1] + 10:, :] = I 
    io.imsave('{}/I_side_by_side.png'.format(output_folder), side_by_side)



def g(x, sigma):
    return np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (2 * np.pi * (sigma ** 2))


def get_difference(I1, I2):
    d = I1 - I2
    Id = (d - np.min(d)) / (np.max(d) - np.min(d))
    return Id


def bilateral_filtering(I, ksize, sigma_s, sigma_r, debugger = Debug()):
    debugger.print_start(I)

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

    # debugger.print_G_S(G_s)

    def compute_filtered_value(p_i, p_j):
        '''
        p_i and p_j are on the padded image
        '''
        debugger.print_progress(I)

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
    



