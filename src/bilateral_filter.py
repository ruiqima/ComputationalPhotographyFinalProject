'''
The goal is to do bilateral filtering on the image.
Type 1: on all pixels
Type 2: on pixels that are not considered as star pixels

Also increase the contrast by making the background darker

Output images:
Type 1: 
    - "I_filtered_{kernel}_{sigma_s * 1000}_{sigma_r * 1000}_{dark_constant * 100}.png"
    - "I_difference_{kernel}_{sigma_s * 1000}_{sigma_r * 1000}_{dark_constant * 100}.png"
    - "I_side_by_side_{kernel}_{sigma_s * 1000}_{sigma_r * 1000}_{dark_constant * 100}.png"

Type 2:
    - "preserve_stars_I_filtered_{kernel}_{sigma_s * 1000}_{sigma_r * 1000}_{dark_constant * 100}.png"
    - "preserve_stars_I_difference_{kernel}_{sigma_s * 1000}_{sigma_r * 1000}_{dark_constant * 100}.png"
    - "preserve_stars_I_side_by_side_{kernel}_{sigma_s * 1000}_{sigma_r * 1000}_{dark_constant * 100}.png"

'''


import numpy as np
import cv2
from skimage import io
from debug import Debug


'''
main function 1 to export!
'''
def run_self_implemented_bilateral_filtering_on_all_non_star_pixels(I, i_coords, j_coords, kernel, sigma_s, sigma_r, dark_constant, debugger, output_folder):
    star_coords = set()
    for i in range(len(i_coords)):
        star_coords.add((i_coords[i], j_coords[i]))
    I_filtered = bilateral_filtering(I, kernel, sigma_s, sigma_r, star_coords, dark_constant, debugger)
    I_diff = get_difference(I_filtered, I)

    io.imsave(f'{output_folder}/preserve_stars_I_filtered_{kernel}_{int(sigma_s * 1000)}_{int(sigma_r * 1000)}_{int(dark_constant * 100)}.png', I_filtered)
    io.imsave(f'{output_folder}/preserve_stars_I_difference_{kernel}_{int(sigma_s * 1000)}_{int(sigma_r * 1000)}_{int(dark_constant * 100)}.png', I_diff)
    side_by_side = np.zeros((I_filtered.shape[0], I_filtered.shape[1] * 2 + 10, I_filtered.shape[2]))
    side_by_side[:, :I_filtered.shape[1], :] = I_filtered
    side_by_side[:, I_filtered.shape[1] : I_filtered.shape[1] + 10, 0] = 255
    side_by_side[:, I_filtered.shape[1] + 10:, :] = I 
    io.imsave(f'{output_folder}/preserve_stars_I_side_by_side_{kernel}_{int(sigma_s * 1000)}_{int(sigma_r * 1000)}_{int(dark_constant * 100)}.png', side_by_side)


'''
main function 2 to export!
Call this function to run the function of this file.
'''
def run_self_implemented_bilateral_filtering_on_all_pixels(I, kernel, sigma_s, sigma_r, dark_constant, debugger, output_folder):
    I_filtered = bilateral_filtering(I, kernel, sigma_s, sigma_r, None, dark_constant, debugger)
    I_diff = get_difference(I_filtered, I)

    io.imsave(f'{output_folder}/I_filtered_{kernel}_{int(sigma_s * 1000)}_{int(sigma_r * 1000)}_{int(dark_constant * 100)}.png', I_filtered)
    io.imsave(f'{output_folder}/I_difference_{kernel}_{int(sigma_s * 1000)}_{int(sigma_r * 1000)}_{int(dark_constant * 100)}.png', I_diff)
    side_by_side = np.zeros((I_filtered.shape[0], I_filtered.shape[1] * 2 + 10, I_filtered.shape[2]))
    side_by_side[:, :I_filtered.shape[1], :] = I_filtered
    side_by_side[:, I_filtered.shape[1] : I_filtered.shape[1] + 10, 0] = 255
    side_by_side[:, I_filtered.shape[1] + 10:, :] = I 
    io.imsave(f'{output_folder}/I_side_by_side_{kernel}_{int(sigma_s * 1000)}_{int(sigma_r * 1000)}_{int(dark_constant * 100)}.png', side_by_side)



def g(x, sigma):
    return np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (2 * np.pi * (sigma ** 2))


def get_difference(I1, I2):
    d = I1 - I2
    Id = (d - np.min(d)) / (np.max(d) - np.min(d))
    return Id


def bilateral_filtering(I, ksize, sigma_s, sigma_r, star_coords, dark_constant, debugger = Debug()):
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

    def compute_filtered_value(p_i, p_j):
        '''
        p_i and p_j are on the padded image
        '''
        debugger.print_progress(I)

        # if it's a star pixel
        if star_coords is not None and (p_i - half_ksize, p_j - half_ksize) in star_coords:
            return Y_padded[p_i, p_j]

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
        return np.sum(normalized_w * Y_padded[q_xv, q_yv]) / dark_constant


    vectorized_compute_filtered_value = np.vectorize(compute_filtered_value)
    xv, yv = np.meshgrid(np.arange(Y_padded.shape[0]), np.arange(Y_padded.shape[1]))

    Y_filtered = vectorized_compute_filtered_value(xv, yv)
    Y_filtered = Y_filtered.T
    Y_filtered = Y_filtered[half_ksize: -half_ksize, half_ksize: -half_ksize]

    # Y_filtered = gamma_decoding(Y_filtered)
    I_filtered = cv2.cvtColor(cv2.merge([Y_filtered.astype(np.uint8), Cr, Cb]), cv2.COLOR_YCrCb2RGB)
    return I_filtered
    



