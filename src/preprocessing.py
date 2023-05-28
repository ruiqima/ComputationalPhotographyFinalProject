'''
The goal is to find the "star pixels".
Filtering should not be applied to these pixels.

Here, we assume these "star pixels" don't need to be denoised
under the fact that these points are bright,
the signal-to-noise ratio should be high.

Input:
a RGB .jpg image: 3D array

Output:
1) an array that contains the coorinates of "star pixels": 2D array.
2) "I_threshold_xx.png" ---> an .png image that marks all the "star-for-sure pixels" to red
3) "I_threshold_ex_xx_xx.png" ---> an .png image that additionally marks all the "star-with-probability pixles" to green

'''

import cv2
from skimage import io
import numpy as np
from debug import Debug


def preprocessing(I, output_folder, intensity_threshold, extension_intensity_threshold, debugger = Debug()):
    # step 1: RGB -> GRAY
    GRAY = RBG2GRAY(I, output_folder)

    # step 2: find the pixels with intensity > threshold
    # Pr[this pixel is a star pixel] is large
    debugger.print_pick_pixels_progress()
    i_coords, j_coords = pick_pixels_with_intensity_larger_than_threshold(I, GRAY, output_folder, intensity_threshold)

    # step 3: expand the range by 1 to 2 pixels around it
    debugger.print_expand_border_progress()
    i_coords_extension, j_coords_extension = expand_star_pixels_border(I, GRAY, i_coords, j_coords, output_folder, intensity_threshold, extension_intensity_threshold)

    return i_coords_extension, j_coords_extension

def RBG2GRAY(I, output_folder):
    GRAY = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    io.imsave(f'{output_folder}/I_gray.png', GRAY)
    return GRAY


def pick_pixels_with_intensity_larger_than_threshold(I, GRAY, output_folder, intensity_threshold):
    i_coords, j_coords = np.where(GRAY > intensity_threshold)
    res = np.array(zip(i_coords, j_coords))

    # mark these coordinates red in the original RGB I
    I_threshold = I.copy()
    I_threshold[i_coords, j_coords, :] = [255, 0, 0]
    io.imsave(f'{output_folder}/I_threshold_{intensity_threshold}.png', I_threshold)
    return i_coords, j_coords


def expand_star_pixels_border(I, GRAY, i_coords, j_coords, output_folder, intensity_threshold, extension_intensity_threshold):
    # can be ran in O(len(i_coords) x log(len(i_coords)))
    set_extension = set()
    for i, j in zip(i_coords, j_coords):
        for n_i in range(i - 1, i + 2):
            for n_j in range(j - 1, j + 2):
                if n_i < 0 or n_i >= GRAY.shape[0] or n_j < 0 or n_j >= GRAY.shape[1]:
                    break
                if GRAY[n_i, n_j] > extension_intensity_threshold and not (n_i == i and n_j == j):
                    set_extension.add((n_i, n_j))

    set_stars = set()
    for i in range(len(i_coords)):
        set_stars.add((i_coords[i], j_coords[i]))
    
    set_added_by_extension = set_extension - set_stars
    i_coords_extended = np.array([x[0] for x in set_added_by_extension])
    j_coords_extended = np.array([x[1] for x in set_added_by_extension])

    I_threshold_extended = I.copy()
    I_threshold_extended[i_coords, j_coords, :] = [255, 0, 0]
    I_threshold_extended[i_coords_extended, j_coords_extended, :] = [0, 255, 0]
    io.imsave(f'{output_folder}/I_threshold_ex_{intensity_threshold}_{extension_intensity_threshold}.png', I_threshold_extended)

    set_union = set_stars.union(set_extension)
    i_coords_extension = np.array([x[0] for x in set_union])
    j_coords_extension = np.array([x[1] for x in set_union])
    return i_coords_extension, j_coords_extension

