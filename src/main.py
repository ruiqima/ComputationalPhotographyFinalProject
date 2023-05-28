from skimage import io
import numpy as np
from bilateral_filter import run_self_implemented_bilateral_filtering_on_all_pixels, run_self_implemented_bilateral_filtering_on_all_non_star_pixels
from debug import Debug
from preprocessing import preprocessing

# debug configuration, set the printing format of numpy array
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})
debugger = Debug(
    debug_println_number = 20,
    is_debugging = True
    )

# IMPORTANT PARAMETERS
KERNEL = 11
SIGMA_S = 4
SIGMA_R = 0.1
INTENSITY_THRESHOLD = 93
EXTENSION_INTENSITY_THRESHOLD = 75
DARK_CONSTANT = 1.2

DATA_FOLDER = 'data'
TEST_IMG_PATH = 'test_data/large_star.jpg'
OUTPUT_FOLDER = 'outputs'


I = io.imread(TEST_IMG_PATH)

i_coords, j_coords = preprocessing(
    I = I,
    output_folder = OUTPUT_FOLDER,
    intensity_threshold = INTENSITY_THRESHOLD,
    extension_intensity_threshold = EXTENSION_INTENSITY_THRESHOLD,
    debugger = debugger
)

run_self_implemented_bilateral_filtering_on_all_non_star_pixels(
    I = I,
    i_coords = i_coords,
    j_coords = j_coords,
    kernel = KERNEL,
    sigma_s = SIGMA_S,
    sigma_r = SIGMA_R,
    dark_constant = DARK_CONSTANT,
    debugger = debugger,
    output_folder = OUTPUT_FOLDER
)

# # run the bilateral filtering implemented by my own
# run_self_implemented_bilateral_filtering_on_all_pixels(
#     I = I,
#     kernel = KERNEL,
#     sigma_s = SIGMA_S,
#     sigma_r = SIGMA_R,
#     dark_constant = DARK_CONSTANT,
#     debugger = debugger,
#     output_folder = OUTPUT_FOLDER
# )


