from skimage import io
import numpy as np
from bilateral_filter import run_self_implemented_bilateral_filtering, Debug
from prepocessing import preprocessing

# debug configuration, set the print format of numpy array
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})
debugger = Debug(
    debug_println_number = 20,
    is_debugging = True
    )

# PARAMETERS
KERNEL = 7
SIGMA_S = 2
SIGMA_R = 0.0001

# convert raw to jpg, load data
DATA_FOLDER = 'data'
TEST_IMG_PATH = 'test_data/10000_8.jpg'
TEST_IMG_NAME = '10000_8'
OUTPUT_FOLDER = 'outputs'

I = io.imread(TEST_IMG_PATH)

preprocessing(I)

# # run the bilateral filtering implemented by my own
# run_self_implemented_bilateral_filtering(
#     I = I,
#     kernel = KERNEL,
#     sigma_s = SIGMA_S,
#     sigma_r = SIGMA_R,
#     debugger = debugger,
#     output_folder = OUTPUT_FOLDER
# )


