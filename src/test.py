from skimage import io
import numpy as np
import cv2

I1 = io.imread('outputs/res_small_star/I.jpg')
# I2 = io.imread('outputs/res_small_star/I_threshold_90.png')
# I3 = io.imread('outputs/res_small_star/I_threshold_ex_90_75.png')
# I4 = io.imread('outputs/res_small_star/preserve_stars_I_filtered_15_4000_100_150.png')

out_f = 'outputs/res_small_star/smaller'

s_x, e_x = 0, 200
s_y, e_y = 250, 480

# io.imsave(f'{out_f}/I.png', I1[s_x:e_x, s_y:e_y])
# io.imsave(f'{out_f}/I_threshold_90.png', I2[s_x:e_x, s_y:e_y])
# io.imsave(f'{out_f}/I_threshold_ex_90_75.png', I3[s_x:e_x, s_y:e_y])
# io.imsave(f'{out_f}/preserve_stars_I_filtered_15_4000_100_150.png', I4[s_x:e_x, s_y:e_y])

# cv_filtered = np.zeros_like(I1.shape)
cv_filtered = cv2.bilateralFilter(I1, 15, 80, 80)
io.imsave(f'{out_f}/cv_filtered.png', cv_filtered[s_x:e_x, s_y:e_y])