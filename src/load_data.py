import subprocess
from skimage import io
import numpy as np
import glob
import os

def modify_path_format(old_path):
    '''only work for paths with only one \\
        convert data\\x.nef to data/x.nef'''
    slash_idx = old_path.index('\\')
    return old_path[:slash_idx] + '/' + old_path[slash_idx + 1:]

def convert_raw_to_tiff_folder(data_folder):
    '''
    note: data\\x.nef format doesn't work, the format should be data/x.nef in dcraw.
    Thus I'm modifying value of path here
    '''
    raw_paths = sorted(glob.glob(os.path.join(data_folder, "*.nef")))
    for path in raw_paths:
        path = modify_path_format(path)
        print('RAW2TIFF: ' + path)
        # dcraw_cmd = ['wsl', 'dcraw', '-w', '-q', '3', '-o', '1', '-4', '-T', path]
        dcraw_cmd = ['wsl', 'dcraw', '-q', '3', '-o', '1', '-4', '-T', path]
        subprocess.check_output(dcraw_cmd)

def convert_raw_to_tiff_file(file_path):
    '''
    file_path format should be data/x.nef in dcraw.
    '''
    print('RAW2TIFF: ' + file_path)
    # dcraw_cmd = ['wsl', 'dcraw', '-w', '-q', '3', '-o', '1', '-4', '-T', path]
    dcraw_cmd = ['wsl', 'dcraw', '-q', '3', '-o', '1', '-4', '-T', file_path]
    subprocess.check_output(dcraw_cmd)



def load_tiffs(tiff_folder):
    '''
    For the data I use
    shape: (4040, 6064, 3)
    '''
    tiff_paths = sorted(glob.glob(os.path.join(tiff_folder, "*.tiff")))
    h, w, c = io.imread(tiff_paths[0]).shape
    res = np.zeros((len(tiff_paths), h, w, c))
    for i in range(len(tiff_paths)):
        res[i, ...] = io.imread(tiff_paths[i]) / (2 ** 16 - 1)
    np.save("data/tiffs.npy", res)
    return res

TIFF_FOLDER = 'data'
load_tiffs(TIFF_FOLDER)

def load_one_tiff(tiff_folder):
    tiff_paths = sorted(glob.glob(os.path.join(tiff_folder, "*.tiff")))    
    np.save("data/one_tiff.npy", io.imread(tiff_paths[0]))
    print(tiff_paths[0])

# load_one_tiff(TIFF_FOLDER)

# I = np.load("data/one_tiff.npy")