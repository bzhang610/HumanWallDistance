import numpy as np
from utils import *

if __name__ == "__main__":
    '''
    Load the input depth image, run functions, and print the output
    '''
    from sys import argv
    img_file = argv[1]                  # get file name from input
    depth_img = np.loadtxt(img_file)    # load data
    x,y,z = get_w_coords(depth_img)     # get world frame coordinates
    X,Y,Z = process_coords(x,y,z)       # filter coordinates
    result = BB_gap(X,Y,Z)              # calculate gap's side and size
    print(result[0],result[1])
