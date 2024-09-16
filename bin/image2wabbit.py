#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24-08-14 by JB

This script takes in an image file and creates a wabbit file for it. Used for denoising

"""
import sys, os, numpy as np
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools
from PIL import Image
import cv2
import scipy.io

import argparse, matplotlib.pyplot as plt

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Converts an image file to a wabbit input file.")
parser.add_argument('IMAGE', type=str, help='Input image file')
parser.add_argument('-o', '--output', type=str, help='Output location of H5 file, if not provided it is saved under same name as input.', default=None)
parser.add_argument('--bs', type=int, help='Block size.', default=20)
parser.add_argument('--level', type=int, help='Level for equidistant grid', default=5)
parser.add_argument('--max-level', type=int, help='Maxmium level for data', default=9)
group1 = parser.add_mutually_exclusive_group()
group1.add_argument('--nearest', action='store_true', help='Interpolate with nearest values.')
group1.add_argument('--linear', action='store_true', help='Interpolate linearly.')
group1.add_argument('--cubic', action='store_true', help='Interpolate cubic-wise.')
parser.add_argument('-d', '--display', action='store_true', help='Display resized data')
parser.add_argument('-n', '--noise', type=float, help='Add noise and provide std for gaussian noise to be added', default=-1)
parser.add_argument('--noise-type', type=str, help='Type of noise: "gaussian" or "uniform, defaults to uniform"', default="uniform")
parser.add_argument('--noise-seed', type=float, help='Set the seed of the noise, defaults to 0"', default=0)
parser.add_argument('--matlab', action='store_true', help="Output file as matlab matrix file as well.")


args = parser.parse_args()
if args.nearest: interpolation = cv2.INTER_NEAREST
elif args.linear: interpolation = cv2.INTER_LINEAR
elif args.cubic: interpolation = cv2.INTER_CUBIC
else:
    print("No Interpolation provided, choosing linear as default")
    interpolation = cv2.INTER_LINEAR

#------------------------------------------------------------------------------

number_blocks = 4**args.level

# prepare data
dim = 2
domain_size = [1, 1]
blocks = np.zeros([number_blocks, args.bs+1, args.bs+1])
treecode = np.zeros(number_blocks)
level = np.ones(number_blocks)*args.level
time = 0.0
iteration = 0

# prepare treecode
for i_b in range(number_blocks):
    # encoding is 1-based
    ix, iy = i_b//(2**args.level)+1, i_b%(2**args.level)+1
    tc = wabbit_tools.tc_encoding([ix, iy], level=args.level, max_level=args.max_level, dim=dim)
    treecode[i_b] = int(tc)

# load in file
image = Image.open(args.IMAGE)
grayscale_image = image.convert('L')
np_image = np.array(grayscale_image)
# rescale to fit wabbit sizes, we need + 1 as saved are redundant grids
resized_image = cv2.resize(np_image, np.array([1, 1])*2**args.level*args.bs + [1, 1], interpolation=interpolation)
# we need to invert y-direction, as for images 0 is top left and for wabbit 0 is bottom left, also 0 should be white so values are inverted too
resized_image = 255 - resized_image[::-1, :].astype(float)

# add noise
noise = 0
if args.noise != -1:
    mean = 0
    std_dev = args.noise
    np.random.seed(args.noise_seed)
    if "gauss" in args.noise_type:
        noise = np.random.normal(mean, std_dev, resized_image.shape)
    elif "uniform" in args.noise_type:
        # for uniformly distributed numbers the variance is (b-a)^2/12 for the interval [a,b]
        noise = np.random.random(size=resized_image.shape)*args.noise*np.sqrt(12)
    else:
        print(f"I do not know this type of noise: {args.noise_type}. exiting ..")
        sys.exit(1)
    resized_image = resized_image + noise

if args.display:
    plt.figure(1, figsize=[7,7])
    plt.imshow(resized_image)
    plt.show()

# fill blocks array
for i_b in range(number_blocks):
    ix, iy = i_b//(2**args.level) * args.bs, i_b%(2**args.level) * args.bs

    # transcribe part of array
    blocks[i_b, :, :] = resized_image[iy:iy+args.bs+1, ix:ix+args.bs+1]

# create wabbit file
w_obj = wabbit_tools.WabbitHDF5file()
w_obj.fill_vars(domain_size, blocks, treecode, level, time, iteration, max_level=args.max_level)
if args.output == None:
    new_name = args.IMAGE.replace(args.IMAGE[args.IMAGE.rfind("."):], ".h5")
else:
    if args.output.endswith("h5"):
        new_name = args.output
    else:
        new_name = args.output + ".h5"
w_obj.write(new_name)

# create matlab file
if args.matlab:
    if args.output == None:
        new_name = args.IMAGE.replace(args.IMAGE[args.IMAGE.rfind("."):], ".mat")
    else:
        if args.output.endswith("mat"):
            new_name = args.output
        else:
            new_name = args.output + ".mat"
    scipy.io.savemat(new_name, {'image':resized_image[:-1,:-1]-noise[:-1,:-1], 'noise': noise[:-1,:-1]})