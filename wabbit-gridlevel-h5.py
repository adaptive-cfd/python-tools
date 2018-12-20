#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:06:35 2018

@author: engels
"""


import wabbit_tools
import sys


time, x0, dx, box, data, treecode = wabbit_tools.read_wabbit_hdf5( sys.argv[1] )

data = wabbit_tools.overwrite_block_data_with_level(treecode, data)

wabbit_tools.write_wabbit_hdf5(sys.argv[2], time, x0, dx, box, data, treecode)