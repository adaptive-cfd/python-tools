#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:02:39 2018

@author: engels
"""

import wabbit_tools
import sys


treecode1 = wabbit_tools.read_treecode_hdf5(sys.argv[1])
treecode2 = wabbit_tools.read_treecode_hdf5(sys.argv[2])

print(wabbit_tools.compare_two_grids(treecode1, treecode2))