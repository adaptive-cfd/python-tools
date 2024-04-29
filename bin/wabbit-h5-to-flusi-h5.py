#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:29:28 2019

@author: engels
"""
import sys, os
sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_dense_error_tools

wabbit_dense_error_tools.to_dense_grid( sys.argv[1], sys.argv[2], dim=3 )