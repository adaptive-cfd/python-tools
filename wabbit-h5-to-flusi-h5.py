#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:29:28 2019

@author: engels
"""
import wabbit_tools
import sys

wabbit_tools.to_dense_grid( sys.argv[1], sys.argv[2], dim=3 )