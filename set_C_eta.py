#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:29:24 2021

@author: engels
"""


import argparse

parser = argparse.ArgumentParser(description='Replace a setting in a WABBIT/FLUSI style INI file')
parser.add_argument('file', action='store', metavar='file', type=str, nargs=1, help='INI-File')
parser.add_argument('Keta', metavar='K_eta', type=float, nargs=1, help='Value for K-eta')
parser.add_argument('Ntau', metavar='N_tau', type=float, nargs='?', help='Value for N_tau (default: 20.0)',default=20.0, )

args = parser.parse_args()
    
file = args.file[0]
K_eta = args.Keta[0]
Ntau = args.Ntau


print("K_eta=%f N_tau=%f" % (K_eta, Ntau))

import wabbit_tools

Bs   = wabbit_tools.get_ini_parameter(file, 'Blocks', 'number_block_nodes', vector=True)
Jmax = wabbit_tools.get_ini_parameter(file, 'Blocks', 'max_treelevel')
L    = wabbit_tools.get_ini_parameter(file, 'Domain', 'domain_size', vector=True)
nu   = wabbit_tools.get_ini_parameter(file, 'ACM-new', 'nu')

dx = L[0]*(2**-Jmax)/(Bs[0]-1)

C_eta = (K_eta*dx)**2 / nu

import os
command = "replace_ini_value.py "+file+" VPM C_eta %e" % (C_eta)
os.system(command)

use_sponge = wabbit_tools.get_ini_parameter(file, 'Sponge', 'use_sponge', dtype=bool, default=False)

if (use_sponge):
    L_sponge = wabbit_tools.get_ini_parameter(file, 'Sponge', 'L_sponge')
    C0 = wabbit_tools.get_ini_parameter(file, 'ACM-new', 'c_0')
    
    C_sponge = L_sponge / (Ntau*C0)
    
    command = "replace_ini_value.py "+file+" Sponge C_sponge %e" % (C_sponge)
    os.system(command)

    