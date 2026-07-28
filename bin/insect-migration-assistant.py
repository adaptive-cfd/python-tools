#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:45:14 2026

@author: engels
"""

import inifile_tools
import argparse

parser = argparse.ArgumentParser(description='Insect siumulation migration assistant')
parser.add_argument('file', action='store', metavar='file', type=str, nargs=1, help='INI-File')
args = parser.parse_args()

inifile_tools.insect_INI_migration( args.file[0])