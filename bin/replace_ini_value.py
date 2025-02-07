#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:14:11 2021

@author: engels
"""

# usage :
    # replace_ini_value.sh PARAMS.ini Time time_max 3.0
    
import argparse
import inifile_tools

parser = argparse.ArgumentParser(description='Replace a setting in a WABBIT/FLUSI style INI file')
parser.add_argument('file', action='store', metavar='file', type=str, nargs=1, help='INI-File')
parser.add_argument('section', metavar='section', type=str, nargs=1, help='Section inside the file, eg Time')
parser.add_argument('keyword', metavar='keyword', type=str, nargs=1, help='keyword (parameter')
parser.add_argument('new_value', metavar='new_value', type=str, nargs=1, help='new value')

args = parser.parse_args()
    
file = args.file[0]
section = args.section[0]
keyword = args.keyword[0]
new_value = args.new_value[0]

inifile_tools.replace_ini_value(file, section, keyword, new_value)