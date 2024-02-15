#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:14:11 2021

@author: engels
"""

# usage :
    # replace_ini_value.sh PARAMS.ini Time time_max 3.0
    

import argparse

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

found_section, found_keyword = False, False
i = 0


with open(file, 'r') as f:
    # read a list of lines into data
    data = f.readlines()
   
    

# loop over all lines
for line in data:
    line = line.lstrip().rstrip()
    if len(line) > 0:
        if line[0] != ';':
            if '['+section+']' in line:
                found_section = True
                
            if ';' in line:
                line_nocomments = line[0:line.index(';')]
            else:
                line_nocomments = ""
            
                
            if '[' in line_nocomments and ']' in line_nocomments and not '['+section+']' in line_nocomments and found_section:
                # left section again
                found_section = False         
                break
                
            if keyword+'=' in line and found_section:
                # found keyword in section
                found_keyword = True
                old_value = line[ line.index(keyword+"="):line.index(";") ]
 
                line = line.replace(old_value, keyword+'='+new_value)
                data[i] = line+'\n'
                
                print("changed: "+old_value+" to: "+keyword+'='+new_value)
                break
    i += 1
   
                
if found_keyword:
    # .... and write everything back
    with open(file, 'w') as f:
        f.writelines( data )
    
                    
                    
                    
