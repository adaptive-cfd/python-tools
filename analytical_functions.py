#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# this script gives us analytical functions that we need to study results
def INICOND_convdiff_blob(xyz, domain_size=[1, 1, 1], blob_pos=[[0.75, 0.5, 0]], blob_width=[0.01], blob_strength=[1]):

    # we assume domain is periodic here so we loop over imaginary neighbouring domains to add periodic effect as well
    out = 0
    for i_blob in range(len(blob_strength)):
        x = xyz[0] - blob_pos[i_blob][0]
        y = xyz[1] - blob_pos[i_blob][1]
        if len(xyz) == 3: z = xyz[2] - blob_pos[i_blob][2]
        else: z = 0
        for i_x in [-1, 0, 1]:
            for i_y in [-1, 0, 1]:
                if len(xyz) == 3: r_z = [-1, 0, 1]
                else: r_z = [0]

                for i_z in r_z:
                    out += blob_strength[i_blob]*np.exp( -( (x + i_x*domain_size[0])**2 + (y + i_y*domain_size[1])**2 + (z + i_z*domain_size[2])**2 ) / blob_width[i_blob] )
    return out