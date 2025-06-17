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

# this function just sets a sine wave onto the field because it's so nicely analytical
def INICOND_sine_wave(xyz, domain_size=[1, 1, 1], frequency=[1, 1, 1], offset=[0, 0, 0], amplitude=[1, 1, 1]):
    out = amplitude[0]*np.sin(frequency[0]*xyz[0]+offset[0]) + amplitude[1]*np.sin(frequency[1]*xyz[1]+offset[1])
    if len(xyz) == 3: out += amplitude[2]*np.sin(frequency[2]*xyz[2]+offset[2])
    return out

# this function sets a sine wave and gaussian blobs onto the field
def INICOND_sine_exp(xyz, domain_size=[1, 1, 1], amplitude_sine=[1,1,1], frequency=[1, 1, 1], offset_sine=[0, 0, 0], amplitude_exp=[1], sigma=[[1, 1, 1]], offset_exp=[[0, 0, 0]]):
    out = 0
    for bx in [-1, 0, 1]:
        for by in [-1, 0, 1]:
            if len(xyz) == 3: bz = [-1, 0, 1]
            else: bz = [0]

            for bz_val in bz:
                ixyz = np.array(xyz, dtype=float)
                # adjust the xyz coordinates to account for periodicity
                ixyz[0] += bx * domain_size[0]
                ixyz[1] += by * domain_size[1]
                if len(xyz) == 3: ixyz[2] += bz_val * domain_size[2]

                # calculate the output
                out += amplitude_sine[0]*np.sin(frequency[0]*ixyz[0]+offset_sine[0]) + amplitude_sine[1]*np.sin(frequency[1]*ixyz[1]+offset_sine[1])
                if len(xyz) == 3: out += amplitude_sine[2]*np.sin(frequency[2]*ixyz[2]+offset_sine[2])
                # add gaussian blob
                for i_b in range(len(amplitude_exp)):
                    if len(xyz) == 2: out += amplitude_exp[i_b]*np.exp( -( (ixyz[0]-offset_exp[i_b][0])**2/sigma[i_b][0]**2 + (ixyz[1]-offset_exp[i_b][1])**2/sigma[i_b][1]**2 ) )
                    else: out += amplitude_exp[i_b]*np.exp( -( (ixyz[0]-offset_exp[i_b][0])**2/sigma[i_b][0]**2 + (ixyz[1]-offset_exp[i_b][1])**2/sigma[i_b][1]**2 + (ixyz[2]-offset_exp[i_b][2])**2/sigma[i_b][2]**2 ) )
    return out

# this function gives the position at a specific dimension
def identity(xyz, dim=0):
    return xyz[dim]