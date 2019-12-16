#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:41:23 2019

@author: tommy
"""
import numpy as np

def Dper(N, h, stencil):
    # convert to numpy array, if not already the case
    stencil = np.asarray(stencil)

    ns = stencil.shape[0]
    pos = np.asarray( np.arange( -np.floor(ns/2), np.floor(ns/2)+1, 1) )

    D = np.zeros( (N,N) )

    for rows in range(N):
       pos1 = rows + pos

       for j in range(ns):
           pos1 = periodic_index( pos[j]+rows, N )

           D[rows,pos1] = stencil[j]

    # apply the spacing
    return D / h


def periodic_index(i,N):
    if (i<0):
        i = i+N

    if (i>N-1):
        i = i-N

    return int(i)


def RK4(u, rhs, dt, params):
    k1 = rhs(u, params)

    u2 = u + dt/2.0 * k1
    k2 = rhs(u2, params)

    u3 =  u + dt/2.0 * k2
    k3 = rhs(u3, params)

    u4 = u + dt*k3
    k4 = rhs(u4, params)

    return( u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4) )



def smoothstep( x, t, h):
    y = np.zeros( x.shape )
    y[ x<=t-h ] = 1.0
    y[ x>=t+h ] = 0.0
    y[ np.abs(x-t) < h ] = 0.5 * ( 1.0 + np.cos((x[ np.abs(x-t) < h ]-t+h)*np.pi/(2.0*h)) )

    return y