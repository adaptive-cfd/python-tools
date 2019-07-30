#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:54:45 2019

@author: engels
"""

def spectrum(u):
    import numpy as np
#    x = np.linspace(0, 2*np.pi, 128, endpoint=False)
#    X,Y = np.meshgrid(x,x)
#    u = np.sin(10.0*X) + np.sin(10.0*Y)

    if (len(u.shape) is not 2):
        print(u.shape)
        raise ValueError('wrong dimension: Spectrum is currently for 2D data only.')

    uk = np.fft.fft2(u)

    N = u.size

    ek = np.abs( uk/N )**2.0
    ek = 0.5 * ek

    dimMax = np.max( u.shape )
    k = np.fft.fftfreq(dimMax) * dimMax


    # Only consider one half of spectrum (due to symmetry)
    halfDim = int( np.floor(dimMax/2) + 1 )

    kx, ky = np.meshgrid( k, k )
    K = np.sqrt( kx**2 + ky**2 )
    K = np.round( K )

    # allocate spectrum
    EK = np.zeros( halfDim );

    # loop over wavenumbers
    for ik in range(0, halfDim):
        # sum up all fourier coefficients that have the same wavenumber k
#        ek = np.ones(ek.shape)
        EK[ik] = np.sum( ek[ np.abs(K-float(ik)) <= 1.0e-10 ])


    k = np.asarray( range(0, halfDim ) )

#    print( np.sum(EK) )
#    print( np.sum(u**2) )
#    print( np.sum(ek) )

    return k, EK

def fft2_resample(u, res):
    """
    Resampling of 2D data field in Fourier space.

     Input:
         - u ... data field, two dimensional, supposed to be square matrix N by N
         - res ... new resolution, i.e. output data is res by res
     Output:
         - u ... now the resampled data with size res by res

     Notes:
         If res > N, then upsampling using zero-padding in Fourier space is performed.
         If res < N, then downsampling (cropping of Fourier coefficients) is performed.
         If res = N, the input is returned unchanged.

         If your data is single precision (i.e. read from standard FLUSI fields)
         and converted to double precision, very large upsampling can amplify the noise
         that comes from the missing precision In that case, better use single precision.
    """
    import numpy as np
    nold = np.asarray(u.shape)
    
    if (np.all(res > nold)):
        u = fft2_upsample(u, res)
    elif (res < u.shape[0]):
        u = fft2_downsample(u, res)

    return u



def fft2_upsample(u, resolution):
    """
    u is a 2d field
    resolution of the upsampled field [nx,ny] or just n 

    """
    import numpy as np
#    import matplotlib.pyplot as plt
    

    if not isinstance(resolution, (list, tuple)):
       res = [resolution]
    else:
        res = resolution
    res=np.asarray(res)
    
    E_in = np.sum(u**2)/np.float64(u.size)

    uk = np.fft.fft2(u)
    uk = np.fft.fftshift( uk )
    
    nold = u.shape

    # zero-pad
    n = np.asarray ( (res - nold) / 2, dtype=np.dtype(int) )
    
    uk = np.pad( uk, ((n[0],n[0]),(n[1],n[1])), 'constant')

    uk = np.fft.ifftshift( uk )

    # renormalize (array.size is nx*ny)
    uk = uk * ( np.float64(uk.size) / np.float64(u.size) )

    # goback to x-space
    u2 =  np.real( np.fft.ifft2( uk ) )

    E_out = np.sum(u2**2)/np.float64(u2.size)

#    plt.figure()
#    plt.pcolormesh(u2)
#    plt.colorbar()


#    #--------------------------
#
#    uk2 = np.fft.rfft2( u )
#    uk2 = np.fft.fftshift( uk2, axes=0  )
#
#    nold = u.shape[0]
#
#    # zero-pad
#    n = int( (res - nold) / 2 )
#    uk3 = np.pad( uk2, ((n,n),(0,n)), 'constant')
#
#    uk3 = np.fft.ifftshift( uk3, axes=0 )
#
#    # renormalize (array.size is nx*ny)
##    uk3 = uk3 * ( float(2.0*uk3.size)/float(u.size) )
#
#    # goback to x-space
#    u22 =  np.real( np.fft.irfft2( uk3 ) )
#
#    u22 *= (np.sum(u**2)) / (np.sum(u22**2))
#
#    plt.figure()
#    plt.pcolormesh(u22)
#    plt.colorbar()
#
#    raise ValueError

    print( " fft: upsampling: energy was=%20.15e is now=%20.15e " % (E_in, E_out) )
    print(" New Resolution:", u2.shape)
    print(" Old Resolution:", nold)
    print( " delta_E=%20.15e" % (E_in - E_out) )
    print( " delta_E=%20.15e" % ((E_in - E_out)/E_in) )
#    print( "rfft: upsampling: energy was=%20.15e is now=%20.15e (from %i to %i points)" % (np.sum(u**2)/u.size, np.sum(u22**2)/u2.size, nold, res))

    return u2

#import insect_tools
#import numpy as np
#time_ref, box_ref, origin_ref, u = insect_tools.read_flusi_HDF5( '/home/engels/dev/WABBIT4-new-physics/three-vortices/equidistant_convergence_RHS_only/../spectral_acm/Re20k_c0_10_gamma1/solution_2048/ux_020000.h5', dtype=np.float64 )
#
#fft2_upsample(u[0,::8,::8], 3072)

def fft2_downsample(u, res):
    import numpy as np

    uk = np.fft.fft2(u)
    uk = np.fft.fftshift( uk )

    nold = u.shape[0]

    # cropping
    n = int ( (nold - res) / 2 )
    uk = uk[n:-n,n:-n]

    uk = np.fft.ifftshift( uk )

    # renormalize (array.size is nx*ny)
    uk = uk * ( float(uk.size)/float(u.size) )

    # goback to x-space
    u2 =  np.real( np.fft.ifft2( uk ) )

    print( "downsampling: energy was=%15.10e is now=%15.10e (loss is desired!)" % (np.sum(u**2)/u.size, np.sum(u2**2)/u2.size))

    return u2




