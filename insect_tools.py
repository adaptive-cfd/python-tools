#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:11:47 2017

@author: engels
"""


import numpy as np
import numpy.ma as ma
import glob
from warnings import warn

def change_color_opacity(color, alpha):
    import matplotlib
    color = list(matplotlib.colors.to_rgba(color))
    color[-1] = alpha
    return color

# I cannot learn this by heart....
def change_figure_dpi(dpi):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = dpi
    
icolor = 0
def get_next_color(ax=None):
    import matplotlib.pyplot as plt
    
    # as of 01/2024, the method below is not available anymore, because PROP_CYCLER
    # has vannished. This current version is a hack.
    
    # if ax is None:
    #     ax = plt.gca()
    # return next(ax._get_lines.prop_cycler)['color']
    
    global icolor
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    this_color = colors[icolor]
    icolor += 1
    if icolor>len(colors)-1:
        icolor=0

    return this_color

# It is often the case that I want to plot several lines with the same color, eg Fx, Fy
# from a specific run. This function gets the last used color for this purpose.
def get_last_color(ax=None):
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
    color = ax.lines[-1].get_color()
    
    return color

def reset_colorcycle( ax=None ):
    global icolor
    icolor = 0

def get_next_marker():
    import itertools
    marker = itertools.cycle(('o', '+', 's', '>', '*', 'd', 'p'))
    return next(marker)


def statistics_stroke_time_evolution( t, y, plot_indiv_strokes=True, N=1000, tstroke=1.0, plot_raw_data=False, color='k', marker='d'):    
    """
    Perform statistics over periodic data.
    Often, we have data for several strokes, say 0 <= t <= 10. This function assumes
    a period time T=1.0, divides the data into chunks corresponding to each of the strokes,
    then computes the average time evolution as well as standard deviation among all strokes.
    
    The data is divided into cycles and (linearily) interpolated to an equidistant time
    grid. Ths grid is equidistant and sampled using N=1000 data points.
    
    Input:
    ------
    
        t: vector, float
           full time vector (say 0<t<10.0)
        y: vector, float
           The actual data to be analyzed (e.g., drag or power)
       plot_indiv_strokes: logical
           If true, each individual stroke is plotted in the current figure.
       
    
    Output:
    -------
        time, y_avg, y_std
        
        
    Todo:
    -----
        The code does not check for incomplete cycles, say 0<t<4.57 is a problem
    """

    import matplotlib.pyplot as plt
    
    # all data is interpolated to an equidistant time grid
    time = np.linspace(0.0, 1.0, num=N, endpoint=False)
    
    # start and end time of data
    t0, t1 = t[0], t[-1]
    
    # how many cycles are there?
    nstrokes = int( np.round( (t1-t0) / tstroke) )
    
    y_interp = np.zeros( (nstrokes, time.shape[0]) )
    
    for i in range(nstrokes):
        # linear interpolation
        y_interp[i,:] = np.interp( time+float(i)*tstroke, t, y)
        
        if plot_indiv_strokes:
            # plot the linear interpolated data for this stroke
            plt.plot(time, y_interp[i,:], color='k', linewidth=0.5)
            
        if plot_raw_data:
            # plot raw (non-interpolated) data points 
            # helpful if data contains holes (NAN) and isolated datapoints, because 
            # we need at least two non-NAN points for linear interpolation
            mask = np.zeros( t.shape, dtype=bool)
            mask[ t>=float(i)*tstroke ]   = True
            mask[ t>=float(i+1)*tstroke ] = False
            plt.plot( t[mask]-float(i)*tstroke, y[mask], marker=marker, color=color, mfc='none', linewidth=0.5, linestyle='none')
            
            
    
    y_avg = np.nanmean(y_interp, axis=0)
    y_std = np.nanstd(y_interp, axis=0)
    
    return time, y_avg, y_std



def plot_errorbar_fill_between(x, y, yerr, color=None, label="", alpha=0.25, fmt='o-', linewidth=1.0, ax=None):
    """
    Plot the data y(x) with a shaded area for error (e.g., standard deviation.)
    
    This is a generic function often used. It does only the plotting, not the computation 
    part.
    
    Input:
    ------
    
        x: vector, float
           data location (usually time)
        y: vector, float
           The actual data to be plotted (e.g., drag or power)
        yerr: vector_float:
            The shaded area will be between y-yerr and y+yerr for all x
        color: color_object
            Well. guess.
        alpha: float (0<=alpha<=1.0)
            The degree of transparency of the shaded area.
       
    
    Output:
    -------
        plotting to figure
        
    """
    import matplotlib.pyplot as plt
    import matplotlib
    
    if ax is None:
        ax = plt.gca()

    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)

    if color is None:
        color = get_next_color()
    color = np.asarray( matplotlib.colors.to_rgba(color) )

    # first, draw shaded area for dev
    color[3] = alpha
    ax.fill_between( x, y-yerr, y+yerr, color=color, alpha=alpha )

    # then, avg data
    color[3] = 1.00
    ax.plot( x, y, fmt, label=label, color=color, mfc='none', linewidth=linewidth )



# chunk a string, regardless of whatever delimiter, after length characters,
# return a list
def chunkstring(string, length):
    return list( string[0+i:length+i] for i in range(0, len(string), length) )

def cm2inch(value):
    return value/2.54

def deg2rad(value):
    return value*np.pi/180.0

def ylim_auto(ax, x, y):
   # ax: axes object handle
   #  x: data for entire x-axes
   #  y: data for entire y-axes
   # assumption: you have already set the x-limit as desired
   lims = ax.get_xlim()
   i = np.where( (x > lims[0]) &  (x < lims[1]) )[0]
   ax.set_ylim( y[i].min(), y[i].max() )

# set axis spacing to equal by modifying only the axis limits, not touching the
# size of the figure
def axis_equal_keepbox( fig, ax ):
    # w, h = fig.get_size_inches()
    
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    w, h = bbox.width, bbox.height

    
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    if (x2-x1)/w > (y2-y1)/h:
        # adjust y-axis
        l_old = (y2-y1)
        l_new = (x2-x1) * h/w
        ax.set_ylim([ y1-(l_new-l_old)/2.0, y2+(l_new-l_old)/2.0])
    else:
        # adjust x-axis
        l_old = (x2-x1)
        l_new = (y2-y1) * w/h
        ax.set_xlim([ x1-(l_new-l_old)/2.0, x2+(l_new-l_old)/2.0])



# read pointcloudfile
def read_pointcloud(file):
    data = np.loadtxt(file, skiprows=1, delimiter=' ')
    if data.shape[1] > 6:
        data = np.delete( data, range(3,data.shape[1]-3) , 1)
    print(data.shape)
    return data

def write_pointcloud(file, data, header):
    write_csv_file( file, data, header=header, sep=' ')





def load_t_file( fname, interp=False, time_out=None, return_header=False,
                verbose=True, time_mask_before=None, T0=None, keep_duplicates=False, 
                remove_outliers=False, optimized_loading=False ):
    """
    Read in an ascii *.t file as generated by flusi or wabbit.
    Returns only unique times (if the simulation ran first until t=3.0 and then is resumed from
    t=2.0, the script will return the entries 2.0<t<3.0 only one time, even though they exist twice 
    in the file)
    
    Input:
    ------
    
         fname: string
             filename to be read
         interp: bool
             is interpolation used or do we just read what is in the file?
         time_out: numpy array
             if interp=True, ou can specify to what time vector we interpolate.
         return_header: bool
             If true, we return the header line of the file as list of strings
         verbose: bool
             be verbose or not.
         keep_duplicates: bool
             remove lines with non-unique time stamps or not? if True, only one is kept. The code takes care that the LAST
             of non-unique time stamps is returned (i.e. if the simulation runs until t=20.0 and is then resumed from t=19.0
             the values of that last resubmission are returned) 
         remove_outliers: bool
             An ugly function to remove obvious outliers that frequently appear for wabbit data at the saving intervals, when the time
             stamp dt gets very small. ugly for presentations but harmless for the data.
         time_mask_before: float
             if set, we return a masked array which masks data before this time. note some routines do not like masked arrays
         T0: list of floats or single float
             can be either one value or two values. In the former case, we extract data t>=T0
             in the latter T0[0]<=t<=T0[1]
       
    
    Output:
    -------
        data: numpy array
            the actual matrix stored in the file, possibly interpolated
        header: list of strings
            the columns headers, if return_header=True. If we did not find a heade in the file, the list is empty.
        
    """ 
    import os

    if verbose:
        print('reading file %s' %fname)

    # does the file exists?
    if not os.path.isfile(fname):
        raise ValueError('load_t_file: file=%s not found!' % (fname))

    # does the user want the header back?
    if return_header:
        # read header line
        f = open(fname, 'r')
        header = f.readline()
        # a header is a comment that begins with % (not all files have one)
        if "%" in header:
            # remove comment character
            header = header.replace('%',' ')
            # convert header line to list of strings
            header = chunkstring(header, 16)
            f.close()

            # format and print header
            for i in range(0,len(header)):
                # remove spaces (leading+trailing, conserve mid-spaces)
                header[i] = header[i].strip()
                # remove newlines
                header[i] = header[i].replace('\n','')
                if verbose:
                    print( 'd[:,%i] %s' % (i, header[i] ) )
        else:
            print('You requested a header, but we did not find one...')
            # return empty list
            header = []

    #--------------------------------------------------------------------------
    # read the data from file
    #--------------------------------------------------------------------------
    # 18/12/2018: we no longer directly use np.loadtxt, because it sometimes fails
    # if a run has been interrupted while writing the file. In those cases, a line
    # sometimes contains less elements, trip-wiring the loadtxt function
    #
    # old call:
    #
    #    data_raw = np.loadtxt( fname, comments="%")
    
    import datetime
    import time
    
    #--------------------------------------------------------------------------
    # RAW data reading
    #--------------------------------------------------------------------------
    
    # optimized loading - convert to NPZ on first call, then read that
    reloading_required = False
    if not os.path.isfile(fname+'.npz'):
        # file does not exist - reloading is required.
        reloading_required = True
    else:
        # file exists        
        if datetime.datetime.fromtimestamp(os.path.getmtime(fname+'.npz')) < datetime.datetime.fromtimestamp(os.path.getmtime(fname)):
            # *.npz file is older than source *.t file - reloading is required
            reloading_required = True

    start = time.time()
    
    if not optimized_loading or reloading_required:
        ncols = None
        # initialize file as list (of lists)
        dat = []
        with open( fname, "r" ) as f:
            # loop over all lines
            for line in f:
                if not '%' in line:
                    # turn line into list
                    tmp = line.split()
                    # did we already figure out how many cols the file has?
                    if ncols is None:
                        ncols = len(tmp)
                        
                    # try if we can convert the list entries to float
                    # sometimes suff like '-8.28380559-104' happens and that cannot be
                    # converted. in this case, we set zero
                    for j in range(len(tmp)):
                        try:
                            dummy = float(tmp[j])
                        except:
                            print( "WARNING %s cannot be converted to float, returning zero instead" % (tmp[j]) )
                            tmp[j] = "0.0"
    
                    if len(tmp) == ncols:
                        dat.append( tmp )
                    else:
                        dat.append( tmp[0:ncols] )
    
        # convert list of lists into an numpy array
        data_raw = np.array( dat, dtype=float )
        
        if optimized_loading:
            # when otpimization is used, save converted data to binary npz file to read that next time.
            np.savez(fname+'.npz', data_raw=data_raw)
    else: 
        if verbose:
            print('Optimized reading from pre-converted *.npz file')
        data_raw = np.load(fname+'.npz')['data_raw']
        
    if verbose:
        print('Completed in:', round(time.time() - start, 4), 'seconds')
        
    #--------------------------------------------------------------------------


    if len(data_raw.shape) == 1:
        return None

    nt_raw, ncols = data_raw.shape

    # retain only unique values (judging by the time stamp, so if multiple rows
    # have exactly the same time, only one of them is kept)
    if not keep_duplicates:
        time_raw = data_raw[:,0]
        
        # old code:
#        dummy, unique_indices = np.unique( time_raw, return_index=True )        
#        data = np.copy( data_raw[unique_indices,:] )
        
        it_unique = np.zeros( time_raw.shape, dtype=bool )
        
        # skip first time stamp
        for it in np.arange( 1, nt_raw ):
            t0, t1 = time_raw[it-1], time_raw[it]
            if t1 < t0:
                # we have found a jump. now, figure out the index where we duplicate time stamps began
                istart = np.argmin( np.abs(time_raw[0:it-1]-t1) )                
                tstart = time_raw[istart]
                
                if (abs(tstart - t1) >= 1.0e-3):
                    print("""Warning. 
                          In %s we found a jump in time (duplicate values) t[%i]=%f t[%i]=%f but the nearest
                          time in the past is t[%i]=%f so it may not be duplicates""" % (fname, it-1, t0, it, t1, istart, tstart) )
                else:
                    # legit jump (duplicate values)
                    it_unique[istart:it] = False
                    it_unique[it] = True
            else:
                # no jump, seems to be okay for now, but is maybe removed later
                it_unique[it] = True
                    
        data = data_raw[ it_unique, :].copy()
        
    else:
        data = data_raw.copy()



    if remove_outliers:
        it_unique = np.zeros( data.shape[0], dtype=bool )
        
        for it in np.arange(1, data.shape[0]-1): # skips first and last point
            # actual data (all columns of file)
            line = data[it,:]
            # linear interpolation using neighbor values (all columns of file)
            line_interp = (data[it-1,:] + data[it+1,:]) *0.5

            # absolute and relative "errors" (difference to linear interpolation)
            err_abs = np.abs(line - line_interp)
            err_rel = err_abs.copy()
            err_rel[ np.abs(line_interp) >= 1.0e-10 ] /= np.abs(line_interp)[ np.abs(line_interp) >= 1.0e-10 ]
            
            # use absolute value where the magnitude is smaller than 1e-7
            err = err_rel
            err[ err_abs <= 1.0e-7 ] = err_abs[ err_abs <= 1.0e-7 ]            
            
            # remove time steps that are detected as 'outliers'
            if (np.max(err)>0.25):
                it_unique[it] = False
            else:
                it_unique[it] = True
                
        # sometimes the last point is a problem....
        # note we cannot assume periodicity, so we use a one-sided difference stencil.
        err_abs = np.abs(data[-2,:]-data[-1,:])
        err_rel = err_abs.copy()
        for ic in range( data.shape[1]):
            # normalize by 2nd to last point not the last one in case its super large
            if np.abs(data[-2,ic]) > 1e-10:
                err_rel[ic] /= np.abs(data[-2,ic])
        
        # use absolute value where the magnitude is smaller than 1e-7
        err = err_rel
        err[ err_abs <= 1.0e-7 ] = err_abs[ err_abs <= 1.0e-7 ]

        if (np.max(err)>1.5):
            it_unique[-1] = False
        else:
            it_unique[-1] = True
                
        data = data[ it_unique, :].copy()
        
        

    if T0 is not None:
        if type(T0) is list and len(T0)==2:
            # extract time instants between T0[0] and T0[1]
            i0 = np.argmin( np.abs(data[:,0]-T0[0]) )
            i1 = np.argmin( np.abs(data[:,0]-T0[1]) )
            # data = np.copy( data[i0:i1,:] )
            data = data[i0:i1,:].copy()

        else:
            # extract everything after T0
            i0 = np.argmin( np.abs(data[:,0]-T0) )
            # data = np.copy( data[i0:,:] )
            data = data[i0:,:].copy()


    # info on data
    nt, ncols = data.shape
    
    if verbose:
        print( 'nt_unique=%i nt_raw=%i ncols=%i' % (nt, nt_raw, ncols) )

    # if desired, the data is interpolated to an equidistant time grid
    if interp:
        if time_out is None:
            # time stamps as they are in the file, possibly nont equidistant
            time_in = np.copy(data[:,0])
            # start & end time
            t1 = time_in[0]
            t2 = time_in[-1]
            # create equidistant time vector
            time_out = np.linspace( start=t1, stop=t2, endpoint=True, num=nt )
        # equidistant time step
        dt = time_out[1]-time_out[0]
        if verbose:
            print('interpolating to nt=%i (dt=%e) points' % (time_out.size, dt) )

            if data[0,0] > time_out[0] or data[-1,0] < time_out[-1]:
                print('WARNING you want to interpolate beyond bounds of data')
                print("Data: %e<=t<=%e Interp: %e<=t<=%e" % (data[0,0], data[-1,0], time_out[0], time_out[-1]))

        data = interp_matrix( data, time_out )

    # return data
    if return_header:
        return data, header
    else:
        return data



def stroke_average_matrix( d, tstroke=1.0, t1=None, t2=None, force_fullstroke=True ):
    """
    Return a matrix of cycle-averaged values from a array from a  *.t file.
    
    Input:
    ------
    
        d: np.ndarray, float
            Data. we assume d[:,0] to be time.
        tstroke: scalar, float
            length of a cycle
        force_fullstrokes: scalar, bool
            If you do not pass t1, t2 then we can round the data time: if the first
            data point is at 0.01, it will rounded down to 0.0.
        t1: scalar, float
            first time instant to begin averaging from
        t2: scalar, float
            last time instant to end averaging at. This can be useful if the very last
            time step is not precisely the end of a stroke (say 2.9999 instead of 3.000)
    
    Output:
    -------
        D: matrix
            stroke averages in a matrix
    """
    # start time of data
    if t1 is None:
        t1 = d[0,0]

    # end time of data
    if t2 is None:
        t2 = d[-1,0]
        
    if force_fullstroke:
        t1 = np.floor( t1/tstroke )*tstroke
        t2 = np.ceil( t2/tstroke )*tstroke

    # will there be any strokes at all?
    if t2-t1 < tstroke:
        print('warning: no complete stroke present, not returning any averages')

    if t1 - np.round(t1) >= 1e-3:
        print('warning: data does not start at full stroke (tstart=%f)' % t1)

    # allocate stroke average matrix
    nt, ncols = d.shape

    navgs = int( np.round((t2-t1)/tstroke) )

    D = np.zeros([navgs,ncols])
    # running index of strokes
    istroke = 0

    # we had some trouble with float equality, so be a little tolerant
    dt = np.mean( d[1:,0]-d[:-1,0] )

    # go in entire strokes
    while t1+tstroke <= t2 + dt:
        # begin of this stroke
        tbegin = t1
        # end of this stroke
        tend = t1+tstroke
        # iterate
        t1 = tend

        # find index where stroke begins:
        i = np.argmin( abs(d[:,0]-tbegin) )
        # find index where stroke ends
        j = np.argmin( abs(d[:,0]-tend) )

        # extract time vector
        time = d[i:j+1,0]
        # replace first and last time instant with stroke begin/endpoint to avoid being just to dt close
        time[0] = tbegin
        time[-1] = tend


        #print('t1=%f t2=%f i1 =%i i2=%i %f %f istroke=%i' % (tbegin, tend, i, j, d[i,0], d[j,0], istroke))

        # actual integration. see wikipedia :)
        # the integral f(x)dx over x2-x1 is the average of the function on that
        # interval. note this script is more precise than the older matlab versions
        # as it is, numerically, higher order. the results are however very similar
        # (below 1% difference)
        for col in range(0,ncols):
            # use interpolation, but actually only for first and last point of a stroke
            # the others are identical as saved in the data file
            dat = np.interp( time, d[:,0], d[:,col] )

            D[istroke,col] = np.trapz( dat, x=time) / (tend-tbegin)

        istroke = istroke + 1
    return D




def write_csv_file( fname, d, header=None, sep=';'):
    # open file, erase existing
    f = open( fname, 'w', encoding='utf-8' )

    # if we specified a header ( a list of strings )
    # write that
    if not header == None:
        # write column headers        
        if isinstance(header, list):
            for name in header[:-1]:
                f.write( name+sep )
            f.write(header[-1])
        else:
            f.write(header)
        # newline after header
        f.write('\n')
        # check

    nt, ncols = d.shape

    for it in range(nt):
        for icol in range(ncols-1):
            f.write( '%e%s' % (d[it,icol], sep) )
        # last column
        f.write( '%e' % (d[it,-1]) )
        # new line
        f.write('\n')
    f.close()


def read_param(config, section, key):
    # read value
    value = config[section].get(key)
    if value is not None:
        # remove comments and ; delimiter, which flusi uses for reading.
        value = value.split(';')[0]
    return value




def read_param_vct(config, section, key):
    value = read_param(config, section, key)
    if "," in value:
        value = np.array( value.split(",") )
    else:
        value = np.array( value.split() )
    value = value.astype(float)
    return value


def fseries(y, n):
    """
    Return the coefficients ai and bi from a truncated Fourier series of signal y
    with n coefficients. Used for kinematics analysis and other encoding.
    
    The coefficients are determined efficiently using FFT, but note the hermitian
    symmetry of real input data. The zeroth mode is multiplied by a factor of
    two, i.e., mean = a0/2.0.
    
    If you request N=20 modes, we return a0 and ai[0:19] so a total of 20 numbers.
    
    Zero mode is returned separately
    
    Input:
    ------
    
        y: vector, float
           The actual data to be analyzed (e.g., angle time series). Note: it is assumed 
           to be sampled on an equidistant grid. 
        n: integer
           Number of Fourier modes to use.
       
    
    Output:
    -------
        a0, ai, bi: numpy arrays containing the real (ai) and imaginary (bi) parts of the n Fourier coefficients.
        
    """ 
    # perform fft
    yk = np.fft.fft(y)

    # data length, for normalization
    N = y.shape[0]

    # return first n values, normalized (note factor 2.0 from hermite symmetry)
    ai = +2.0*np.real( yk[0:n+1] ) / float(N)
    bi = -2.0*np.imag( yk[0:n+1] ) / float(N)
    
    a0 = ai[0]
    ai, bi = ai[1:], bi[1:]
    
    # I was not aware the the bi coefficients need to switch signs, but it seems they
    # do have to indeed. this is related to the minus sign in the basis function (exp-i....)
    # see: https://pages.mtu.edu/~tbco/cm416/fft1.pdf
    return( a0, ai, bi )


def Fserieseval(a0, ai, bi, time):
    """
    evaluate the Fourier series given by a0, ai, bi at the time instant time
    note we divide amplitude of constant by 2 (which is compatible with "fseries")
     
    function is vectorized; pass a vector of time instants for evaluation.
    
    Input:
    ------
    
        a0: float
           zero mode (constant) for historical reasons, it is divided by two.
        ai: vector, float
           real parts of fourier coefficients
        bi: vector, float
            imag parts of fourier coefficients
        time: vector, float
            Output time vector at which to evaluate the hermite interpolation
       
    
    Output:
    -------
        u: vector, float
            Resulting data sampled at "time"        
        
    """
    if ai.shape[0] != bi.shape[0]:
        raise ValueError("ai and bi must be of the same length!")
        
    y = np.zeros_like(time)
    y[:] = a0/2.0
    
    for k in range( ai.size ):
        # note pythons tedious 0-based indexing, so wavenumber is k+1
        y = y + ai[k]*np.cos(2.0*np.pi*float(k+1)*time) + bi[k]*np.sin(2.0*np.pi*float(k+1)*time)
        
    return y



def Hserieseval(a0, ai, bi, time):
    """
    evaluate hermite series, given by coefficients ai (function values)
    and bi (derivative values) at the locations x. Note that x is assumed periodic;
    do not include x=1.0.
    a valid example is x=(0:N-1)/N
    
    Input:
    ------
    
        a0: float
           UNUSED dummy argument
        ai: vector, float
           Function values
        bi: vector, float
            Derivative values
        time: vector, float
            Output time vector at which to evaluate the hermite interpolation
       
    
    Output:
    -------
        u: vector, float
            Resulting data sampled at "time"        
        
    """
    
    # 
    # function is vectorized; pass a vector of time instants for evaluation.
    #
    if len( ai.shape ) != 1:
        raise ValueError("ai must be a vector")

    if len( bi.shape ) != 1:
        raise ValueError("bi must be a vector")
        
    if ai.shape[0] != bi.shape[0]:
        raise ValueError("length of ai and bi must be the same")

    time2 = time.copy()

    # time periodization
    while ( np.max(time2) >= 1.0 ):
        time2[ time2 >= 1.0 ] -= 1.0
        
    n = ai.shape[0]
    dt = 1.0 / n
    j1 = np.floor(time2/dt) # zero-based indexing in python
    j1 = np.asarray(j1, dtype=int)
    j2 = j1 + 1
    
    # periodization
    j2[ j2 > n-1 ] = 0 # zero-based indexing in python
    
    # normalized time (between two data points)
    t = (time2 - j1*dt) / dt

    # values of hermite interpolant
    h00 = (1.0+2.0*t)*((1.0-t)**2)
    h10 = t*((1.0-t)**2)
    h01 = (t**2)*(3.0-2.0*t)
    h11 = (t**2)*(t-1.0)

    # function value
    u = h00*ai[j1] + h10*dt*bi[j1] + h01*ai[j2] + h11*dt*bi[j2]
    
    return u


def read_kinematics_file( fname, unit_out='deg' ):
    import os
    import inifile_tools
    
    if not os.path.isfile(fname):
        raise ValueError("File "+fname+" not found!")
        
        
    if inifile_tools.exists_ini_section(fname, 'kinematics'):
        convention = inifile_tools.get_ini_parameter(fname, 'kinematics', 'convention', dtype=str, default='flusi')
        series_type = inifile_tools.get_ini_parameter(fname, 'kinematics', 'type', dtype=str)
        
        # input file units
        unit_in = inifile_tools.get_ini_parameter(fname,'kinematics', 'units', default='deg', dtype=str)
        
        # options tolerated by FLUSI/WABBIT
        if unit_in in ["degree","DEGREE","Degree","DEG","deg"]:
            # simplified to deg/rad
            unit_in = 'deg'
            
        # options tolerated by FLUSI/WABBIT
        if unit_in == ["radian","RADIAN","Radian","radiant","RADIANT","Radiant","rad","RAD"]:
            # simplified to deg/rad
            unit_in = 'rad'
        
        if convention != "flusi":
            raise ValueError("The kinematics file %s is using a convention not supported yet" % (fname))
            
        if series_type == "fourier":
            a0_phi   = inifile_tools.get_ini_parameter(fname, 'kinematics', 'a0_phi')
            a0_alpha = inifile_tools.get_ini_parameter(fname, 'kinematics', 'a0_alpha')
            a0_theta = inifile_tools.get_ini_parameter(fname, 'kinematics', 'a0_theta')
        else:
            a0_phi, a0_theta, a0_alpha = 0.0, 0.0, 0.0
            
        ai_alpha = inifile_tools.get_ini_parameter(fname, 'kinematics', 'ai_alpha', vector=True)
        bi_alpha = inifile_tools.get_ini_parameter(fname, 'kinematics', 'bi_alpha', vector=True)

        ai_theta = inifile_tools.get_ini_parameter(fname, 'kinematics', 'ai_theta', vector=True)
        bi_theta = inifile_tools.get_ini_parameter(fname, 'kinematics', 'bi_theta', vector=True)

        ai_phi   = inifile_tools.get_ini_parameter(fname, 'kinematics', 'ai_phi', vector=True)
        bi_phi   = inifile_tools.get_ini_parameter(fname, 'kinematics', 'bi_phi', vector=True)
        
        if unit_out != unit_in:
            # factor1 converts input to deg
            if unit_in == 'deg':
                factor1 = 1.0
            else:
                factor1 = 180.0/np.pi
            # factor2 ensures desired output
            if unit_out == 'deg':
                factor2 = 1.0
            else:
                factor2 = np.pi/180.0
                
            a0_phi   *= factor1*factor2
            a0_theta *= factor1*factor2
            a0_alpha *= factor1*factor2
                
            ai_alpha *= factor1*factor2
            bi_alpha *= factor1*factor2
            
            ai_theta *= factor1*factor2
            bi_theta *= factor1*factor2
            
            ai_phi   *= factor1*factor2
            bi_phi   *= factor1*factor2
           

        return a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta, series_type
            
    else:
        raise ValueError('This seems to be an invalid ini file as it does not contain the kinematics section')



def visualize_kinematics_file(fname, ax=None, savePDF=True, savePNG=False):
    """ Read an INI file with wingbeat kinematics and plot the 3 angles over the period. Output written to a PDF and PNG file.
    """

    import matplotlib.pyplot as plt
    
    if ax is None:
        plt.figure( figsize=(cm2inch(12), cm2inch(7)) )
        plt.subplots_adjust(bottom=0.16, left=0.14)
        
        ax = plt.gca()        

    t, phi, alpha, theta = eval_angles_kinematics_file(fname)

    # plt.rcParams["text.usetex"] = False

    

    ax.plot(t, phi  , label='$\\phi$ (flapping)')
    ax.plot(t, alpha, label='$\\alpha$ (feathering)')
    ax.plot(t, theta, label='$\\theta$ (deviation)')

    ax.legend()
    ax.set_xlim([0,1])
    ax.set_xlabel('$t/T$')
        
    # axis y in degree
    from matplotlib.ticker import EngFormatter
    ax.yaxis.set_major_formatter(EngFormatter(unit="°"))

    ax.set_title('$\\Phi=%2.2f^\\circ$ $\\phi_m=%2.2f^\\circ$ $\\phi_\\mathrm{max}=%2.2f^\\circ$ $\\phi_\\mathrm{min}=%2.2f^\\circ$' % (np.max(phi)-np.min(phi), np.mean(phi), np.max(phi), np.min(phi)))
    
    indicate_strokes(ax=ax)
    
    ax.tick_params( which='both', direction='in', top=True, right=True )
    
    if savePDF:
        plt.savefig( fname.replace('.ini','.pdf'), format='pdf' )
    if savePNG:
        plt.savefig( fname.replace('.ini','.png'), format='png' )


def csv_kinematics_file(fname):
    """ 
    Read an INI file with wingbeat kinematics and store the 3 angles over the period in a *.csv file
    """
    
    t, phi, alpha, theta = eval_angles_kinematics_file(fname, time=np.linspace(0,1,100, endpoint=False) )
    
    d = np.zeros([t.shape[0], 4])
    d[:,0] = t
    d[:,1] = phi
    d[:,2] = alpha
    d[:,3] = theta
    
    write_csv_file( fname.replace('.ini', '.csv'), d, header=['time', 'phi', 'alpha', 'theta'], sep=';')


def eval_angles_kinematics_file(fname, time=None, unit_out='deg'):
    """
    
    Read a *.ini file that contains the description of a periodic wing beat. The kinematics
    are described by the three angles phi, alpha, theta and evaluated as Fourier or Hermite
    series depending on what is stored in the *.ini file. 
    
    Parameters
    ----------
    fname : string
        Ini file to read the kinematics from.
    time : array, optional
        Time arry. If none is passed, we sample [0.0, 1.0) with n=1000 samples and return this as well
    unit_out : str, optional
        'rad' or 'deg' output of angles. defaults to 'deg'

    Returns
    -------
    t : array
        time. A copy of the input array or the default if no input time vector is given.
    phi : array
        flapping angle.
    alpha : array
        feathering angle.
    theta : array
        deviation angle.

    """
    # read the kinematics INI file
    a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta, kine_type = read_kinematics_file(fname, unit_out=unit_out)
    
    if time is None:
        # time vector for plotting
        t = np.linspace(0.0, 1.0, 1000, endpoint=False)
    else:
        t = time.copy()
        
    if kine_type == "fourier":
        alpha = Fserieseval(a0_alpha, ai_alpha, bi_alpha, t)
        phi   = Fserieseval(a0_phi  , ai_phi  , bi_phi  , t)
        theta = Fserieseval(a0_theta, ai_theta, bi_theta, t)
        
    elif kine_type == "hermite":
        alpha = Hserieseval(a0_alpha, ai_alpha, bi_alpha, t)
        phi   = Hserieseval(a0_phi  , ai_phi  , bi_phi  , t)
        theta = Hserieseval(a0_theta, ai_theta, bi_theta, t)
    
   
    return t, phi, alpha, theta




def Rx( angle, unit_in="rad" ):
    if unit_in != "rad":
        angle = deg2rad(angle)
    # rotation matrix around x axis
    Rx = np.asarray([[1.0,0.0,0.0],[0.0,np.cos(angle),np.sin(angle)],[0.0,-np.sin(angle),np.cos(angle)]])        
    return Rx


def Ry( angle, unit_in="rad" ):
    if unit_in != "rad":
        angle = deg2rad(angle)
    # rotation matrix around y axis
    Rx = np.asarray([[np.cos(angle),0.0,-np.sin(angle)],[0.0,1.0,0.0],[+np.sin(angle),0.0,np.cos(angle)]])
    return Rx


def Rz( angle, unit_in="rad" ):
    if unit_in != "rad":
        angle = deg2rad(angle)
    # rotation matrix around z axis
    Rx = np.asarray([[ np.cos(angle),+np.sin(angle),0.0],[-np.sin(angle),np.cos(angle),0.0],[0.0,0.0,1.0]])
    return Rx


def Rmirror( x0, n ):
    # mirror by a plane through origin x0 with given normal n
    # source: https://en.wikipedia.org/wiki/Transformation_matrix#Reflection_2
    Rmirror =  np.zeros([4,4])

    a, b, c = n[0], n[1], n[2]
    d = -(a*x0[0] + b*x0[1] + c*x0[2])

    Rmirror = np.asarray([ [1-2*a**2,-2*a*b,-2*a*c,-2*a*d], [-2*a*b,1-2*b**2,-2*b*c,-2*b*d], [-2*a*c,-2*b*c,1-2*c**2,-2*c*d],[0,0,0,1] ])

    return(Rmirror)

def M_stroke(eta, wing):
    warn("Using M_stroke is deprecated, use get_M_b2s instead", DeprecationWarning, stacklevel=2)
    return get_M_b2s(eta, wing)

def M_wing(alpha, theta, phi, wing):
    warn("Using M_wing is deprecated, use get_M_s2w instead", DeprecationWarning, stacklevel=2)
    return get_M_s2w(alpha, theta, phi, wing)

def M_body(psi, beta, gamma):
    warn("Using print is deprecated, use get_M_g2b instead", DeprecationWarning, stacklevel=2)
    return get_M_g2b(psi, beta, gamma)


def get_M_b2s(eta, side, unit_in="rad"):
    """
    Rotation matrix from body to stroke system, as defined in Engels et al. 2016 SISC

    Parameters
    ----------
    eta : rad
        stroke plane angle
    side : str
        left or right

    Returns
    -------
    M_stroke rotation matrix
    """
    if unit_in != "rad":
        eta = deg2rad(eta)
        
    ce, se = np.cos(eta), np.sin(eta)
        
    if side =="left":
        # this syntax is easier for humans but slower for computers (relevant for QSM)
        # M_stroke = Ry(eta)
        
        M_stroke = np.array([
                    [ce, 0, -se],
                    [0,  1,   0],
                    [se, 0,  ce]  ])        
        
    elif side == "right":
        # this syntax is easier for humans but slower for computers (relevant for QSM)
        # M_stroke = Rx(np.pi) @ Ry(eta)
        
        M_stroke = np.array([
                    [ce, 0,  -se],
                    [0,  -1,   0],
                    [-se, 0,  -ce]  ])     
    else:
        raise("Neither right nor left wing")
        
    return M_stroke   


def get_M_s2w(alpha, theta, phi, side, unit_in='rad'):
    """
    Rotation matrix from stroke to wing system, as defined in Engels et al. 2016 SISC

    Parameters
    ----------
    alpha : rad or deg
        feathering angle
    theta : rad or deg
        deviation angle
    phi : rad or deg
        flapping angle
    wing : str
        left or right

    Returns
    -------
    M_wing rotation matrix

    """    
    if unit_in != "rad":
        alpha = deg2rad(alpha)
        theta = deg2rad(theta)
        phi = deg2rad(phi)
    
    # this syntax is easier for humans but slower for computers (relevant for QSM)
    # if side =="left":
    #     M = Ry(alpha) @ Rz(theta) @ Rx(phi)
    # elif side == "right":
    #     M = Ry(-alpha) @ Rz(theta) @ Rx(-phi)
    # else:
    #     raise("Neither right nor left wing")
    
    if side == 'right':
        alpha *= -1.0
        phi *= -1.0
        
    # Precompute trigonometric functions
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    M = np.array([
        [ca * ct, ca * st * cp + sa * sp, ca * st * sp - sa * cp],
        [-st    ,                ct * cp,                ct * sp],
        [sa * ct, sa * st * cp - ca * sp, sa * st * sp + ca * cp]    ])        
        
    return M



def get_M_g2b(psi, beta, gamma, unit_in='rad'):
    """
    Rotation matrix from lab to body reference frame, as defined in Engels et al. 2016 SISC
    
    Parameters
    ----------
    psi : rad or deg
        roll angle
    beta : rad or deg
        pitch angle
    gamma : rad or deg
        yaw angle

    Returns
    -------
    M_body : TYPE
        Body rotation matrix

    """
    if unit_in != "rad":
        psi = deg2rad(psi)
        beta = deg2rad(beta)
        gamma = deg2rad(gamma)
    
    # this syntax is easier for humans but slower for computers (relevant for QSM)
    # M_body = Rx(psi) @ Ry(beta) @ Rz(gamma)
    
    cg, sg = np.cos(gamma), np.sin(gamma)
    cb, sb = np.cos(beta), np.sin(beta)
    cp, sp = np.cos(psi), np.sin(psi)
    
    M_body = np.array([
             [cg * cb               ,                    sg * cb,         -sb],
             [cg * sb * sp - sg * cp,     sg * sb * sp + cg * cp,     cb * sp],
             [cg * sb * cp + sg * sp,     sg * sb * cp - cg * sp,     cb * cp]    ])
    
    return M_body




def get_M_b2w(alpha, theta, phi, eta, side, unit_in='rad'):
    # composite matrix
    return get_M_s2w(alpha, theta, phi, side, unit_in) @ get_M_b2s(eta, side, unit_in)


def get_many_M_b2w(alpha, theta, phi, eta, side, unit_in='rad'):

    return get_many_M_s2w(alpha, theta, phi, side, unit_in) @ get_many_M_b2s(eta, side, unit_in)

def get_many_M_s2w(alpha, theta, phi, side, unit_in='rad'):
    """
    Same as get_M_s2w, but creates a (N x 3 x 3) array with N matrices for the N 
    entries in alpha, theta, phi.

    """
    if side == 'right':            
        ca, sa = np.cos(-alpha), np.sin(-alpha)
        ct, st = np.cos( theta), np.sin( theta)
        cp, sp = np.cos(-phi), np.sin(-phi)
    else:                            
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        
    if alpha.shape != theta.shape != phi.shape:
        raise ValueError("not all of the size")

    M_s2w = np.zeros( (alpha.shape[0],3,3))
    M_s2w[:,0,0] = ca * ct
    M_s2w[:,0,1] = ca * st * cp + sa * sp
    M_s2w[:,0,2] = ca * st * sp - sa * cp
    
    M_s2w[:,1,0] = -st
    M_s2w[:,1,1] = ct * cp
    M_s2w[:,1,2] = ct * sp
    
    M_s2w[:,2,0] = sa * ct
    M_s2w[:,2,1] = sa * st * cp - ca * sp
    M_s2w[:,2,2] = sa * st * sp + ca * cp
    
    return M_s2w


def get_many_M_g2b(psi, beta, gamma, unit_in='rad'):
    """
    Same as get_M_g2b, but creates a (N x 3 x 3) array with N matrices for the N 
    entries in psi, beta, gamma.

    """
    if unit_in != "rad":
        psi = deg2rad(psi)
        beta = deg2rad(beta)
        gamma = deg2rad(gamma)
        
    if psi.shape != beta.shape != gamma.shape:
        raise ValueError("not all of the size")
    # this syntax is easier for humans but slower for computers (relevant for QSM)
    # M_body = Rx(psi) @ Ry(beta) @ Rz(gamma)
    
    cg, sg = np.cos(gamma), np.sin(gamma)
    cb, sb = np.cos(beta), np.sin(beta)
    cp, sp = np.cos(psi), np.sin(psi)
    
    M = np.zeros( (psi.shape[0],3,3))
    
    M[:,0,0] = cg * cb
    M[:,0,1] = sg * cb
    M[:,0,2] = -sb
    
    M[:,1,0] = cg * sb * sp - sg * cp
    M[:,1,1] = sg * sb * sp + cg * cp
    M[:,1,2] = cb * sp
    
    M[:,2,0] = cg * sb * cp + sg * sp
    M[:,2,1] = sg * sb * cp - cg * sp
    M[:,2,2] = cb * cp
    
    return M

def get_many_M_b2s(eta, side, unit_in="rad"):
    """
    Same as get_M_s2w, but creates a (N x 3 x 3) array with N matrices for the N 
    entries in alpha, theta, phi.

    """
    ce, se = np.cos(eta), np.sin(eta)
    
    M = np.zeros( (eta.shape[0],3,3))
    
    if side =="left":
        M[:,0,0] = ce
        M[:,0,1] = 0
        M[:,0,2] = -se
        
        M[:,1,0] = 0
        M[:,1,1] = 1
        M[:,1,2] = 0
        
        M[:,2,0] = se
        M[:,2,1] = 0
        M[:,2,2] = ce
        
    elif side == "right":
        M[:,0,0] = ce
        M[:,0,1] = 0
        M[:,0,2] = -se
        
        M[:,1,0] = 0
        M[:,1,1] = -1
        M[:,1,2] = 0
        
        M[:,2,0] = -se
        M[:,2,1] = 0
        M[:,2,2] = -ce
        
    return M



def visualize_wingpath_chord( fname, psi=0.0, gamma=0.0, beta=0.0, eta_stroke=0.0, equal_axis=True, DrawPath=False, PathColor='k',
                             x_pivot_b=[0,0,0], wing='left', chord_length=0.1,
                             draw_true_chord=False, meanflow=None, reverse_x_axis=False, colorbar=False, 
                             time=np.linspace( start=0.0, stop=1.0, endpoint=False, num=40), cmap=None, 
                             ax=None, savePNG=False, savePDF=True, draw_stoke_plane=True, mark_pivot=True,
                             force_vectors=False, fname_forces=None, T0_forces=0.0, scale_forces=0.02, fcoef=1.0,
                             force_scale_vector_length=1.0, force_scale_vector_label="1.0*fcoef", cmap_forces=None):
    """
    Lollipop-diagram. 
    
    This type of diagram shows a "wing section", a line with dot for the leading edge. This is called a lollipop.
    The visualization takes place in the sagittal plane (index _m for midplane, because _s is already taken for stroke plane).
    It would be more natural to draw this diagram in the body coordinate system, but that's not the convention. The sagittal
    plane is the body reference frame with an additional rotation by -beta, where beta is pitch angle.
    
    As the body angles may be time dependent, you can pass them (still in deg) as vectors with the same size as time. 
    In fact, this is important only for the Lollipop diagram with forces, because we read in the *.t file, which is in the
    global coordinate system. You can obviously only plot force vectors after the simulation is completed.    

    Parameters
    ----------
    fname : TYPE
        *.ini file to take the kinematics from
    psi : TYPE, optional
        body roll angle, scalar or vector same length as time
    gamma : TYPE, optional
        body yaw angle, scalar or vector same length as time
    beta : TYPE, optional
        body pitch angle, scalar or vector same length as time
    eta_stroke : TYPE, optional
        anatomical stroke plane angle
    equal_axis : TYPE, optional
        DESCRIPTION. The default is True.
    DrawPath : TYPE, optional
        Draw the wing tip path as dashed line or not.
    x_pivot_b : TYPE, optional
        DESCRIPTION. The default is [0,0,0].
    wing : TYPE, optional
        DESCRIPTION. The default is 'left'.
    chord_length : TYPE, optional
        DESCRIPTION. The default is 0.1.
    draw_true_chord : TYPE, optional
        DESCRIPTION. The default is False.
    meanflow : TYPE, optional
        If specified, we add the mean flow vector to the drawing, which is sometimes useful. Provide a 3D vector [ux,uy,uz]
    reverse_x_axis : TYPE, optional
        DESCRIPTION. The default is False.
    colorbar : TYPE, optional
        DESCRIPTION. The default is False.
    time : TYPE, optional
        DESCRIPTION. The default is np.linspace( start=0.0, stop=1.0, endpoint=False, num=40).
    cmap : TYPE, optional
        Colormap encoding time. Can also be a constant string for fixed colors.
    cmap_forces :
        As cmap, but for force vector arrows. Defaults to cmap
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    savePNG : TYPE, optional
        DESCRIPTION. The default is False.
    savePDF : TYPE, optional
        DESCRIPTION. The default is True.
    draw_stoke_plane : TYPE, optional
        DESCRIPTION. The default is True.
    mark_pivot : TYPE, optional
        DESCRIPTION. The default is True.
    force_vectors : TYPE, optional
        Plot force vectors on top of lollipops?. The default is False.
    fname_forces : TYPE, optional
        *.t file to get the wing forces from (e.g. forces_leftwing.t)
    T0_forces : TYPE, optional
        Which cycle to use for forces, give the start point (ie 1.0 for the 2nd cycle)
    scale_forces : TYPE, optional
        Scaling constant for the force vectors. Not to be confused with fcoef. This here just scales the vector length on the plot.
    fcoef : TYPE, optional
        Coefficient to make forces dimensional, if fcoef=1.0 then the dimenionless units from the CFD are used.
    force_scale_vector_length : TYPE, optional
        How long should the little reference vector be, that gives the reader the scale of the force arrows?
        Give this in the same unit as fcoef.
    force_scale_vector_label : TYPE, optional
        Label for the reference force vector (e.g. "100N"). The default is "1.0*fcoef".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib

    

    if not os.path.isfile(fname):
        raise ValueError("The file "+fname+" does not exist.")
        
    if ax is None:
        plt.figure( figsize=(cm2inch(12), cm2inch(7)) )
        plt.subplots_adjust(bottom=0.16, left=0.14)
        ax = plt.gca() # we need that to draw lines...
        
        # this is a manually set size, which should be the same as what is produced by visualize kinematics file
        # plt.gcf().set_size_inches([4.71, 2.75] )
        plt.gcf().set_size_inches([4.0, 3.6] )
        plt.gcf().subplots_adjust(hspace=0.0, right=0.88, bottom=0.12, left=0.16, top=0.92)
        
    if type(beta) != np.ndarray:
        beta = beta * np.ones_like(time)
    if type(psi) != np.ndarray:
        psi = psi * np.ones_like(time)
    if type(gamma) != np.ndarray:
        gamma = gamma * np.ones_like(time)
        
    # this diagram would be most convenient in the body coordinate system,
    # but that is not the convention. The convention is the sagittal plane, looking from the 
    # side at the insect, i.e. you can see the pitch angle. However as this is arbitrary, it
    # seems to be more logical to me to use the mean pitch angle for that, rather than the
    # instantaneous one. In most cases, this is the same, as beta is often constant.
    beta_sagittal = np.mean(beta)

    # In the Lollipop diagram, we look at the insect from the side. The pitch angle
    # is visible, but yaw and roll are not (otherwise the fig is distorted and very 
    # difficult to interpret). This translates to an additional rotation around y 
    # by -beta, after going to the body system.
    # Index _m because _s is stroke plane.
    M_b2sagittal = Ry(-1.0*deg2rad(beta_sagittal))
        
    # read kinematics data:
    time, phi, alpha, theta = eval_angles_kinematics_file(fname, time=time, unit_out='deg')
    
        
    # wing tip in wing coordinate system
    x_tip_w = np.asarray([0.0, 1.0, 0.0])
    x_le_w  = np.asarray([ 0.5*chord_length,1.0,0.0])
    x_te_w  = np.asarray([-0.5*chord_length,1.0,0.0])
    x_pivot_b = np.asarray(x_pivot_b)

    # array of color (note normalization to 1 for query values)
    if cmap is None:
        cmap = plt.cm.jet
    if type(cmap) == matplotlib.colors.LinearSegmentedColormap:
        colors = cmap( (np.arange(time.size) / time.size) )
    else:
        # if its a constant color, jus create a list of colors
        colors = time.size*[cmap]
        
    # default is using same colormap for both
    if cmap_forces is None:
        cmap_forces = cmap
    
    if type(cmap_forces) == matplotlib.colors.LinearSegmentedColormap:
        colors_forces = cmap( (np.arange(time.size) / time.size) )
    else:
        # if its a constant color, jus create a list of colors
        colors_forces = time.size*[cmap_forces]    
       
    # read forces
    if force_vectors:
        d_forces = load_t_file(fname_forces)

    # step 1: draw the symbols for the wing section for some time steps
    for i in range(time.size):        
        # (true) body transformation matrix
        M_g2b = get_M_g2b(psi[i], beta[i], gamma[i], unit_in='deg')
            
        # rotation matrix (body -> wing)
        M_b2w = get_M_b2w(alpha[i], theta[i], phi[i], eta_stroke, wing, unit_in='deg')

        # convert wing points to sagittal coordinate system
        x_tip_m =  M_b2sagittal @ ( np.transpose(M_b2w) @ x_tip_w + x_pivot_b ) 
        x_le_m  =  M_b2sagittal @ ( np.transpose(M_b2w) @ x_le_w  + x_pivot_b ) 
        x_te_m  =  M_b2sagittal @ ( np.transpose(M_b2w) @ x_te_w  + x_pivot_b )

        if not draw_true_chord:
            # the wing chord changes in length, as the wing moves and is oriented differently
            # note if the wing is perpendicular, it is invisible
            # so this vector goes from leading to trailing edge:
            e_chord = x_te_m - x_le_m
            e_chord[1] = 0.0

            # normalize it to have the right length
            e_chord = e_chord / (np.linalg.norm(e_chord))

            # pseudo TE and LE. note this is not true TE and LE as the line length changes otherwise
            x_le_m = x_tip_m - e_chord * chord_length/2.0
            x_te_m = x_tip_m + e_chord * chord_length/2.0

        # draw actual lollipop
        # mark leading edge with a marker
        ax.plot( x_le_m[0], x_le_m[2], marker='o', color=colors[i], markersize=4 )
        # draw wing chord
        ax.plot( [x_te_m[0], x_le_m[0]], [x_te_m[2], x_le_m[2]], '-', color=colors[i])
        
        # draw arrow for forces
        if force_vectors:
            # interpolate the value for the force vector. Note forces are in global 
            # reference frame.
            Fx_g = np.interp( time[i]+T0_forces, d_forces[:,0], d_forces[:,1] )
            Fy_g = np.interp( time[i]+T0_forces, d_forces[:,0], d_forces[:,2] )
            Fz_g = np.interp( time[i]+T0_forces, d_forces[:,0], d_forces[:,3] )
            # to sagittal plane
            F_m = M_b2sagittal @ M_g2b @ np.asarray([Fx_g, Fy_g, Fz_g])  
            
            # force vector starts at mid-lollipop
            point0x = 0.5*(x_te_m[0] + x_le_m[0])
            point0y = 0.5*(x_te_m[2] + x_le_m[2])
            # force vector end point            
            point1x = point0x + scale_forces * F_m[0]
            point1y = point0y + scale_forces * F_m[2]

            ax.arrow( point0x, point0y, point1x-point0x, point1y-point0y, head_width=0.04, color=colors_forces[i])
            
            # plt.text( x_le_g[0]*1.02, x_le_g[2]*1.02, "F=%2.2f nN" % (fcoef*np.sqrt(Fx**2 + Fz**2)), color=colors_forces )
            
            # plot the scale arrow with 100 nN length
            if (i==0):
                scale_arrow_length = scale_forces*force_scale_vector_length / fcoef # nN
                
                ax.arrow(0.5, 1.0, scale_arrow_length, 0.0, head_width=0.04, color=colors_forces[i])
                ax.text(0.5+0.5*scale_arrow_length, 1.05, force_scale_vector_label, horizontalalignment='center')


    # step 2: draw the path of the wingtip
    if DrawPath:
        # refined time vector for drawing the wingtip path
        time2 = np.linspace( start=0.0, stop=1.0, endpoint=False, num=1000)
        xpath, zpath = np.zeros_like(time2), np.zeros_like(time2)
        
        # different time vector
        time2, phi, alpha, theta = eval_angles_kinematics_file(fname, time=time2, unit_out='deg')

        for i in range(time2.size):
            # rotation matrix from body to wing coordinate system 
            M_b2w = get_M_b2w(alpha[i], theta[i], phi[i], eta_stroke, wing, unit_in='deg')
            # convert wing points to sagittal coordinate system
            x_tip_m = M_b2sagittal @ np.transpose(M_b2w) @ x_tip_w + x_pivot_b

            xpath[i] = x_tip_m[0]
            zpath[i] = x_tip_m[2]
        ax.plot( xpath, zpath, linestyle='--', color=PathColor, linewidth=1.0 )


    # Draw stroke plane as a dashed line
    # NOTE: if beta is not constant, there should be more lines...
    if draw_stoke_plane:
        M_b2s = get_M_b2s(eta_stroke, wing, unit_in='deg')
        
        # we draw the line between [0,0,-1] and [0,0,1] in the stroke system        
        x1_s = np.asarray([0.0, 0.0, +1.0])
        x2_s = np.asarray([0.0, 0.0, -1.0])
        
        # bring these points back to the global system
        x1_m = M_b2sagittal @ ( np.transpose(M_b2s) @ x1_s + x_pivot_b )
        x2_m = M_b2sagittal @ ( np.transpose(M_b2s) @ x2_s + x_pivot_b )       
    
        # remember we're in the x-z plane
        ax.plot( [x1_m[0],x2_m[0]], [x1_m[2],x2_m[2]], color='k', linewidth=1.0, linestyle='--')


    if mark_pivot:
        color = 'k'
        if ax.get_facecolor() == (0.0, 0.0, 0.0, 1.0):
            color = 'w'
        ax.plot(x_pivot_b[0], x_pivot_b[2], 'p', color=color)
    
    if equal_axis:
        axis_equal_keepbox( plt.gcf(), ax )

    if meanflow is not None:
        if max(abs(np.asarray(meanflow))) > 0.0:
            x0, y0 = 0.0, 0.0
            plt.arrow( x0, y0, meanflow[0], meanflow[2], width=0.000001, head_width=0.025 )
            plt.text(x0+meanflow[0]*1.1, y0+meanflow[2]*1.1, '$u_\\infty$' )

    if reverse_x_axis:
        ax.invert_xaxis()
        
    if colorbar:
        sm = plt.cm.ScalarMappable( cmap=cmap, norm=plt.Normalize(vmin=0,vmax=1) )
        sm._A =[]
        plt.colorbar(sm)
    
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel('x_{sagittal}/R')
    ax.set_ylabel('z_{sagittal}/R')
    
    axis_equal_keepbox(plt.gcf(), ax)
    
    # modify ticks in matlab-style.
    ax.tick_params( which='both', direction='in', top=True, right=True )
    
    if savePDF:
        plt.savefig( fname.replace('.ini','_path.pdf'), format='pdf' )
    if savePNG:
        plt.savefig( fname.replace('.ini','_path.png'), format='png', dpi=300 )



def wingtip_path( fname, time=None, wing='left', eta_stroke=0.0, psi=0.0, beta=0.0, gamma=0.0, x_pivot_b = [0.0, 0.0, 0.0],
                 alpha=None, theta=None, phi=None):
   
    if time is None:
        time = np.linspace(0, 1.0, 1000, endpoint=True)
        
    # wing tip in wing coordinate system
    x_tip_w   = np.asarray([0.0, 1.0, 0.0])
    x_body_g  = np.asarray([0.0, 0.0, 0.0])
    x_pivot_b = np.asarray(x_pivot_b)
        
    # body transformation matrix
    M_g2b = get_M_g2b(psi, beta, gamma, unit_in='deg')
    
    if fname is not None and alpha is None and theta is None and phi is None:
        # evaluate kinematics fom file
        t, phi_l, alpha_l, theta_l = eval_angles_kinematics_file(fname, time=time)
    else:
        # use values passed as arrays
        phi_l, alpha_l, theta_l = phi, alpha, theta
    
    # allocation
    xb, yb, zb = np.zeros(time.shape), np.zeros(time.shape), np.zeros(time.shape)
       
    for i in range(time.shape[0]):
        # rotation matrix from body to wing coordinate system
        M_b2w = get_M_b2w(alpha_l[i], theta_l[i], phi_l[i], eta_stroke, side=wing, unit_in='deg')

        # convert wing points to global coordinate system
        x_tip_g = np.transpose(M_g2b) @ ( np.transpose(M_b2w) @ x_tip_w + x_pivot_b ) + x_body_g

        xb[i] = (x_tip_g[0])
        yb[i] = (x_tip_g[1])
        zb[i] = (x_tip_g[2])  
    
    return xb, yb, zb


def wingtip_velocity( fname_kinematics, time=None, side='left' ):
    """ Compute wingtip velocity as a function of time, given a wing kinematics parameter
    file. Note we assume the body at rest (hence relative to body).
    """    
    if time is None:
        # for good temporal accuracy (differentiation)
        time = np.linspace(0, 1.0, 1000, endpoint=True)
        
    # because velocity magnitude does not depend on the stroke plane
    eta_stroke = 0.0
    
    # evaluate kinematics file (may be hermite or Fourier file)        
    t, phi, alpha, theta = eval_angles_kinematics_file(fname_kinematics, time=time)
    
    # wing tip in wing coordinate system
    x_tip_w = np.asarray([0.0, 1.0, 0.0])
    
    # allocation
    v_tip_b = np.zeros(time.shape)
    
    for i in range(time.size-1):
        # we use simple differentiation (finite differences) to get the velocity
        dt = time[1]-time[0]

        # rotation matrix from body to wing coordinate system (time i)
        M_b2w = get_M_b2w(alpha[i], theta[i], phi[i], eta_stroke, side)
        
        # convert wing points to body coordinate system
        x1_tip_b = np.transpose(M_b2w) @ x_tip_w

        # rotation matrix from body to wing coordinate system (time i+1)
        M_b2w = get_M_b2w(alpha[i+1], theta[i+1], phi[i+1], eta_stroke, side)

        # convert wing points to body coordinate system
        x2_tip_b = np.transpose(M_b2w) @ x_tip_w

        v_tip_b[i] = np.linalg.norm( (x2_tip_b - x1_tip_b)/dt )

    # periodization (because we did not compute the last point)
    v_tip_b[-1] = v_tip_b[0]
    
    return v_tip_b


def interp_matrix( d, time_new ):
    from scipy.interpolate import interp1d
    
    # interpolate matrix d using given time vector
    nt_this, ncols = d.shape
    nt_new = len(time_new)

    # allocate target array
    d2 = np.zeros( [nt_new, ncols] )
    # copy time vector
    d2[:,0] = time_new

    # loop over columns and interpolate
    for i in range(1,ncols):
        # interpolate this column i to equidistant data
        d2[:,i]  = interp1d(d[:,0], d[:,i], fill_value='extrapolate')(time_new)

    return d2



def indicate_strokes( force_fullstrokes=True, tstart=None, ifig=None, tstroke=1.0, ax=None, color=[0.85, 0.85, 0.85, 1.0] ):
    """
    Add shaded background areas to xy plots, often used to visually distinguish
    up- and downstrokes.

    Input:
    ------

        force_fullstrokes : bool
            If set we use integer stroke numbers, even if the currents plots
            axis are not (often the x-axis goes from say -0.1 to 1.1)
        tstart : list of float
            Manually give starting points of strokes. Shading starts after
            tstroke/2 for a duration of tstroke/2
        tstroke : duration of a stroke, if not units.


    Output:
    -------
        directly in currently active figure, or figure /axis given in call
    """

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    if ifig == None:
        # get current axis
        if ax is None:
            ax = plt.gca() # we need that to draw rectangles...
    else:
        if ax is None:
            plt.figure(ifig)
            ax = plt.gca()


    # initialize empty list of rectangles
    rects = []

    # current axes extends
    t1, t2 = ax.get_xbound()
    y1, y2 = ax.get_ybound()

    if force_fullstrokes:
        t1 = np.round(t1 / tstroke) * tstroke
        t2 = np.round(t2 / tstroke) * tstroke
        
    # will there be any strokes at all?
    if abs(t2-t1) < tstroke:
        print('warning: no complete stroke present, not indicating any strokes.')
        return

    if abs(t1 - np.round(t1)) >= 1e-3:
        print('warning: data does not start at full stroke (tstart=%f)' % t1)

    if tstart is None:
        # go in entire strokes
        while t1+tstroke <= t2:
            # begin of this stroke
            tbegin = t1
            # end of this stroke
            tend = t1 + tstroke / 2.0
            # iterate
            t1 = tbegin + tstroke
            # create actual rectangle
            r = Rectangle( [tbegin,y1], tend-tbegin, y2-y1, fill=True)
            rects.append(r)
    else:
        for tbegin in tstart:
            # end of this stroke
            tend = tbegin + tstroke / 2.0
            # create actual rectangle
            r = Rectangle( [tbegin,y1], tend-tbegin, y2-y1, fill=True)
            rects.append(r)


    # Create patch collection with specified colour/alpha
    pc = PatchCollection(rects, facecolor=color, edgecolor=color, zorder=-2)

    # Add collection to axes
    ax.add_collection(pc)


def add_shaded_background( t0, t1, ax=None, color=[0.85, 0.85, 0.85, 1.0] ):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()    
        
    # initialize empty list of rectangles
    rects = []

    # current axes extends
    y1, y2 = ax.get_ybound()
    
    # create actual rectangle
    r = Rectangle( [t0,y1], t1-t0, y2-y1, fill=True)
    rects.append(r)
    
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(rects, facecolor=color, edgecolor=color, zorder=-2)

    # Add collection to axes
    ax.add_collection(pc)


def insectSimulation_postProcessing( run_directory='./', output_filename='data_wingsystem.csv', plot=True, filename_plot='forcesMoments_wingSystem.pdf' ):
    """ 
    Post-Processes an existing insect simulation done with WABBIT.
    
    Reads the forces_XXwing.t, moments_XX_wing.t and kinematics.t (XX=left/right)
    and computes the forces and moments in the respective WING reference frame.
    
    Output is saved to CSV file and plotted to PDF file.
    """
    import numpy as np
    
    # avoid silly mistakes and be sure there is a slash at the end of the string
    run_directory += '/'
 
    # indices [zero-based] in kinematics.t:
    # 0	time
    # 1	xc_body_g_x
    # 2	xc_body_g_y
    # 3	xc_body_g_z
    # 4	psi
    # 5	beta
    # 6	gamma
    # 7	eta_stroke
    # 8	alpha_l
    # 9	phi_l
    # 10	theta_l
    # 11	alpha_r
    # 12	phi_r
    # 13	theta_r    
    # 14	rot_rel_l_w_x
    # 15	rot_rel_l_w_y
    # 16	rot_rel_l_w_z    
    # 17	rot_rel_r_w_x
    # 18	rot_rel_r_w_y
    # 19	rot_rel_r_w_z   
    # 20	rot_dt_l_w_x
    # 21	rot_dt_l_w_y
    # 22	rot_dt_l_w_z    
    # 23	rot_dt_r_w_x
    # 24	rot_dt_r_w_y
    # 25	rot_dt_r_w_z
    k = load_t_file( run_directory+'kinematics.t' )
    
    # works only for two winged insects at the moment
    if (k.shape[1] > 26):
        raise ValueError("This kinematics.t file appears to be for a 4-winged insect, not yet implemented")

    time    = k[:,0]
    psi     = k[:,4]
    beta    = k[:,5]
    gamma   = k[:,6]
    eta_stroke = k[:,7]    
    alpha_l    = k[:,8]
    phi_l      = k[:,9]
    theta_l    = k[:,10]    
    alpha_r    = k[:,11]
    phi_r      = k[:,12]
    theta_r    = k[:,13]
    
    import os
    
    forces_R = np.zeros(k.shape)
    forces_L = np.zeros(k.shape)
    moments_R = np.zeros(k.shape)
    moments_L = np.zeros(k.shape)
    
    
    if os.path.isfile(run_directory+'forces_rightwing.t'):    
        forces_R  = load_t_file( run_directory+'forces_rightwing.t' )        
        
    if os.path.isfile(run_directory+'forces_leftwing.t'):    
        forces_L  = load_t_file( run_directory+'forces_leftwing.t'  )    
        
    if os.path.isfile(run_directory+'moments_rightwing.t'):            
        moments_R = load_t_file( run_directory+'moments_rightwing.t')
        
    if os.path.isfile(run_directory+'moments_leftwing.t'):    
        moments_L = load_t_file( run_directory+'moments_leftwing.t' )

    data_new = np.zeros( [k.shape[0], 13] )
    
    for it in range(k.shape[0]):
        data_new[it, 0] = time[it]
        
        #--- body rotation matrix
        M_g2b = get_M_g2b(psi[it],beta[it], gamma[it], unit_in='rad')
        
        #--- rotation matrix from body to wing coordinate system        
        M_b2w_r = get_M_b2w(alpha_r[it], theta_r[it], phi_r[it], eta_stroke[it], 'right', unit_in='rad')
        M_b2w_l = get_M_b2w(alpha_l[it], theta_l[it], phi_l[it], eta_stroke[it], 'left', unit_in='rad')

        #--- right wing
        F = M_b2w_r @ M_g2b @ forces_R[it,1:3+1]
        M = M_b2w_r @ M_g2b @ moments_R[it,1:3+1]

        data_new[it,1], data_new[it,2], data_new[it,3] = F[0], F[1], F[2]
        data_new[it,4], data_new[it,5], data_new[it,6] = M[0], M[1], M[2]
        
        #--- left wing
        F = M_b2w_l @ M_g2b @ forces_L[it,1:3+1]
        M = M_b2w_l @ M_g2b @ moments_L[it,1:3+1]
        
        data_new[it,1+6], data_new[it,2+6], data_new[it,3+6] = F[0], F[1], F[2]
        data_new[it,4+6], data_new[it,5+6], data_new[it,6+6] = M[0], M[1], M[2]


    #--- save output to CSV file
    if not output_filename is None:
        fid = open( output_filename, 'w' ) #open file, erase existing
        for it in range(k.shape[0]):
            for j in range(13-1):
                fid.write('%e;' % (data_new[it,j]))
            fid.write('%e\n' % (data_new[it,-1]))
        fid.close()
    
    
    #--- create and save a plot
    if plot:
        d = data_new
        import matplotlib.pyplot as plt    
        
        plt.figure()
        plt.subplot(2,2,2)
        plt.plot(d[:,0], d[:,1], label='$F_{R,x}^{(w)}$')
        plt.plot(d[:,0], d[:,2], label='$F_{R,y}^{(w)}$')
        plt.plot(d[:,0], d[:,3], label='$F_{R,z}^{(w)}$')
        indicate_strokes()
        plt.legend()
        
        plt.subplot(2,2,4)
        plt.plot(d[:,0], d[:,4], label='$M_{R,x}^{(w)}$')
        plt.plot(d[:,0], d[:,5], label='$M_{R,y}^{(w)}$')
        plt.plot(d[:,0], d[:,6], label='$M_{R,z}^{(w)}$')
        indicate_strokes()
        plt.legend()
        
        plt.subplot(2,2,1)
        plt.plot(d[:,0], d[:,7], label='$F_{L,x}^{(w)}$')
        plt.plot(d[:,0], d[:,8], label='$F_{L,y}^{(w)}$')
        plt.plot(d[:,0], d[:,9], label='$F_{L,z}^{(w)}$')
        indicate_strokes()
        plt.legend()
        
        plt.subplot(2,2,3)
        plt.plot(d[:,0], d[:,10], label='$M_{L,x}^{(w)}$')
        plt.plot(d[:,0], d[:,11], label='$M_{L,y}^{(w)}$')
        plt.plot(d[:,0], d[:,12], label='$M_{L,z}^{(w)}$')
        indicate_strokes()
        plt.legend()
        plt.gcf().set_size_inches([10,10])
        plt.tight_layout()
        plt.savefig( run_directory+filename_plot)


def read_flusi_HDF5( fname, dtype=np.float64, verbose=True):
    """  Read HDF5 file generated by FLUSI.
    Returns: time, box, origin, data
    """
    import flusi_tools
    time, box, origin, data = flusi_tools.read_flusi_HDF5( fname, dtype=dtype, verbose=verbose)
    return time, box, origin, data


def write_flusi_HDF5( fname, time, box, data, viscosity=0.0, origin=np.array([0.0,0.0,0.0]), dtype=np.float32 ):
    import flusi_tools
    flusi_tools.write_flusi_HDF5( fname, time, box, data, viscosity=viscosity, origin=origin, dtype=dtype )


def load_image( infilename ):
    from PIL import Image
    import numpy as np

    img = Image.open( infilename )
    img.load()
    img = img.convert('RGB')
    data = np.asarray( img , dtype=float )
    
    # Funny: a tall image is loaded as 10000x100 here, but GIMP etc
    # read it as 100x10000, so I guess its a good idea to swap the axis
    #
    # In matix convention (IJ), first index is rows, and a tall image thus should have 
    # the first index large. 
    #
    # GIMP etc give, I think, WxH because of a stupid convention
#    data = np.swapaxes(data, 0, 1)

    return data


def tiff2hdf( dir, outfile, dx=1, origin=np.array([0,0,0]) ):
    print('******************************')
    print('* tiff2hdf                   *')
    print('******************************')

    # first, get the list of tiff files to process
    files = glob.glob( dir+'/*.tif*' )
    files.sort()
    print("Converting dir %s (%i files) to %s" % (dir, len(files), outfile))

    nz = len(files)

    if nz>0:
        # read in first file to get the resolution
        data = load_image(files[0])
        nx, ny = data.shape

        print( "Data dimension is %i %i %i" % (nx,ny,nz))

        # allocate (single precision) data
        data = np.zeros([nx,ny,nz], dtype=np.float32)

        # it is useful to now use the entire array, so python can crash here if
        # out of memory, and not after waiting a long time...
        data = data + 1.0


        for i in range(nz):
            sheet = load_image( files[i] )
            data[:,:,i] = sheet.copy()

        write_flusi_HDF5( outfile, 0.0, [float(nx)*dx,float(ny)*dx,float(nz)*dx], data, viscosity=0.0, origin=origin )



def write_kinematics_ini_file(fname, alpha, phi, theta, nfft, header=['header goes here'], 
                              unit_in='deg'):
    """
     given the angles alpha, phi and theta and a vector of numbers, perform
     Fourier series approximation and save result to INI file. 

     Input:
         - fname: filename (ini-file)
         - alpha, phi, theta: angles for kinematics approximation
         - nfft=[n1,n2,n3]: number of fourier coefficients for each angle
    """
    
    import matplotlib.pyplot as plt
    import easygui

    if len(nfft) != 3:
        raise ValueError("not the right number of fourier coefficients!")

    # open file, erase existing
    f = open( fname, 'w' )
    
    if header is not None:
        for h in header:
            f.write('; %s\n' % (h))

    f.write('[kinematics]\n')

    f.write('; if the format changes in the future\n')
    f.write('format=2015-10-09; currently unused\n')
    f.write('convention=flusi;\n')
    f.write('; what units, radiant or degree?\n')
    if unit_in == 'deg':
        f.write('units=degree;\n')
    else:
        f.write('units=rad;\n')
        
    f.write('; is this hermite or Fourier coefficients?\n')
    f.write('type=fourier;\n')

    i = 0
    for data, name in zip( [alpha,phi,theta], ['alpha','phi','theta']):
        
        if nfft[i] == -1:
            fig = plt.figure()
            
            # first guess:
            nfft[i] = 10
            choice = ""
            
            while choice != "happy":   
                fig.clf()
                ax = plt.gca()
                
                a0, ai, bi = fseries( data, nfft[i] )                
                data_fft = Fserieseval(a0, ai, bi, np.linspace(0.0, 1.0, data.shape[0], endpoint=False))
    
                ax.plot( data, 'k-')
                ax.plot( data_fft, 'r--')
#                ax.set_title('INTERACTIVE FOURIER FIT')
                fig.show()
                
                choice = easygui.buttonbox(title='Interactive fourier fit', 
                                           msg='figure shows the current fft approx with %i modes' % (nfft[i]),
                                           choices=('N=10', 'N=20', 'N=30', 'N+5', 'N+1', 'happy', 'N-1', 'N-5') )    
    
                if choice == "N+5":
                    nfft[i] += 5
                elif choice == "N+1":
                    nfft[i] += 1
                elif choice == "N-1":
                    nfft[i] -= 1
                elif choice == "N-5":
                    nfft[i] -= 5
                elif choice == "N=10":
                    nfft[i] = 10
                elif choice == "N=20":
                    nfft[i] = 20
                elif choice == "N=30":
                    nfft[i] = 30

        a0, ai, bi = fseries( data, nfft[i] )

        f.write('; %s\n' % (name))
        f.write('nfft_%s=%i;\n' % (name, nfft[i]) )

        f.write('a0_%s=%e;\n' % (name, a0) )
        f.write('ai_%s=' % (name))
        for k in range(nfft[i]-1):
            f.write('%e ' % (ai[k]))
        f.write('%e;\n' % (ai[nfft[i]-1]))


        f.write('bi_%s=' % (name) )
        for k in range(nfft[i]-1):
            f.write('%e ' % (bi[k]))
        f.write('%e;\n' % (bi[nfft[i]-1]))

        i += 1

    f.close()


def write_kinematics_ini_file_hermite(fname, alpha, phi, theta, alpha_dt, phi_dt, theta_dt, header=None):
    """
    Given the angles alpha, phi and theta and their time derivatives, create an 
    kinematics INI file for HERMITE approximation.
    
    Unlike the Fourier approximation, the Hermite files indeed contain the function values
    and derivatives directly. Interpolation in time is done when evaluating the file, e.g.,
    in WABBIT or FLUSI.
    
    In python, the evaluation is done using insect_tools.Hserieseval
    
    Input:
    ------
    
        fname: string
            output ini file name
        alpha, phi, theta: numpy arrays
            the kinematic angles, sampled between 0<=t<1.0 (excluding t=1.0 !!)
        alpha_dt, phi_dt, theta_dt: numpy arrays
            the kinematic angles time derivatives, sampled between 0<=t<1.0 (excluding t=1.0 !!)
       
    
    Output:
    -------
        Written to file directly.
    """


    # open file, erase existing
    f = open( fname, 'w' )
    
    if header is not None:
        f.write('; %s\n' % (header))

    f.write('[kinematics]\n')

    f.write('; if the format changes in the future\n')
    f.write('format=2015-10-09; currently unused\n')
    f.write('convention=flusi;\n')
    f.write('; what units, radiant or degree?\n')
    f.write('units=degree;\n')
    f.write('; is this hermite or Fourier coefficients?\n')
    f.write('type=hermite;\n')

    i = 0
    for data, data_dt, name in zip( [alpha,phi,theta], [alpha_dt,phi_dt,theta_dt], ['alpha','phi','theta']):

        f.write('; %s\n' % (name))
        f.write('nfft_%s=%i;\n' % (name, data.shape[0] ))

        # function values
        f.write('ai_%s=' % (name))
        for k in range(data.shape[0]-1):
            f.write('%e ' % (data[k]))
        f.write('%e;\n' % (data[-1]))

        # values of derivatives
        f.write('bi_%s=' % (name) )
        for k in range(data.shape[0]-1):
            f.write('%e ' % (data_dt[k]))
        f.write('%e;\n' % (data_dt[-1]))

        i += 1

    f.close()
  
    
def wing_contour_from_file(fname, N=1024):
    """
    Compute wing outline (shape) from an *.INI file. Returns: xc, yc, the coordinates
    of outline points, and the wings area (surface). Note: if a damage mask is applied,
    then the area returned here will not be the true (effective) area, but rather the one of
    an intact wing.
    """
    import os
    import inifile_tools
    from shapely.geometry.polygon import Polygon
    
    # does the ini file exist?
    if not os.path.isfile(fname):
        raise ValueError("Inifile: %s not found!" % (fname))
        
    if not inifile_tools.exists_ini_section(fname, "Wing"):
        raise ValueError("The ini file you specified does not contain the [Wing] section "+
                         "so maybe it is not a wing-shape file after all?")

    wtype = inifile_tools.get_ini_parameter(fname, "Wing", "type", str)
    
    if wtype != "fourier" and wtype != "linear" and wtype != 'kleemeier' and wtype != 'fourierY':
        print(wtype)
        raise ValueError("Not a fourier nor linear wing. This function currently only supports a "+
                         "Fourier or linear encoded wing (maybe with bristles)")
        
    x0 = inifile_tools.get_ini_parameter(fname, "Wing", "x0w", float, default=0.0)
    y0 = inifile_tools.get_ini_parameter(fname, "Wing", "y0w", float, default=0.0)
    
    #--------------------------------------------------------------------------
    # planform (contour)
    #--------------------------------------------------------------------------
    if wtype == "fourier":
        # description with fourier series
        a0 = inifile_tools.get_ini_parameter(fname, "Wing", "a0_wings", float)
        ai = inifile_tools.get_ini_parameter(fname, "Wing", "ai_wings", float, vector=True)
        bi = inifile_tools.get_ini_parameter(fname, "Wing", "bi_wings", float, vector=True)
        
        # compute outer wing shape (membraneous part)
        theta2 = np.linspace(-np.pi, np.pi, num=N, endpoint=False)
        r_fft = Fserieseval(a0, ai, bi, (theta2 + np.pi) / (2.0*np.pi) )
        
        xc = x0 + np.cos(theta2)*r_fft
        yc = y0 + np.sin(theta2)*r_fft
        
        area = Polygon( zip(xc,yc) ).area      
        
    elif wtype == "linear":
        # description with points (not a radius)
        R_i = inifile_tools.get_ini_parameter(fname, "Wing", "R_i", float, vector=True)
        theta_i = inifile_tools.get_ini_parameter(fname, "Wing", "theta_i", float, vector=True) 
        # A word on theta:
        # Python does imply [-pi:+pi] but WABBIT uses 0:2*pi
        # Therefore, we subtract pi here (because data are in WABBIT convention)
        theta_i = theta_i - np.pi
        
        # down- or upsampling
        theta_i_new = np.linspace( start=theta_i[0], stop=theta_i[-1], num=N, endpoint=True)
        R_i_new = np.interp(theta_i_new, theta_i, R_i)
        
        R_i = R_i_new.copy()
        theta_i = theta_i_new.copy()
    
        xc = x0 + np.cos(theta_i)*R_i
        yc = y0 + np.sin(theta_i)*R_i        
        
        # there was a bug in here somewhere but i am too lazy to find it ... now I just
        # use polygon.area and am less unhappy. TE 12/feb/2025
        area = Polygon( zip(xc,yc) ).area
        
    elif wtype == 'fourierY':
        # used for the 2021 Nature Paper (Paratuposa)
        a0 = inifile_tools.get_ini_parameter(fname, "Wing", "a0_wings", float)
        ai = inifile_tools.get_ini_parameter(fname, "Wing", "ai_wings", float, vector=True)
        bi = inifile_tools.get_ini_parameter(fname, "Wing", "bi_wings", float, vector=True)
                
        Rblade = inifile_tools.get_ini_parameter(fname, "Wing", "y0w", float)
        
        y = np.linspace(0, Rblade, 100)
        theta = np.arccos(1.0 - 2.0*y/Rblade)
        
        xle = Fserieseval(a0, ai, bi, (theta + np.pi) / (2.0*np.pi) )
        xte = Fserieseval(a0, ai, bi, 1-(theta + np.pi) / (2.0*np.pi) )
                
        xc = np.hstack([xle,xte[::-1]])
        yc = np.hstack([y[::-1],y])
        area = Polygon( zip(xc,yc) ).area
        
    elif wtype == "kleemeier":
        B, H = 8.6/130, 100/130
        xc = [-B/2, -B/2, +B/2, +B/2, +B/2]
        yc = [0.0, H, H, 0.0, 0.0]
        area = B*H
  
    return xc, yc, area

def compute_wing_inertia_tensor(fname, density_membrane=1.0, density_bristles=0.0, dx=1e-3):
    """
    COmpute the inertia tensor of wing INI file.

    Parameters
    ----------
    fname : path to inifile
        File to read the wing geometry from.
    density_membrane : float
        Density of the membrane [mass/R**2]. The output will be in the same unit. If you pass kg/R**2, then the 
        inertia tensor will also be in kg*R**2. Note you'd M_wing (in kg) and divide it by area returned by wing_contour_from_file.
        The density then had the dimension kg/R² -
        You can also pass unity and add units afterwards.
    density_bristles : float
        Density of bristles in mass/R (mass per lengh). If you 
        The default is 0.0 - this neglects bristles in inertia computation. You shall use the same unit as
        for the membrane density. 

    Returns
    -------
    Jxx, Jyy, Jzz, Jxy

    """
    import inifile_tools
    
    
    # evaluate wing file
    x, y = get_wing_membrane_grid(fname, dx=dx, dy=dx)
    # assume a thin wing
    z = x * 0.0
    
    Jxx = np.sum( y**2 + z**2 )*dx*dx*density_membrane
    Jyy = np.sum( x**2 + z**2 )*dx*dx*density_membrane
    Jzz = np.sum( x**2 + y**2 )*dx*dx*density_membrane
    Jxy = np.sum( x*y )*dx*dx*density_membrane
    
    del x,y,z 
    
    # does the wing have bristled?
    if inifile_tools.get_ini_parameter(fname, "Wing", "bristles", bool, default=False):
        # read in the bristles array
        bristles_coords = inifile_tools.get_ini_parameter(fname, "Wing", "bristles_coords", matrix=True)
        
        for j in range( bristles_coords.shape[0]):
            x0 = bristles_coords[j,0]
            y0 = bristles_coords[j,1]          
            x1 = bristles_coords[j,2]
            y1 = bristles_coords[j,3]  
            
            # unit vector in bristle direction
            ex, ey = x1-x0, y1-y0
            L = np.sqrt(ex**2 + ey**2)
            ex /= L
            ey /= L
            dL = dx
            
            l = np.linspace(0, L, int(np.round(L/dL)) )
            # print(l.shape)
            
            for i in range(l.shape[0]):
                x, y, z = x0 + ex*float(i)*dL, y0 + ey*float(i)*dL, 0.0
                
                Jxx += ( y**2 + z**2 )*dL*density_bristles
                Jyy += ( x**2 + z**2 )*dL*density_bristles
                Jzz += ( x**2 + y**2 )*dL*density_bristles
                Jxy += ( x*y )*dL*density_bristles
                # Jxy += dL
                
    
    # note how Jxz and Jyz are zero by the thin wing assumption.
    return np.asarray([Jxx, Jyy, Jzz, Jxy])
    

def compute_wing_geom_factors(fname):
    """
    Compute geometrical factors for a wing-shape *.ini file. 
    
    To be extended in the future.
    
    returns area, S1, S2, 
    """
    dx, dy = 1e-3, 1e-3 
    
    # evaluate wing file
    x, y = get_wing_membrane_grid(fname, dx=dx, dy=dy)
    
    area = np.sum( y**0 * dx*dy)
    S1 = np.sum( y**1 * dx*dy)
    S2 = np.sum( y**2 * dx*dy)
    
    return area, S1, S2

    
def visualize_wing_shape_file(fname, ax=None, fig=None, savePNG=True, fill=False, fillAlpha=0.15, 
                              savePDF=False, color_contour='r', color_fill_mask='k'):
    """
    Reads in a wing shape ini file and visualizes the wing as 2D plot.
    
    Input:
    ------
    
        fname: string
            ini file name describing the wing shape. It must contan the [Wing] section
            and describe a Fourier wing. The wing may have bristles, no problem.
            
    Output:
    -------
        Written to file (png/svg/pdf) directly.
    """
    import os
    import inifile_tools
    import matplotlib.pyplot as plt
    from shapely.geometry.polygon import Polygon
    from shapely.geometry import Point
    
    
    plt.rcParams["text.usetex"] = False

    
    
    # open the new figure:
    if fig is None and ax is None:
        fig = plt.figure()    
        
    if ax is None:
        ax = plt.gca()
        
    # -------------------------------------------------------------------------
    # damage (if present)
    # Drawn first as a background
    # -------------------------------------------------------------------------  
    damaged = inifile_tools.get_ini_parameter(fname, "Wing", "damaged", bool, default=False)
    
    if damaged:
        # actual 0/1 damage mask:
        mask = inifile_tools.get_ini_parameter( fname, 'Wing', 'damage_mask', dtype=float, vector=False, default=None, matrix=True )
        mask = mask.T
        
        # the bounding box of the mask is set here:
        bbox = inifile_tools.get_ini_parameter( fname, 'Wing', 'corrugation_array_bbox', dtype=float, vector=True, default=None )
        
        n1, n2 = mask.shape[0], mask.shape[1]
        
        x1, x2 = np.linspace(bbox[0], bbox[1], num=n2, endpoint=True), np.linspace(bbox[2], bbox[3], num=n1, endpoint=True)
        X, Y = np.meshgrid(x2, x1)
        
        plt.contourf(Y.T, X.T, mask, levels=[0.5, 1.0], colors=[color_fill_mask])
        
        dx, dy = x1[1]-x1[0], x2[1]-x2[0]
        
        
        # evaluate wing shape file.
        xc, yc, area = wing_contour_from_file(fname)         
        # create a polygon with the wing contour
        polygon = Polygon( zip(xc,yc) )        
        
        # compute area of damaged wing (=intersection of damage mask and contour)
        # loop over all grid points (2D grid and check if the point is inside the polygon)
        area_damaged = 0.0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if polygon.contains( Point(Y.T[i,j], X.T[i,j]) ) and (mask[i,j]>0.0):
                    # yes, the point is inside the contour and its damage mask is >0
                    area_damaged += dx*dy 
                            
    # -------------------------------------------------------------------------  
    # contour
    # -------------------------------------------------------------------------    
    xc, yc, area = wing_contour_from_file(fname)
            
    # plots wing outline
    ax.plot( xc, yc, '-', color=color_contour, label='wing')
    
    if fill:
        color = change_color_opacity(color_contour, fillAlpha)
        ax.fill( np.append(xc, xc[0]), np.append(yc, yc[0]), color=color )
    
    ax.axis('equal')
    
    title = "wing shape visualization: \n%s\nA=%2.2f AR=%2.2f" % (fname, area, 1.0/area)
    
    if damaged:
        title += ' A_damaged=%f (%2.2f%%)' % (area_damaged, 100*area_damaged/area)
    
    # draw rotation axis a bit longer than the wing
    d = 0.1
    # plot the rotation axes
    ax.plot( [np.min(xc)-d, np.max(xc)+d], [0.0, 0.0], 'k--', label='rotation axis ($x^{(w)}$, $y^{(w)}$)')
    ax.plot( [0.0, 0.0], [np.min(yc)-d, np.max(yc)+d], 'k--')
    # ax.grid()
    ax.legend()
    ax.set_title(title)
    
    # -------------------------------------------------------------------------
    # bristles (if present)
    # -------------------------------------------------------------------------
    # if present, add bristles    
    bristles = inifile_tools.get_ini_parameter(fname, "Wing", "bristles", bool, default=False)
    if bristles:
        bristles_coords = inifile_tools.get_ini_parameter(fname, "Wing", "bristles_coords", matrix=True)
        print(bristles_coords.shape)
        for j in range( bristles_coords.shape[0]):
            ax.plot( [bristles_coords[j,0], bristles_coords[j,2]], [bristles_coords[j,1], bristles_coords[j,3]], 'r-')
    
    # -------------------------------------------------------------------------
    # save to image file
    # -------------------------------------------------------------------------
    if savePNG:
        plt.savefig( fname.replace('.ini','')+'_shape.png', dpi=300 )
    if savePDF:
        plt.savefig( fname.replace('.ini','')+'_shape.pdf', dpi=300 )
        
        
def piecewise_linear_function(x, xi, fi, periodize=True ):
    """
    Piecewise linear function defined by fi(xi), evaluated at x 

    Returns
    -------
    x - the function evaluated at x
    """
    
    if periodize:
        xi = np.hstack( (np.asarray(xi)-1.0, np.asarray(xi), np.asarray(xi)+1.0) )
        fi = np.hstack( (np.asarray(fi), np.asarray(fi), np.asarray(fi)) )
    
    return np.interp(x, xi, fi)
    
def musca_kinematics_model( PHI, phi_m, dTau=0.03, alpha_down=61.0, alpha_up=-37.0, time=None ):
    """
    Kinematics model for musca wing with 3 parameters (used for compensation model + 2 parameters, the angle
    of attack during up an downstroke). 

    Parameters
    ----------
    PHI : float, scalar
        Stroke amplitude
    phi_m : float, scalar
        Mean stroke angle
    dTau : float, scalar
        Delay parameter of supination/pronation
    alpha_down : float, scalar, optional
        Featherng angle during downstroke. The default is 61.0.
    alpha_up : float, scalar, optional
        Feathering angle during upstroke. The default is -37.0.
    time : vector of time, optional
        Time vector. The default is 1000 samples between 0 and 1.
    

    Returns
    -------
    time, alpha, phi, theta
    """
    
    if time is None:
        time = np.linspace(0.0, 1.0, endpoint=False, num=1000)
        
    ai = [-5.96937, -5.91301, -5.77952, -5.56493, -5.27093, -4.90575, -4.48439, -4.02806, -3.56276, -3.1172, -2.72019, -2.39778, -2.1705, -2.05098, -2.04232, 
          -2.13731, -2.3188, -2.56108, -2.83215, -3.0968, -3.31993, -3.47001, -3.52207, -3.46009, -3.27838, -2.98191, -2.58555, -2.11235, -1.59104, -1.05315, 
          -0.530025, -0.0501387, 0.363058, 0.692459, 0.927748, 1.06495, 1.10525, 1.05327, 0.915113, 0.696493, 0.401239, 0.0303748, -0.418047, -0.948353, 
          -1.56597, -2.27564, -3.07944, -3.97478, -4.9528, -5.99741, -7.08508, -8.18545, -9.26293, -10.2788, -11.194, -11.972, -12.5815, -12.9987, -13.2093, 
          -13.2092, -13.0046, -12.6108, -12.0514, -11.3555, -10.5557, -9.6858, -8.77847, -7.8637, -6.96751, -6.11136, -5.312, -4.58185, -3.92959, -3.36088, 
          -2.87913, -2.48605, -2.18201, -1.9662, -1.8365, -1.78934, -1.81942, -1.91963, -2.08103, -2.29315, -2.54446, -2.82304, -3.11744, -3.41739, -3.71449, 
          -4.00265, -4.27809, -4.53909, -4.78538, -5.01718, -5.23424, -5.43486, -5.6151, -5.76847, -5.88597, -5.95678]
    
    bi = [2.03178, 9.3903, 17.382, 25.507, 33.1457, 39.6195, 44.2673, 46.5268, 46.0121, 42.5768, 36.3521, 27.7563, 17.4694, 6.37686, -4.51359, -14.173, 
          -21.6587, -26.2157, -27.3597, -24.9324, -19.1211, -10.4408, 0.319376, 12.1794, 24.0718, 34.9513, 43.9008, 50.2174, 53.4725, 53.5371, 50.5714, 
          44.9815, 37.3492, 28.3461, 18.6422, 8.82289, -0.676032, -9.60717, -17.9193, -25.7343, -33.2979, -40.9111, -48.8534, -57.3084, -66.3036, -75.6722, 
          -85.0437, -93.8651, -101.45, -107.049, -109.936, -109.495, -105.299, -97.1696, -85.2116, -69.8176, -51.6409, -31.5412, -10.5105, 10.4139, 30.2368,
          48.0847, 63.2685, 75.3235, 84.0231, 89.3685, 91.5558, 90.9287, 87.9223, 83.0074, 76.6416, 69.2331, 61.1194, 52.5624, 43.7574, 34.852, 25.9693, 
          17.2307, 8.7729, 0.755196, -6.64331, -13.237, -18.8542, -23.3612, -26.6843, -28.8255, -29.8681, -29.9696, -29.3418, -28.2219, -26.8357, -25.362, 
          -23.9012, -22.4561, -20.927, -19.1259, -16.8057, -13.7049, -9.59889, -4.3532 ]
    
    theta = Hserieseval(0.0, np.asarray(ai), np.asarray(bi), time)
    
    ai = [71.9014, 72.091, 72.0661, 71.8312, 71.3914, 70.7516, 69.917, 68.893, 67.6849, 66.298, 64.7376, 63.0089, 61.117, 59.067, 56.8639, 54.5128, 52.0187, 
          49.3867, 46.6221, 43.7303, 40.7169, 37.588, 34.3499, 31.0095, 27.5741, 24.0517, 20.4509, 16.7809, 13.0517, 9.27419, 5.45988, 1.62113, -2.22892, 
          -6.07636, -9.90655, -13.7042, -17.4533, -21.1374, -24.7396, -28.2427, -31.629, -34.8811, -37.9814, -40.9124, -43.6572, -46.1991, -48.5223, -50.6115, 
          -52.4526, -54.0325, -55.3394, -56.3628, -57.0937, -57.5248, -57.6504, -57.4667, -56.9717, -56.1653, -55.0496, -53.6282, -51.9071, -49.8939, -47.5984, 
          -45.0319, -42.2076, -39.1404, -35.8466, -32.3438, -28.6512, -24.7887, -20.7773, -16.6388, -12.3954, -8.06993, -3.68517, 0.735928, 5.17061, 9.59643, 
          13.9915, 18.3344, 22.6048, 26.7831, 30.8509, 34.7906, 38.5864, 42.2233, 45.688, 48.9685, 52.0542, 54.9361, 57.6063, 60.0587, 62.2882, 64.2912, 66.0653,
          67.6092, 68.9227, 70.0066, 70.8627, 71.4932 ]
    
    bi = [29.8242, 8.16142, -13.0625, -33.8136, -54.0663, -73.8029, -93.0125, -111.69, -129.834, -147.45, -164.541, -181.114, -197.177, -212.734, -227.789, 
          -242.341, -256.386, -269.914, -282.91, -295.353, -307.213, -318.455, -329.038, -338.912, -348.022, -356.306, -363.697, -370.124, -375.512, -379.783,
          -382.857, -384.656, -385.103, -384.121, -381.641, -377.596, -371.93, -364.592, -355.542, -344.752, -332.205, -317.898, -301.841, -284.06, -264.595,
          -243.502, -220.853, -196.736, -171.253, -144.52, -116.669, -87.8447, -58.2024, -27.909, 2.85937, 33.9192, 65.081, 96.1511, 126.934, 157.232, 186.853,
          215.603, 243.3, 269.763, 294.825, 318.326, 340.123, 360.081, 378.086, 394.036, 407.847, 419.454, 428.808, 435.88, 440.659, 443.151, 443.381, 441.389,
          437.235, 430.99, 422.741, 412.588, 400.641, 387.022, 371.858, 355.285, 337.441, 318.469, 298.513, 277.716, 256.22, 234.161, 211.674, 188.884, 165.911,
          142.867, 119.855, 96.9658, 74.2843, 51.8828 ]
    
    phi = Hserieseval(0.0, np.asarray(ai), np.asarray(bi), time)
    
    
    # modify phi with input parameters
    phi = phi - np.mean(phi)
    phi *= PHI / (np.max(phi)-np.min(phi))
    phi += phi_m
    
    # theta is not modified 
    theta = theta
    
    # alpha is a new function
    # d_tau is the timing of pronation and supination, which Muijres2016 identified as important parameter in the compensation
    # degrees
    alpha_tau  = 0.225 # fixed parameters (rotation duration) upstroke->downstroke
    alpha_tau1 = 0.200 # downstroke->upstroke
    
    T1 = alpha_tau1/2.0
    T2 = 0.5 - alpha_tau/2.0
    T3 = T2  + alpha_tau
    T4 = 1.0 - alpha_tau1/2.0
    TT = T4
    
    pi = np.pi
    a  = (alpha_up-alpha_down)/alpha_tau     
    a1 = (alpha_up-alpha_down)/alpha_tau1   
    
    alpha = np.zeros(time.shape)
    
    # Masks
    m1 = time < T1
    m2 = (time >= T1) & (time < T2)
    m3 = (time >= T2) & (time < T3)
    m4 = (time >= T3) & (time < T4)
    m5 = time >= T4
    
    # Vectorized assignments
    alpha[m1] = alpha_down - a1 * (time[m1] - alpha_tau1 / 2.0 - (alpha_tau1 / (2.0 * pi)) * np.sin(2.0 * pi * (time[m1] - alpha_tau1 / 2.0) / alpha_tau1))
    alpha[m2] = alpha_down
    alpha[m3] = alpha_down + a * (time[m3] - T2 - (alpha_tau / (2.0 * pi)) * np.sin(2.0 * pi * (time[m3] - T2) / alpha_tau))
    alpha[m4] = alpha_up
    alpha[m5] = alpha_up - a1 * (time[m5] - TT - (alpha_tau1 / (2.0 * pi)) * np.sin(2.0 * pi * (time[m5] - TT) / alpha_tau1))
   
      
    # this now is the important part that circularily shifts the entire vector. 
    # it thus changes the "timing of pronation and supination"
    dt = time[1]-time[0]
    shift = int( np.round(dTau/dt) )
    alpha = np.roll(alpha, shift)

    return time, alpha, phi, theta



def bumblebee_kinematics_model( PHI=115.0, phi_m=24.0, dTau=0.00, alpha_down=70.0, alpha_up=-40.0, tau=0.22, theta=12.55/2, time=None):
    """
    Kinematics model for a bumblebee bombus terrestris [Engels et al PRL 2016, PRF 2019]

    Note motion starts with downstroke. Defaults are set to values used in [Engels et al PRL 2016, PRF 2019].
    
    Alpha is piecewise constant with sin transition, theta is constant and phi is sinusoidal.

    Parameters
    ----------
    PHI : float, scalar
        Stroke amplitude (deg)
    phi_m : float, scalar
        Mean stroke angle (deg)
    dTau : float, scalar
        Delay parameter of supination/pronation
    alpha_down : float, scalar, optional
        Featherng angle during downstroke. (deg)
    alpha_up : float, scalar, optional
        Feathering angle during upstroke. (deg)
    tau : 
        duration of wing rotation
    theta : 
        constant deviation angle (deg)
    time : vector of time, optional
        Time vector. The default is 1000 equidistant samples between 0 and 1.
    

    Returns
    -------
    time, alpha, phi, theta
    """
    
    if time is None:
        time = np.linspace(0.0, 1.0, endpoint=False, num=1000)

    # phi is sinusoidal function with fixed phase (variable amplitude+offset)
    phi = phi_m + (PHI/2.0)*np.sin(2.0*np.pi*(time+0.25))
    # theta is a constant value
    theta = np.zeros_like(time) + theta
    
    # alpha is a new function
    # d_tau is the timing of pronation and supination, which Muijres2016 identified as important parameter in the compensation
    # degrees
    alpha_tau  = tau # fixed parameters (rotation duration) upstroke->downstroke
    alpha_tau1 = tau # downstroke->upstroke
    
    T1 = alpha_tau1/2.0
    T2 = 0.5 - alpha_tau/2.0
    T3 = T2  + alpha_tau
    T4 = 1.0 - alpha_tau1/2.0
    
    pi = np.pi
    a  = (alpha_up-alpha_down)/alpha_tau     
    a1 = (alpha_up-alpha_down)/alpha_tau1   
    
    alpha = np.zeros_like(time)
   
    for it, t in enumerate(time):
        if t < T1:
            alpha[it] = alpha_down - a1*(  t-alpha_tau1/2.0 - (alpha_tau1/2.0/pi) * np.sin(2.0*pi*(t-alpha_tau1/2.0)/alpha_tau1)    )
            
        elif t>=T1 and t < T2:
            alpha[it] = alpha_down
                            
        elif t>=T2 and t < T3:
            alpha[it] = alpha_down + a*(  t-T2 - (alpha_tau/2/pi)*np.sin(2*pi*(t-T2)/alpha_tau)    )
            
        elif t>=T3 and t < T4:
            alpha[it] = alpha_up
            
        elif t >= T4:
            TT = 1.0-alpha_tau1/2.0
            alpha[it] = alpha_up - a1*(  t-TT - (alpha_tau1/2/pi) * np.sin(2*pi*( (t-TT)/alpha_tau1)    ) )
       
    # this now is the important part that circularily shifts the entire vector. 
    # it thus changes the "timing of pronation and supination"
    dt = time[1]-time[0]
    shift = int( np.round(dTau/dt) )
    alpha = np.roll(alpha, shift)

    return time, alpha, phi, theta


def compute_aero_power_individual_wings(run_directory, file_output='aero_power_individual.csv'):
    """
    Post-processing: WABBIT outputs the aerodynamic power in the file aero_power.t
    but this file contains the power of all (up to four) wings together. This small python
    routine computes instead the power for each wing individually and stores the result as
    a CSV file

    Parameters
    ----------
    run_directory : TYPE
        Directory of the CFD simulation. We look for the PARAMS.ini file, as well as the *.t files there.
    file_output : *.csv file to dump the result of the calculation to.
    """
    import matplotlib.pyplot as plt
    
    # indices [zero-based] in kinematics.t:
    # d[:,0] time
    # d[:,1] xc_body_g_x
    # d[:,2] xc_body_g_y
    # d[:,3] xc_body_g_z
    # d[:,4] psi
    # d[:,5] beta
    # d[:,6] gamma
    # d[:,7] eta_stroke
    # d[:,8] alpha_l
    # d[:,9] phi_l
    # d[:,10] theta_l
    # d[:,11] alpha_r
    # d[:,12] phi_r
    # d[:,13] theta_r
    # d[:,14] rot_rel_l_w_x
    # d[:,15] rot_rel_l_w_y
    # d[:,16] rot_rel_l_w_z
    # d[:,17] rot_rel_r_w_x
    # d[:,18] rot_rel_r_w_y
    # d[:,19] rot_rel_r_w_z
    # d[:,20] rot_dt_l_w_x
    # d[:,21] rot_dt_l_w_y
    # d[:,22] rot_dt_l_w_z
    # d[:,23] rot_dt_r_w_x
    # d[:,24] rot_dt_r_w_y
    # d[:,25] rot_dt_r_w_z
    # d[:,26] alpha_l2
    # d[:,27] phi_l2
    # d[:,28] theta_l2
    # d[:,29] alpha_r2
    # d[:,30] phi_r2
    # d[:,31] theta_r2
    # d[:,32] rot_rel_l2_w_x
    # d[:,33] rot_rel_l2_w_y
    # d[:,34] rot_rel_l2_w_z
    # d[:,35] rot_rel_r2_w_x
    # d[:,36] rot_rel_r2_w_y
    # d[:,37] rot_rel_r2_w_z
    # d[:,38] rot_dt_l2_w_x
    # d[:,39] rot_dt_l2_w_y
    # d[:,40] rot_dt_l2_w_z
    # d[:,41] rot_dt_r2_w_x
    # d[:,42] rot_dt_r2_w_y
    # d[:,43] rot_dt_r2_w_z
    
    d = load_t_file(run_directory+'/kinematics.t')
    
    if d.shape[1] > 25:
        # four wings
        wings = ['leftwing', 'rightwing', 'leftwing2', 'rightwing2']
    else:
        # two wings
        wings = ['leftwing', 'rightwing']
        
    nt = d.shape[0]
    all_power = np.zeros( (nt,5) )
        
    for ID, wing in enumerate(wings):
        m = load_t_file(run_directory+'/moments_'+wing+'.t')
        
        
        for it in range(nt):
            # body angles
            psi, beta, gamma, eta = d[it,4], d[it,5], d[it,6], d[it,7]
            
            if wing == 'leftwing':
                ia, ip, i2, ix, iy, iz, side = 8, 9, 10, 14, 15, 16, 'left'
            elif wing == 'rightwing':
                ia, ip, i2, ix, iy, iz, side = 11, 12, 13, 17, 18, 19, 'right'
            elif wing == 'leftwing2':
                ia, ip, i2, ix, iy, iz, side = 26, 27, 28, 32, 33, 34, 'left'
            elif wing == 'rightwing2':
                ia, ip, i2, ix, iy, iz, side = 29, 30, 31, 35, 36, 37, 'right'
                
            # wing angles
            alpha, phi, theta = d[it,ia], d[it,ip], d[it,i2]
            # angular velocity vector (in wing system)
            omega_wing_w = np.asarray([d[it,ix], d[it,iy], d[it,iz]])
                
            # moment in global system
            T_g = np.asarray(m[it, 1:3+1])
            # rotation matrix
            M_g2w = get_M_b2w(alpha, theta, phi, eta, side) @ get_M_g2b(psi, beta, gamma)
            # moment in wing system
            T_w = M_g2w @ T_g
            
            power = -np.dot( T_w.T, omega_wing_w )
            
            all_power[it, 0], all_power[it, ID+1] = d[it,0], power
    
    power_control = load_t_file( run_directory+'/aero_power.t')
    
    # plt.figure()
    # plt.plot(np.sum(all_power[:,1:], axis=1))
    # plt.plot(power_control[:,1])
    
    if np.max( power_control[:,1] - np.sum(all_power[:,1:], axis=1)) > 1e-4:
        raise ValueError("something is wrong")
    
    write_csv_file(run_directory+'/'+file_output, all_power, header=['time','aero power leftwing','aero power rightwing','aero power leftwing2','aero power rightwing2'])
    
    

def compute_inertial_power(file_kinematics='kinematics.t', Jxx=0.0, Jyy=0.0, Jzz=0.0, Jxy=0.0):
    """
    Post-processing: Compute the insects inertial power. 
    
    The routine reads the kinematics.t file (for angular velocity and acceleration
    of the wings). The inertia tensor can be passed to this routine
    The background is that often, during the simulation, we do not bother to compute the
    inertia tensor, and it is thus not contained in the PARAMS.ini files
    
    Input:
    ------
    
        fname: string
            ini file name describing the wing shape. It must contan the [Wing] section
            and describe a Fourier wing. The wing may have bristles, no problem.
            
    Output:
    -------
        written to inertial_power_postprocessing.t
    """
    
    # indices [zero-based] in kinematics.t:
    # d[:,0] time
    # d[:,1] xc_body_g_x
    # d[:,2] xc_body_g_y
    # d[:,3] xc_body_g_z
    # d[:,4] psi
    # d[:,5] beta
    # d[:,6] gamma
    # d[:,7] eta_stroke
    # d[:,8] alpha_l
    # d[:,9] phi_l
    # d[:,10] theta_l
    # d[:,11] alpha_r
    # d[:,12] phi_r
    # d[:,13] theta_r
    # d[:,14] rot_rel_l_w_x
    # d[:,15] rot_rel_l_w_y
    # d[:,16] rot_rel_l_w_z
    # d[:,17] rot_rel_r_w_x
    # d[:,18] rot_rel_r_w_y
    # d[:,19] rot_rel_r_w_z
    # d[:,20] rot_dt_l_w_x
    # d[:,21] rot_dt_l_w_y
    # d[:,22] rot_dt_l_w_z
    # d[:,23] rot_dt_r_w_x
    # d[:,24] rot_dt_r_w_y
    # d[:,25] rot_dt_r_w_z
    # d[:,26] alpha_l2
    # d[:,27] phi_l2
    # d[:,28] theta_l2
    # d[:,29] alpha_r2
    # d[:,30] phi_r2
    # d[:,31] theta_r2
    # d[:,32] rot_rel_l2_w_x
    # d[:,33] rot_rel_l2_w_y
    # d[:,34] rot_rel_l2_w_z
    # d[:,35] rot_rel_r2_w_x
    # d[:,36] rot_rel_r2_w_y
    # d[:,37] rot_rel_r2_w_z
    # d[:,38] rot_dt_l2_w_x
    # d[:,39] rot_dt_l2_w_y
    # d[:,40] rot_dt_l2_w_z
    # d[:,41] rot_dt_r2_w_x
    # d[:,42] rot_dt_r2_w_y
    # d[:,43] rot_dt_r2_w_z
   
    # k = load_t_file( file_kinematics )
    
    # for it in range(k.shape[0]):
        
    #     rot_dt_wing_l_w1, rot_dt_wing_l_w2, rot_dt_wing_l_w3 = k[it,20], k[it,21], k[it,22]
    #     rot_dt_wing_r_w1, rot_dt_wing_r_w2, rot_dt_wing_r_w3 = k[it,23], k[it,24], k[it,25]
        
    #     rot_rel_wing_l_w1, rot_rel_wing_l_w2, rot_rel_wing_l_w3 = k[it,14], k[it,15], k[it,16]
    #     rot_rel_wing_r_w1, rot_rel_wing_r_w2, rot_rel_wing_r_w3 = k[it,17], k[it,18], k[it,19]
        
    #     #-- LEFT WING
    #     a1 = Jxx * rot_dt_wing_l_w1 + Jxy * rot_dt_wing_l_w2
    #     a2 = Jxy * rot_dt_wing_l_w1 + Jyy * rot_dt_wing_l_w2
    #     a3 = Jzz * rot_dt_wing_l_w3
        
    #     b1 = Jxx * rot_rel_wing_l_w1 + Jxy * rot_rel_wing_l_w2
    #     b2 = Jxy * rot_rel_wing_l_w1 + Jyy * rot_rel_wing_l_w2
    #     b3 = Jzz * rot_rel_wing_l_w3
        
    #     iwmoment1 = (a1 + rot_rel_wing_l_w2*b3 - rot_rel_wing_l_w3*b2)
    #     iwmoment2 = (a2 + rot_rel_wing_l_w3*b1 - rot_rel_wing_l_w1*b3)
    #     iwmoment3 = (a3 + rot_rel_wing_l_w1*b2 - rot_rel_wing_l_w2*b1)
        
    #     inertial_power_left = rot_rel_wing_l_w1 * iwmoment1 + rot_rel_wing_l_w2 * iwmoment2 + rot_rel_wing_l_w3 * iwmoment3
        
    #     #-- RIGHT WING
    #     a1 = Jxx * rot_dt_wing_r_w1 + Jxy * rot_dt_wing_r_w2
    #     a2 = Jxy * rot_dt_wing_r_w1 + Jyy * rot_dt_wing_r_w2
    #     a3 = Jzz * rot_dt_wing_r_w3
        
    #     b1 = Jxx * rot_rel_wing_r_w1 + Jxy * rot_rel_wing_r_w2
    #     b2 = Jxy * rot_rel_wing_r_w1 + Jyy * rot_rel_wing_r_w2
    #     b3 = Jzz * rot_rel_wing_r_w3
        
    #     iwmoment1 = (a1 + rot_rel_wing_r_w2*b3 - rot_rel_wing_r_w3*b2)
    #     iwmoment2 = (a2 + rot_rel_wing_r_w3*b1 - rot_rel_wing_r_w1*b3)
    #     iwmoment3 = (a3 + rot_rel_wing_r_w1*b2 - rot_rel_wing_r_w2*b1)
        
    #     inertial_power_right = rot_rel_wing_r_w1 * iwmoment1 + rot_rel_wing_r_w2 * iwmoment2 + rot_rel_wing_r_w3 * iwmoment3

def get_wing_membrane_grid(fname, dx=1e-3, dy=1e-3):
    """
    Get a list of (x,y) points on the wing membrane, but on a regular grid with spacing
    dx, dy. This function is convenient when performing geometrical tasks (integration of S2 etc)
    on the wing. 
    
    NOTE: we also have a version with randomized points.    

    Parameters
    ----------
    fname : inifile
        Wing shape file.
    dx,dy : float
        Lattice spacing in both directions.

    Returns
    -------
    x, y : list
        A list of points on a regular, cartesian grid that are inside the wing contour.

    """
    
    from shapely.geometry.polygon import Polygon
    from shapely.geometry import Point
    import numpy as np
    import inifile_tools
    
    damaged = inifile_tools.get_ini_parameter(fname, "Wing", "damaged", bool, default=False)
    
    if damaged:
        raise ValueError("This function is not yet ready for damaged wings!!!!")
    
    # evaluate wing shape file.
    xc, yc, area = wing_contour_from_file(fname) 
    
    # create a polygon with the wing contour
    polygon = Polygon( zip(xc,yc) )
    
    
    # target grid
    x_grid = np.linspace( np.min(xc), np.max(xc), int(np.round( (np.max(xc)- np.min(xc))/dx )) )
    y_grid = np.linspace( np.min(yc), np.max(yc), int(np.round( (np.max(yc)- np.min(yc))/dy )) )
    
    # initialize lists to return
    x, y = [], []
    
    # loop over all grid points (2D grid and check if the point is inside the polygon)
    for i in range(x_grid.shape[0]):
        for j in range(y_grid.shape[0]):
            if polygon.contains( Point(x_grid[i], y_grid[j]) ):
                # yes, the point is on the membrane -> add it to the return list
                x.append(x_grid[i])
                y.append(y_grid[j])   
                
    return np.asarray(x), np.asarray(y)

def get_wing_pointcloud_from_inifile(fname, N=1000, N_contour=128):
    """
    Returns a list of points (x,y) that are on the wing. The coordinates are 
    understood in the wing frame of reference (w), where 
        x is from trailing to leading edge 
        
        y is from root to tip
        
    If the wing is bristled, we include the start- and endpoints of all bristles.
    The points on the wings membrane are uniformily distributed. The wing outline
    is included in the list of points.
    
    This function is useful for collision checking.

    Parameters
    ----------
    fname : string
        ini file describing the wing
    

    Returns
    -------
    xw_w_pointcloud : array of size [N, 3]
        The size of the array might vary.

    """   
    
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import shapely.plotting
    import numpy as np
    import inifile_tools
    
    x, y = [], []
    
    # membrane contour
    x_membrane, y_membrane, area_membrane = wing_contour_from_file(fname, N=N_contour)
    
    # find points on membrane
    poly_list = []
    for i in range(x_membrane.shape[0]):
        poly_list.append( (x_membrane[i], y_membrane[i]) )
        
        x.append(x_membrane[i])
        y.append(y_membrane[i])
        
    
    polygon = Polygon(poly_list)
    # shapely.plotting.plot_polygon(polygon)
        
    # N = 50000
    # x_rand, y_rand = np.random.uniform(-1.0, 1.0, size=N), np.random.uniform(-1.0, 1.0, size=N)
        
    for i in range(N-N_contour):
        x_rand, y_rand = np.random.uniform(-1.0, 1.0, size=1)[0], np.random.uniform(-1.0, 1.0, size=1)[0]
        
        while not polygon.contains( Point(x_rand, y_rand) ):
            x_rand, y_rand = np.random.uniform(-1.0, 1.0, size=1)[0], np.random.uniform(-1.0, 1.0, size=1)[0]
            
        # if polygon.contains( Point(x_rand[i], y_rand[i]) ):
        x.append(x_rand)
        y.append(y_rand)
       
    
    # if present, add bristles    
    if inifile_tools.get_ini_parameter(fname, "Wing", "bristles", bool, default=False):
        # read in the bristles array
        bristles_coords = inifile_tools.get_ini_parameter(fname, "Wing", "bristles_coords", matrix=True)
        
        for j in range( bristles_coords.shape[0]):
            # append start- and end point to pointcloud array
            x.append(bristles_coords[j,0])
            y.append(bristles_coords[j,1])            
            x.append(bristles_coords[j,2])
            y.append(bristles_coords[j,3])    
            
    x, y = np.asarray(x), np.asarray(y)
    
    # construct pointcloud
    xw_w_pointcloud = np.zeros( (x.shape[0],3))
    xw_w_pointcloud[:,0] = x
    xw_w_pointcloud[:,1] = y
    
    return xw_w_pointcloud



def wing_shape_from_SVG( svg_file, fname_out, contour_color, axis_color, bristled=False, 
                        R_bristle=0.01, bristle_color='#000080', debug_plot=True,
                        additional_markers=False, marker_line_color='#008080', interactively_select_centre=False):
    """
    Create WABBIT compatible wing shape file from an SVG file. 
    
    The code identifies different wing elements by their stroke color (set eg in inkscape), so use unique
    coloring for the individual parts.
    
    Give colors as string:
        #0000ff etc (including #)

    Parameters
    ----------
    svg_file : string
        Input file name. The file can contain many things, and only the ones colored accordingly (stroke-color) are used.
        Any additional items, such as text or images, are ignored.
    fname_out : string
        Output file name (INI file)
    contour_color : string, optional
        Stroke color of wing contour.
    axis_color : string, optional
        Stroke color of rotation axis
    bristled : bool, optional
        Look for wing bristles or dont. The default is False.
    bristle_color : string, optional
        Color of bristle elements (which are straight lines)
    
    Returns
    -------
    None. Output saved to INI file.

    """
    import matplotlib.pyplot as plt
    from svgpathtools import svg2paths
    
    paths, attributes = svg2paths( svg_file )
    
    if bristled:
        bristles = []
        
    if debug_plot:
        plt.figure()
        
    print("wing_shape_from_SVG: file=%s, npaths=%i" % (svg_file, len(paths)))
    
    # find the bristles, tip and root point by searching for their colors
    # note functional parts need to have unique contour color in inkscape
    # !!!!!!they must not be grouped!!!!!!!!!!!
    for k, path in enumerate(paths):
        if 'style' in attributes[k]:
            c = attributes[k]['style']
        else:
            c = ""
        
        if "stroke:"+bristle_color in c and bristled:
            # this is a bristle
            bristles.append(path)
        if "stroke:"+contour_color in c:
            contour = path
        if "stroke:"+axis_color in c:
            axis = path
        if additional_markers:
            if  "stroke:"+marker_line_color in c:
                marker_line = path

    
    if not len(contour._segments) > 1:
        raise ValueError("This may be the wrong item for the contour!")
        
    if len(contour._segments) < 50:
        # because we need quite a lot of points for our "linear" approach to work
        print('Warning\n\nYour wing contour has only few points - consider creating more points in inskape (nodes tool->select all->plus button in tool bar')
        
    # extract contour points
    xb, yb = [], []
    # first drawing
    for line in contour._segments:
        # complex
        Z1, Z2 = line.start, line.end

        # warning: plot reveals there is some transformation!!
        # y axis revered
        
        x1, y1 = np.real(Z1), -np.imag(Z1) #-
        x2, y2 = np.real(Z2), -np.imag(Z2)
        
        xb.append(x1)
        yb.append(y1)
        
        if debug_plot:
            plt.plot( [x1,x2], [y1,y2], 'k-'  )
    
    print('wing contour #points=%i' % (len(xb)))
    xb, yb = np.asarray(xb), np.asarray(yb)
    
    # extract additional markers if desired
    if additional_markers:
        x_markers, y_markers = [], []
        for line in marker_line._segments:
            Z1, Z2 = line.start, line.end            
            x1, y1 = np.real(Z1), -np.imag(Z1)
            x2, y2 = np.real(Z2), -np.imag(Z2)
            x_markers.append(x1)
            y_markers.append(y1)
            
        # the last one:
        x_markers.append(x2)
        y_markers.append(y2)
            
        if debug_plot:
            plt.plot( x_markers, y_markers, 'c*', label='additional marker'  )
        
    # extract bristles
    if bristled:        
        all_bristles = np.zeros( [len(bristles), 5])    
        for i, bristle in enumerate(bristles):
            # complex
            Z1, Z2 = bristle._segments[0].start, bristle._segments[0].end
            
            # warning: plot reveals there is some transformation!!
            # y axis revered
            x1, y1 = np.real(Z1), -np.imag(Z1)
            x2, y2 = np.real(Z2), -np.imag(Z2)
            
            all_bristles[i,:] = [x1,y1,x2,y2,R_bristle]
            
            if debug_plot:
                plt.plot( [x1,x2], [y1,y2], 'k-'  )    
        
    # root
    x1, y1 = np.real(axis._start), -np.imag(axis._start) # attention on sign
    # tip
    x2, y2 = np.real(axis._end), -np.imag(axis._end) # attention on sign
    # centre point for polar coordinates
    # xc, yc = 98, -128
    xc, yc = np.sum(xb)/xb.shape[0], np.sum(yb)/yb.shape[0]
    
    if interactively_select_centre:
        tmp = plt.ginput()
        tmp = tmp[0]
        print(tmp)
        xc, yc = tmp[0], tmp[1]
    
    if debug_plot:
        plt.plot(x1, y1, 'o', label='root')
        plt.plot(x2, y2, 'o', label='tip')
        plt.plot(xc, yc, 'o', label='centre point')
        plt.legend()
        plt.axis('equal')
        plt.title('svg input')

    #%% scaling, shifting and rotation

    e_span  = np.asarray([x2-x1, y2-y1])
    e_chord = np.asarray([(y2-y1), -(x2-x1)])
    
    e_span /= np.linalg.norm(e_span)
    e_chord /= np.linalg.norm(e_chord)
    
    # shift all to origin, scale by rot-axis length
    R = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )

    # normalization
    x1, y1 = x1/R, y1/R
    x2, y2 = x2/R, y2/R    
    xc, yc = xc/R, yc/R    
    xb, yb = xb/R, yb/R
    if additional_markers:
        x_markers, y_markers = x_markers/R, y_markers/R
    
    if bristled:
        # do not scale last entry by R as is supposed to be scaled already
        all_bristles[:,0:-1] /= R
        # translation
        all_bristles[:,0] -= x1
        all_bristles[:,2] -= x1
        all_bristles[:,1] -= y1
        all_bristles[:,3] -= y1
        
        # annoying but necessary (classical mistake)
        all_bristles_new = all_bristles.copy()
        
        # projection on unit vectors
        all_bristles_new[:,0] = all_bristles[:,0]*e_chord[0] + all_bristles[:,1]*e_chord[1]
        all_bristles_new[:,1] = all_bristles[:,0]* e_span[0] + all_bristles[:,1] *e_span[1]
        
        all_bristles_new[:,2] = all_bristles[:,2]*e_chord[0] + all_bristles[:,3]*e_chord[1]
        all_bristles_new[:,3] = all_bristles[:,2]* e_span[0] + all_bristles[:,3] *e_span[1]
        
        all_bristles = all_bristles_new

    # translation
    xb, yb = xb-x1, yb-y1
    x2, y2 = x2-x1, y2-y1
    xc, yc = xc-x1, yc-y1
    if additional_markers:
        x_markers, y_markers = x_markers-x1, y_markers -y1
    x1, y1 = 0.0, 0.0
    
    # projection on unit vectors
    xbn, ybn = xb*e_chord[0] + yb*e_chord[1], xb*e_span[0] + yb*e_span[1]
    x2n, y2n = x2*e_chord[0] + y2*e_chord[1], x2*e_span[0] + y2*e_span[1]
    xcn, ycn = xc*e_chord[0] + yc*e_chord[1], xc*e_span[0] + yc*e_span[1]
    if additional_markers:
        xmn, ymn = x_markers*e_chord[0] + y_markers*e_chord[1], x_markers*e_span[0] + y_markers*e_span[1]
        x_markers, y_markers = xmn, ymn
    # annoying but necessary (classical mistake)
    xb, yb, x2, y2, xc, yc = xbn, ybn, x2n, y2n, xcn, ycn
        
    
    if debug_plot:
        plt.figure()
        plt.plot(x1, y1, 'or', mfc='none')
        plt.plot(x2, y2, 'or', mfc='none')
        plt.plot(xc, yc, 'xb')
        plt.plot( [x1,x2], [y1,y2], 'r')
        plt.plot(xb, yb, 'b-')
        
        if bristled:
            for i in range(all_bristles.shape[0]):
                plt.plot([all_bristles[i,0], all_bristles[i,2]], [all_bristles[i,1], all_bristles[i,3]], 'k-')
        if additional_markers:
            plt.plot( x_markers, y_markers, 'c*', label='additional marker'  )
            for i in range(x_markers.shape[0]):
                plt.text(x_markers[i], y_markers[i], "%i" % (i+1))
        plt.axis('equal')
        plt.title('Wing model, scaled, translated and rotated (as used in wabbit thus)')
        plt.show()
        
    if additional_markers:
        print('additional markers in wing system (scaled!)')
        for i in range(x_markers.shape[0]):
            print( "%f,%f" % (x_markers[i], y_markers[i]) )
        
    #%% contour in polar descripion
    # compute radius
    r = np.sqrt( (xb-xc)**2 + (yb-yc)**2 )
    # compute angle
    theta = np.angle( (xb-xc) + 1j*(yb-yc))
    # sort and sample equidistanly (for FFT)
    theta, r = zip(*sorted(zip(theta, r)))
    theta, r = np.asarray(theta), np.asarray(r)
    # to test if the linear interpolation does work and not produce artifacts
    theta2 = np.linspace(-np.pi, np.pi, num=2000, endpoint=False)
    r2 = np.interp(theta2, theta, r)
    
    if debug_plot:
        plt.figure()
        ax1 = plt.gca()

        ax1.plot( xc+np.cos(theta)*r, yc+np.sin(theta)*r, 'b-', label='r(theta)')
        ax1.plot( xc+np.cos(theta2)*r2, yc+np.sin(theta2)*r2, 'b--', label='Interpolated data points')

        ax1.axis('equal')
        ax1.plot( x1,y1, 'b*')
        ax1.legend()
        plt.title('Check if centre point and angle evaluation worked.\n Note for linear description the resolution needs to be higher')
        plt.show()
        
        
    #%% final step: save the INI file.
            
    fid = open(fname_out, 'w')
    fid.write("[Wing]\n")
    fid.write("type=linear;\n\n")
    
    fid.write("; if type==linear, we give theta and R(theta). both are vectors of the same length. Use the matrix notation (/ /)\n")
    fid.write("; The centre-point of the wing is still assumed x0w, y0w just in the Fourier case.\n")
    
    # A word on theta.
    # Python does imply [-pi:+pi] but WABBIT uses 0:2*pi
    # Therefore, we add pi here.
    fid.write("theta_i=(/")
    for a in theta[:-1]:
        fid.write('%e, ' % (a+np.pi))
    fid.write('%e/)\n' % (theta[-1]+np.pi))
    
    fid.write("R_i=(/")
    for a in r[:-1]:
        fid.write('%e, ' % (a))
    fid.write('%e/)\n' % (r[-1]))
    
    fid.write("x0w=%e;\n" % (xc))
    fid.write("y0w=%e;\n" % (yc))
        
    if bristled:
        fid.write("bristles=yes;\n")  
        
        fid.write("; if bristles, give a four column matrix (x1,y1,x2,y2,R)\n")
        fid.write("; note last entry is R and not D. \n")
    
        fid.write("bristles_coords=(/%e %e %e %e %e\n" % (all_bristles[0,0],all_bristles[0,1],all_bristles[0,2],all_bristles[0,3],all_bristles[0,4]) )
        for k in np.arange(1, all_bristles.shape[0] ):
            fid.write("%e %e %e %e %e\n" % (all_bristles[k,0],all_bristles[k,1],all_bristles[k,2],all_bristles[k,3],all_bristles[k,4]) )
        k = -1
        fid.write("%e %e %e %e %e/)\n" % (all_bristles[k,0],all_bristles[k,1],all_bristles[k,2],all_bristles[k,3],all_bristles[k,4]) )
    else:
        fid.write("bristles=no;\n")  
        
    fid.close()
    

def collision_test( time, wing_pointcloud_L_w, alpha_L, theta_L, phi_L, x_hinge_L, 
                          wing_pointcloud_R_w, alpha_R, theta_R, phi_R, x_hinge_R,
                          wing_pointcloud_L2_w=None, alpha_L2=None, theta_L2=None, phi_L2=None, x_hinge_L2=None, 
                          wing_pointcloud_R2_w=None, alpha_R2=None, theta_R2=None, phi_R2=None, x_hinge_R2=None,
                          plot_animation=False, hold_on_collision=True, verbose=True ):
    """
    Check if wings collide. Makes sense only if you have two wings. Body attitude and 
    stroke plane angle do not matter for this routine. Loops over time instances for the check.
    
    We check if either wing crosses the bodys symmetry plane (only then collisons are 
    possible), and if so we compute the exact distance between them from the point cloud. 
    This speeds up the process.

    Parameters
    ----------
    time : array [nt]
        time instants to be checked for collision. linspace(0,1,endpoint=True, num=100) is a good choice
    wing_pointcloud_L_w : array [N,3]
        Points on the left wing, can be generated using get_wing_pointcloud_from_inifile or differently (in the case
        of E.mundus for example fore and hind wing constitute a unit but are stored in two files, so they are externally
        concatenated). Use get_wing_pointcloud_from_inifile for this.
    alpha_L : array [nt]
        wing angle
    theta_L : array [nt]
        wing angle
    phi_L : array [nt]
        wing angle
    wing_pointcloud_R_w :  array [N,3]
        See wing_pointcloud_L_w
    alpha_R : array [nt]
        wing angle
    theta_R : array [nt]
        wing angle
    phi_R : array [nt]
        wing angle

    Returns
    -------
    none (screen log)

    """
    import matplotlib.pyplot as plt
    
    # the collisions do not depend on the body- or stroke plane axis. 
    # useful only for visualization
    eta = deg2rad(180)

    def closest_distance_pointclouds( p1, p2 ):
        # implementation with meshgrid allows at least vectorization (two nested
        # loops are geologically slow, even for just 5k points)
        x1, x2 = np.meshgrid(p1[:,0], p2[:,0])
        y1, y2 = np.meshgrid(p1[:,1], p2[:,1])
        z1, z2 = np.meshgrid(p1[:,2], p2[:,2])
            
        d = np.power( np.power(x1-x2, 2) + np.power(y1-y2, 2) + np.power(z1-z2, 2), 0.5 )
        dist = np.min(d)
                           
        return dist    

    
    if plot_animation:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')

    collision = np.zeros(time.shape[0])
    for it in range(time.shape[0]):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # first pair of wings (forewings or in diptera the only wings)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ML = get_M_b2w(alpha_L[it], theta_L[it], phi_L[it], eta, 'left')
        wing_pointcloud_L_g = np.transpose( ML.T @ wing_pointcloud_L_w.T) 
        
        MR = get_M_b2w(alpha_R[it], theta_R[it], phi_R[it], eta, 'right')
        wing_pointcloud_R_g = np.transpose( MR.T @ wing_pointcloud_R_w.T) 
        
        for dim in range(3):
            wing_pointcloud_L_g[:,dim] += x_hinge_L[dim]
            wing_pointcloud_R_g[:,dim] += x_hinge_R[dim]
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # second wing pair
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if wing_pointcloud_L2_w is not None:
            ML2 = get_M_b2w(alpha_L2[it], theta_L2[it], phi_L2[it], eta, 'left')
            wing_pointcloud_L2_g = np.transpose( ML2.T @ wing_pointcloud_L2_w.T) 
            
            for dim in range(3):
                wing_pointcloud_L2_g[:,dim] += x_hinge_L2[dim]
                
            # simply append the second wing pair to the pointcloud of the first..    
            wing_pointcloud_L_g = np.vstack( (wing_pointcloud_L_g, wing_pointcloud_L2_g) )
            
        if wing_pointcloud_R2_w is not None:
            MR2 = get_M_b2w(alpha_R2[it], theta_R2[it], phi_R2[it], eta, 'right')
            wing_pointcloud_R2_g = np.transpose( MR2.T @ wing_pointcloud_R2_w.T) 
            
            for dim in range(3):
                wing_pointcloud_R2_g[:,dim] += x_hinge_R2[dim]
            
            # simply append the second wing pair to the pointcloud of the first..
            wing_pointcloud_R_g = np.vstack( (wing_pointcloud_R_g, wing_pointcloud_R2_g) )
            
        
            
        if plot_animation:
            ax1.cla()
            ax1.set_xlim([-2,2]) 
            ax1.set_ylim([-2,2])
            ax1.set_zlim([-2,2])    
            ax1.scatter(wing_pointcloud_L_g[:,0], wing_pointcloud_L_g[:,1], wing_pointcloud_L_g[:,2], 'r.')
            ax1.scatter(wing_pointcloud_R_g[:,0], wing_pointcloud_R_g[:,1], wing_pointcloud_R_g[:,2], 'b.')
            ax1.set_title('animation')
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            
        # print("%f %f" % (min(wing_pointcloud_L_g[:,1]), min(wing_pointcloud_R_g[:,1])) )
        
        # if either wing crosses the mid-plane, then a collision is possible.
        # only in those case we compute the expensive exact distance between both wings
        # This is only required if the wing beat is not symmetric. In the symmetric
        # case, the 1st condition is sufficient to detect collisions.
        if min(wing_pointcloud_L_g[:,1]) >= 0.01 or min(wing_pointcloud_R_g[:,1]) <= 0.01:            
            dist = closest_distance_pointclouds( wing_pointcloud_R_g, wing_pointcloud_L_g )
            
            if dist < 0.02:
                if verbose:
                    print('COLLISION (very likely!) t=%2.2f dist=%e' % (time[it], dist) )    
                collision[it] = 1
                
                if hold_on_collision:                
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.cla()
                    ax.set_xlim([-2,2]) 
                    ax.set_ylim([-2,2])
                    ax.set_zlim([-2,2])    
                    ax.scatter(wing_pointcloud_L_g[:,0], wing_pointcloud_L_g[:,1], wing_pointcloud_L_g[:,2], 'r.')
                    ax.scatter(wing_pointcloud_R_g[:,0], wing_pointcloud_R_g[:,1], wing_pointcloud_R_g[:,2], 'b.')
                    ax.set_title('!! Collision detected !!')
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                    break
        if verbose: 
            print('no collision at t=%2.2f detected' % (time[it]) )   
    return collision
        
        
def lollipop_diagram( run_directory, wing='right', ax=None, savePDF=True, savePNG=True, N_lollipops=40):
    """
    A wrapper for visualize_wingpath_chord() that extracts all required parameters
    from a WABBIT/FLUSI type Parameter file (PARAMS.ini), including body angles etc.
    
    Give a simulation folder (run_directory) that contains the main PARAMS.ini file as well 
    as the individual INI files for wing shape and wing kinematics.
    
    This script figures out the body orientation, the stroke plane angle, hinge points etc
    and calls visualize_wingpath_chord() to draw the Lollipop diagram.
    
    Work in progress (09/2025, TE)
    """
    import inifile_tools
    import wabbit_tools
    
    # find the INI file of the run
    PARAMS = wabbit_tools.find_WABBIT_main_inifile(run_directory)
    
    # extract the kienmatics file name from the main ini file
    fname_kine = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'FlappingMotion_'+wing, dtype=str)
    if fname_kine == 'from_file':
        fname_kine = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'infile', dtype=str)
        
    elif 'from_file::' in fname_kine:
        fname_kine = fname_kine.replace('from_file::','')
        
    else:
        raise ValueError("Unfortunately, the Lollipop diagram cannot be created for this run because it uses pre-defined kinematics that are not read from a separate INI file.")

    # required...
    fname_kine = run_directory + '/' + fname_kine
        
    # get stroke plane angle
    eta = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'eta0', dtype=float)
    
    # get the wing hinge (shoulder)
    if wing == 'left':
        x_pivot_b = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'x_pivot_l', dtype=float, vector=True)
    elif wing == 'right':
        x_pivot_b = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'x_pivot_r', dtype=float, vector=True)
    elif wing == 'left2':
        x_pivot_b = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'x_pivot_l2', dtype=float, vector=True)
    elif wing == 'right2':
        x_pivot_b = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'x_pivot_r2', dtype=float, vector=True)
            
    
    
    motion = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'BodyMotion', dtype=str)
        
    if motion == "tethered":
        yawpitchroll_0 = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'yawpitchroll_0', dtype=float, vector=True)
    elif motion == "yawpitchroll_param":
        # used for the EM simulations, here the pitch angle oscillates sinusoidally. we just pass the mean pitch angle
        # to the lollipop diagram.
        # This is the zero-mode of the body angles:
        yawpitchroll_0 = inifile_tools.get_ini_parameter(PARAMS, 'Insects', 'yawpitchroll_0', dtype=float, vector=True)
    else:
        raise ValueError("Unfortunately, the Lollipop diagram cannot be created for this run because it is not tethered flight.")
        
    u_infty = inifile_tools.get_ini_parameter(PARAMS, 'ACM-new', 'u_mean_set', dtype=float, vector=True)
    
    
    visualize_wingpath_chord(fname_kine, psi=yawpitchroll_0[2], gamma=yawpitchroll_0[0], 
                             beta=yawpitchroll_0[1], eta_stroke=eta, ax=ax, 
                             meanflow=u_infty, savePDF=savePDF, savePNG=savePNG,
                             x_pivot_b=x_pivot_b,
                             time = np.linspace(0.0, 1.0, num=N_lollipops, endpoint=False))
    
