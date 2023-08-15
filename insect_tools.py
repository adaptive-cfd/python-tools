#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:11:47 2017

@author: engels
"""


import numpy as np
import numpy.ma as ma
import glob

# I cannot learn this by heart....
def change_figure_dpi(dpi):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = dpi
    

def get_next_color(ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    return next(ax._get_lines.prop_cycler)['color']

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

# construct a column-vector for math operatrions. I hate python.
def vct(x):
    v = np.matrix(x)
    v = v[np.newaxis]
    v = v.reshape(len(x),1)
    return v

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
    w, h = fig.get_size_inches()
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


def reset_colorcycle( ax=None ):
    import matplotlib.pyplot as plt
    # reset color cycle
    if ax is None:
        plt.gca().set_prop_cycle(None)
    else:
        ax.set_prop_cycle(None)


def load_t_file( fname, interp=False, time_out=None, return_header=False,
                verbose=True, time_mask_before=None, T0=None, keep_duplicates=False, remove_outliers=False ):
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

                if len(tmp) == ncols:
                    dat.append( tmp )
                else:
                    dat.append( tmp[0:ncols] )

    # convert list of lists into an numpy array
    data_raw = np.array( dat, dtype=float )

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
            if t1 <= t0:
                # we have found a jump (which is often quite big so we must not worry about <=)
                # now, figure out the index where we duplicate time stamps began
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
            # actual data
            line = data[it,:]
            # linear interpolation using neighbor values
            line_interp = (data[it-1,:] + data[it+1,:]) *0.5

            err_abs = np.abs(line - line_interp)
            err_rel = np.abs(line - line_interp) / np.abs(line_interp)
            
            err = err_rel
            err[ err_abs <= 1.0e-7 ] = err_abs[ err_abs <= 1.0e-7 ]            
            
            if (np.max(err)>0.25):
                it_unique[it] = False
            else:
                it_unique[it] = True
                
        # sometimes the last point is a problem:
        err_abs = np.abs(data[-2,:]-data[-1,:])
        err_rel = err_abs / np.abs(data[-2,:])
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

    # hide first times, if desired
    if time_mask_before is not None:
        data = np.ma.array( data, mask=np.repeat( data[:,0]<time_mask_before, data.shape[1]))

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

    navgs = np.int( np.round((t2-t1)/tstroke) )

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
            f.write(name)
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
    
    If you request N=20 modes, we return a0 and ai[0:19] so a total of 21 numbers.
    
    Input:
    ------
    
        y: vector, float
           The actual data to be analyzed (e.g., angle time series). Note: it is assumed 
           to be sampled on an equidistant grid. 
        n: integer
           Number of Fourier modes to use.
       
    
    Output:
    -------
        ai, bi: numpy arrays containing the real (ai) and imaginary (bi) parts of the n Fourier coefficients.
        
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
    
    y = a0/2.0
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

    # time periodization
    while ( np.max(time) >= 1.0 ):
        time[ time >= 1.0 ] -= 1.0
        
    n = ai.shape[0]
    dt = 1.0 / n
    j1 = np.floor(time/dt) # zero-based indexing in python
    j1 = np.asarray(j1, dtype=int)
    j2 = j1 + 1
    
    # periodization
    j2[ j2 > n-1 ] = 0 # zero-based indexing in python
    
    # normalized time (between two data points)
    t = (time - j1*dt) / dt

    # values of hermite interpolant
    h00 = (1.0+2.0*t)*((1.0-t)**2)
    h10 = t*((1.0-t)**2)
    h01 = (t**2)*(3.0-2.0*t)
    h11 = (t**2)*(t-1.0)

    # function value
    u = h00*ai[j1] + h10*dt*bi[j1] + h01*ai[j2] + h11*dt*bi[j2]
    
    return u


def read_kinematics_file( fname ):
    import configparser
    import os
    
    if not os.path.isfile(fname):
        raise ValueError("File "+fname+" not found!")

    config = configparser.ConfigParser( inline_comment_prefixes=(';'), allow_no_value=True )
    # read the ini-file
    config.read(fname)

    if config['kinematics']:
        convention = read_param(config,'kinematics','convention')
        series_type = read_param(config,'kinematics','type')

        if convention != "flusi":
            raise ValueError("The kinematics file %s is using a convention not supported yet" % (fname))

        if series_type == "fourier":
            a0_phi   = float(read_param(config,'kinematics','a0_phi'))
            a0_alpha = float(read_param(config,'kinematics','a0_alpha'))
            a0_theta = float(read_param(config,'kinematics','a0_theta'))
        else:
            a0_phi, a0_theta, a0_alpha = 0.0, 0.0, 0.0
            
        ai_alpha = read_param_vct(config,'kinematics','ai_alpha')
        bi_alpha = read_param_vct(config,'kinematics','bi_alpha')

        ai_theta = read_param_vct(config,'kinematics','ai_theta')
        bi_theta = read_param_vct(config,'kinematics','bi_theta')

        ai_phi   = read_param_vct(config,'kinematics','ai_phi')
        bi_phi   = read_param_vct(config,'kinematics','bi_phi')


        return a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta, series_type
    else:
        print('This seems to be an invalid ini file as it does not contain the kinematics section')



def visualize_kinematics_file(fname):
    """ Read an INI file with wingbeat kinematics and plot the 3 angles over the period. Output written to a PDF and PNG file.
    """

    import matplotlib.pyplot as plt

    t, phi, alpha, theta = eval_angles_kinematics_file(fname)

    plt.rcParams["text.usetex"] = False

    plt.figure( figsize=(cm2inch(12), cm2inch(7)) )
    plt.subplots_adjust(bottom=0.16, left=0.14)

    plt.plot(t, phi  , label='$\\phi$ (flapping)')
    plt.plot(t, alpha, label='$\\alpha$ (feathering)')
    plt.plot(t, theta, label='$\\theta$ (deviation)')

    plt.legend()
    plt.xlim([0,1])
    plt.xlabel('$t/T$')
        
    # axis y in degree
    from matplotlib.ticker import EngFormatter
    plt.gca().yaxis.set_major_formatter(EngFormatter(unit="°"))

    plt.title('$\\Phi=%2.2f^\\circ$ $\\phi_m=%2.2f^\\circ$ $\\phi_\\mathrm{max}=%2.2f^\\circ$ $\\phi_\\mathrm{min}=%2.2f^\\circ$' % (np.max(phi)-np.min(phi), np.mean(phi), np.max(phi), np.min(phi)))
    
    indicate_strokes()
    
    ax = plt.gca()
    ax.tick_params( which='both', direction='in', top=True, right=True )
    plt.savefig( fname.replace('.ini','.pdf'), format='pdf' )


def csv_kinematics_file(fname):
    """ Read an INI file with wingbeat kinematics and store the 3 angles over the period in a *.csv file
    """
    
    t, phi, alpha, theta = eval_angles_kinematics_file(fname, time=np.linspace(0,1,100, endpoint=False) )
    
    d = np.zeros([t.shape[0], 4])
    d[:,0] = t
    d[:,1] = phi
    d[:,2] = alpha
    d[:,3] = theta
    
    write_csv_file( fname.replace('.ini', '.csv'), d, header=['time', 'phi', 'alpha', 'theta'], sep=';')


def eval_angles_kinematics_file(fname, time=None):
    """
    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    time : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    t : array
        time.
    phi : array
        flapping angle.
    alpha : array
        feathering angle.
    theta : array
        deviation angle.

    """
    # read the kinematics INI file
    a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta, kine_type = read_kinematics_file(fname)
    
    if time is None:
        # time vector for plotting
        t = np.linspace(0.0, 1.0, 1000, endpoint=False)
    else:
        t = time
        
    if kine_type == "fourier":
        alpha = Fserieseval(a0_alpha, ai_alpha, bi_alpha, t)
        phi   = Fserieseval(a0_phi  , ai_phi  , bi_phi  , t)
        theta = Fserieseval(a0_theta, ai_theta, bi_theta, t)
        
    elif kine_type == "hermite":
        alpha = Hserieseval(a0_alpha, ai_alpha, bi_alpha, t)
        phi   = Hserieseval(a0_phi  , ai_phi  , bi_phi  , t)
        theta = Hserieseval(a0_theta, ai_theta, bi_theta, t)
    
        
    
    return t, phi, alpha, theta




def Rx( angle ):
    # rotation matrix around x axis
    Rx = np.ndarray([3,3])
    Rx = [[1.0,0.0,0.0],[0.0,np.cos(angle),np.sin(angle)],[0.0,-np.sin(angle),np.cos(angle)]]
    # note the difference between array and matrix (it is the multiplication)
    Rx = np.matrix( Rx )
    return Rx


def Ry( angle ):
    # rotation matrix around y axis
    Rx = np.ndarray([3,3])
    Rx = [[np.cos(angle),0.0,-np.sin(angle)],[0.0,1.0,0.0],[+np.sin(angle),0.0,np.cos(angle)]]
    # note the difference between array and matrix (it is the multiplication)
    Rx = np.matrix( Rx )
    return Rx


def Rz( angle ):
    # rotation matrix around z axis
    Rx = np.ndarray([3,3])
    Rx = [[ np.cos(angle),+np.sin(angle),0.0],[-np.sin(angle),np.cos(angle),0.0],[0.0,0.0,1.0]]
    # note the difference between array and matrix (it is the multiplication)
    Rx = np.matrix( Rx )
    return Rx


def Rmirror( x0, n):
    # mirror by a plane through origin x0 with given normal n
    # source: https://en.wikipedia.org/wiki/Transformation_matrix#Reflection_2
    Rmirror =  np.zeros([4,4])

    a, b, c = n[0], n[1], n[2]
    d = -(a*x0[0] + b*x0[1] + c*x0[2])

    Rmirror = [ [1-2*a**2,-2*a*b,-2*a*c,-2*a*d], [-2*a*b,1-2*b**2,-2*b*c,-2*b*d], [-2*a*c,-2*b*c,1-2*c**2,-2*c*d],[0,0,0,1] ]
    # note the difference between array and matrix (it is the multiplication)
    Rmirror = np.matrix( Rmirror )

    return(Rmirror)


def visualize_wingpath_chord( fname, psi=0.0, gamma=0.0, beta=0.0, eta_stroke=0.0, equal_axis=True, DrawPath=False,
                             x_pivot_b=[0,0,0], x_body_g=[0,0,0], wing='left', chord_length=0.1,
                             draw_true_chord=False, meanflow=None, reverse_x_axis=False, 
                             time=np.linspace( start=0.0, stop=1.0, endpoint=False, num=40) ):
    """ visualize the wing chord
    
    give all angles in degree
    
    visualize_wingpath_chord( fname, psi=0.0, gamma=0.0, beta=0.0, eta_stroke=0.0, equal_axis=True, DrawPath=False,
                             x_pivot_b=[0,0,0], x_body_g=[0,0,0], wing='left', chord_length=0.1,
                             draw_true_chord=False, meanflow=None ):
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib


    if not os.path.isfile(fname):
        raise ValueError("The file "+fname+" does not exist.")

    # read kinematics data:
    a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta, kine_type = read_kinematics_file( fname )
        
    # length of wing chord to be drawn. note this is not correlated with the actual
    # wing thickness at some position - it is just a marker.
    wing_chord = chord_length

    # wing tip in wing coordinate system
    x_tip_w = vct([0.0, 1.0, 0.0])
    x_le_w  = vct([ 0.5*wing_chord,1.0,0.0])
    x_te_w  = vct([-0.5*wing_chord,1.0,0.0])

    x_pivot_b = vct(x_pivot_b)
    x_body_g  = vct(x_body_g)


    # body transformation matrix
    M_body = Rx(deg2rad(psi))*Ry(deg2rad(beta))*Rz(deg2rad(gamma))


    # rotation matrix from body to stroke coordinate system:
    M_stroke_l = Ry(deg2rad(eta_stroke))
    M_stroke_r = Rx(np.pi)*Ry(deg2rad(eta_stroke))


    plt.figure( figsize=(cm2inch(12), cm2inch(7)) )
    plt.subplots_adjust(bottom=0.16, left=0.14)
    ax = plt.gca() # we need that to draw lines...


    # array of color (note normalization to 1 for query values)
    colors = plt.cm.jet( (np.arange(time.size) / time.size) )
    
           
    if kine_type == "fourier":
        alpha_l = Fserieseval(a0_alpha, ai_alpha, bi_alpha, time)
        phi_l   = Fserieseval(a0_phi  , ai_phi  , bi_phi  , time)
        theta_l = Fserieseval(a0_theta, ai_theta, bi_theta, time)
        
    elif kine_type == "hermite":
        alpha_l = Hserieseval(a0_alpha, ai_alpha, bi_alpha, time)
        phi_l   = Hserieseval(a0_phi  , ai_phi  , bi_phi  , time)
        theta_l = Hserieseval(a0_theta, ai_theta, bi_theta, time)
        

    # step 1: draw the symbols for the wing section for some time steps
    for i in range(time.size):
        # rotation matrix from body to wing coordinate system
        if wing == 'left':
            M_wing = Ry(deg2rad(alpha_l[i]))*Rz(deg2rad(theta_l[i]))*Rx(deg2rad(phi_l[i]))*M_stroke_l

        elif wing == 'right':
            M_wing = Ry(-deg2rad(alpha_l[i]))*Rz(+deg2rad(theta_l[i]))*Rx(-deg2rad(phi_l[i]))*M_stroke_r


        # convert wing points to global coordinate system
        x_tip_g = np.transpose(M_body) * ( np.transpose(M_wing) * x_tip_w + x_pivot_b ) + x_body_g
        x_le_g  = np.transpose(M_body) * ( np.transpose(M_wing) * x_le_w  + x_pivot_b ) + x_body_g
        x_te_g  = np.transpose(M_body) * ( np.transpose(M_wing) * x_te_w  + x_pivot_b ) + x_body_g


        if not draw_true_chord:
            # the wing chord changes in length, as the wing moves and is oriented differently
            # note if the wing is perpendicular, it is invisible
            # so this vector goes from leading to trailing edge:
            e_chord = x_te_g - x_le_g
            e_chord[1] = [0.0]

            # normalize it to have the right length
            e_chord = e_chord / (np.linalg.norm(e_chord))

            # pseudo TE and LE. note this is not true TE and LE as the line length changes otherwise
            x_le_g = x_tip_g - e_chord * wing_chord/2.0
            x_te_g = x_tip_g + e_chord * wing_chord/2.0

        # mark leading edge with a marker
        plt.plot( x_le_g[0], x_le_g[2], marker='o', color=colors[i], markersize=4 )

        # draw wing chord
        plt.plot( [x_te_g[0,0], x_le_g[0,0]], [x_te_g[2,0], x_le_g[2,0]], '-', color=colors[i])


    # step 2: draw the path of the wingtip
    if DrawPath:
        # refined time vector for drawing the wingtip path
        time = np.linspace( start=0.0, stop=1.0, endpoint=False, num=1000)
        xpath = time.copy()
        zpath = time.copy()
        
        alpha_l, phi_l, theta_l = np.zeros(time.shape), np.zeros(time.shape), np.zeros(time.shape)
           
        if kine_type == "fourier":
            alpha_l = Fserieseval(a0_alpha, ai_alpha, bi_alpha, time)
            phi_l   = Fserieseval(a0_phi  , ai_phi  , bi_phi  , time)
            theta_l = Fserieseval(a0_theta, ai_theta, bi_theta, time)
            
        elif kine_type == "hermite":
            alpha_l = Hserieseval(a0_alpha, ai_alpha, bi_alpha, time)
            phi_l   = Hserieseval(a0_phi  , ai_phi  , bi_phi  , time)
            theta_l = Hserieseval(a0_theta, ai_theta, bi_theta, time)


        for i in range(time.size):
            # rotation matrix from body to wing coordinate system
            # rotation matrix from body to wing coordinate system
            if wing == 'left':
                M_wing = Ry(deg2rad(alpha_l[i]))*Rz(deg2rad(theta_l[i]))*Rx(deg2rad(phi_l[i]))*M_stroke_l
            elif wing == 'right':
                M_wing = Ry(-deg2rad(alpha_l[i]))*Rz(+deg2rad(theta_l[i]))*Rx(-deg2rad(phi_l[i]))*M_stroke_r

            # convert wing points to global coordinate system
            x_tip_g = np.transpose(M_body) * ( np.transpose(M_wing) * x_tip_w + x_pivot_b ) + x_body_g

            xpath[i] = (x_tip_g[0])
            zpath[i] = (x_tip_g[2])
        plt.plot( xpath, zpath, linestyle='--', color='k', linewidth=1.0 )


    # Draw stroke plane as a dashed line
    if wing == 'left':
        M_stroke = M_stroke_l
    elif wing == 'right':
        M_stroke = M_stroke_r

    # we draw the line between [0,0,-1] and [0,0,1] in the stroke system
    xs1 = vct([0.0, 0.0, +1.0])
    xs2 = vct([0.0, 0.0, -1.0])
    # bring these points back to the global system
    x1 = np.transpose(M_body) * ( np.transpose(M_stroke)*xs1 + x_pivot_b ) + x_body_g
    x2 = np.transpose(M_body) * ( np.transpose(M_stroke)*xs2 + x_pivot_b ) + x_body_g

    plt.ylim([-1, 1])

    # remember we're in the x-z plane
    l = matplotlib.lines.Line2D( [x1[0],x2[0]], [x1[2],x2[2]], color='k', linewidth=1.0, linestyle='--')
    ax.add_line(l)

    # this is a manually set size, which should be the same as what is produced by visualize kinematics file
    # plt.gcf().set_size_inches([4.71, 2.75] )
    plt.gcf().set_size_inches([4.0, 4.0] )
    plt.gcf().subplots_adjust(hspace=0.0, right=0.88, bottom=0.12, left=0.16, top=0.92)
    
    if equal_axis:
        axis_equal_keepbox( plt.gcf(), plt.gca() )

    # annotate plot
#    plt.rcParams["text.usetex"] = False
#    plt.xlabel('$x^{(g)}$')
#    plt.ylabel('$z^{(g)}$')

    if meanflow is not None:
        x0, y0 = 0.0, 0.0
        plt.arrow( x0, y0, meanflow[0], meanflow[2], width=0.000001, head_width=0.025 )
        plt.text(x0+meanflow[0]*1.4, y0+meanflow[2]*1.4, '$u_\infty$' )

    if reverse_x_axis:
        plt.gca().invert_xaxis()

    
    ax = plt.gca()
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-1.1, 1.1])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel('x/R')
    ax.set_ylabel('y/R')
    
    # modify ticks in matlab-style.
    ax.tick_params( which='both', direction='in', top=True, right=True )
    plt.savefig( fname.replace('.ini','_path.pdf'), format='pdf' )
#    plt.savefig( fname.replace('.ini','_path.png'), format='png', dpi=300 )

def wingtip_path( fname, time=None, wing='left', eta_stroke=0.0, psi=0.0, beta=0.0, gamma=0.0):
   
    if time is None:
        time = np.linspace(0, 1.0, 1000, endpoint=True)
        
    # wing tip in wing coordinate system
    x_tip_w   = vct([0.0, 1.0, 0.0])
    x_pivot_b = vct([0.0, 0.0, 0.0])
    x_body_g  = vct([0.0, 0.0, 0.0])
        
    # body transformation matrix
    M_body = Rx(deg2rad(psi))*Ry(deg2rad(beta))*Rz(deg2rad(gamma))

    # rotation matrix from body to stroke coordinate system:
    M_stroke_l = Ry(deg2rad(eta_stroke))
    M_stroke_r = Rx(np.pi)*Ry(deg2rad(eta_stroke))
    
    # evaluate kinematics
    t, phi_l, alpha_l, theta_l = eval_angles_kinematics_file(fname, time=time)
    
    # allocation
    xb, yb, zb = np.zeros(time.shape), np.zeros(time.shape), np.zeros(time.shape)
       
    for i in range(time.shape[0]):
        # rotation matrix from body to wing coordinate system
        if wing == 'left':
            M_wing = Ry(deg2rad(alpha_l[i]))*Rz(deg2rad(theta_l[i]))*Rx(deg2rad(phi_l[i]))*M_stroke_l
        elif wing == 'right':
            M_wing = Ry(-deg2rad(alpha_l[i]))*Rz(+deg2rad(theta_l[i]))*Rx(-deg2rad(phi_l[i]))*M_stroke_r

        # convert wing points to global coordinate system
        x_tip_g = np.transpose(M_body) * ( np.transpose(M_wing) * x_tip_w + x_pivot_b ) + x_body_g

        xb[i] = (x_tip_g[0])
        yb[i] = (x_tip_g[1])
        zb[i] = (x_tip_g[2])  
    
    return xb, yb, zb

def wingtip_velocity( fname_kinematics, time=None ):
    """ Compute wingtip velocity as a function of time, given a wing kinematics parameter
    file. Note we assume the body at rest (hence relative to body).
    """
    # # read kinematics data:
    # a0_phi, ai_phi, bi_phi, a0_alpha, ai_alpha, bi_alpha, a0_theta, ai_theta, bi_theta, kine_type = read_kinematics_file( fname_kinematics )
    
    if time is None:
        time = np.linspace(0, 1.0, 200, endpoint=True)
    
    # if kine_type == "fourier":
    #     alpha_l = Fserieseval(a0_alpha, ai_alpha, bi_alpha, time)
    #     phi_l   = Fserieseval(a0_phi  , ai_phi  , bi_phi  , time)
    #     theta_l = Fserieseval(a0_theta, ai_theta, bi_theta, time)
        
    # elif kine_type == "hermite":
    #     alpha_l = Hserieseval(a0_alpha, ai_alpha, bi_alpha, time)
    #     phi_l   = Hserieseval(a0_phi  , ai_phi  , bi_phi  , time)
    #     theta_l = Hserieseval(a0_theta, ai_theta, bi_theta, time)
        
    t, phi_l, alpha_l, theta_l = eval_angles_kinematics_file(fname_kinematics, time=time)
    
    # wing tip in wing coordinate system
    x_tip_w = vct([0.0, 1.0, 0.0])
    
    v_tip_b = np.zeros(time.shape)
    
    for i in range(time.size-1):
        # we use simple differentiation (finite differences) to get the velocity
        dt = time[1]-time[0]

        # rotation matrix from body to wing coordinate system
        M_wing_l = Ry(deg2rad(alpha_l[i]))*Rz(deg2rad(theta_l[i]))*Rx(deg2rad(phi_l[i]))
        
        # convert wing points to body coordinate system
        x1_tip_b = np.transpose(M_wing_l) * x_tip_w

        # rotation matrix from body to wing coordinate system
        M_wing_l = Ry(deg2rad(alpha_l[i+1]))*Rz(deg2rad(theta_l[i+1]))*Rx(deg2rad(phi_l[i+1]))

        # convert wing points to body coordinate system
        x2_tip_b = np.transpose(M_wing_l) * x_tip_w

        v_tip_b[i] = np.linalg.norm( (x2_tip_b - x1_tip_b)/dt )

    v_tip_b[-1] = v_tip_b[0]
    return v_tip_b


def interp_matrix( d, time_new ):
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
        d2[:,i] = np.interp( time_new, d[:,0], d[:,i] )#, right=0.0 )

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


def make_white_plot( ax ):
    # for the poster, make a couple of changes: white font, white lines, all transparent.
    legend = ax.legend()
    if not legend is None:
        frame = legend.get_frame()
        frame.set_alpha(0.0)
        # set text color to white for all entries
        for label in legend.get_texts():
            label.set_color('w')


    ax.xaxis.label.set_color('w')
    ax.tick_params(axis='x', colors='w')

    ax.yaxis.label.set_color('w')
    ax.tick_params(axis='y', colors='w')

    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['right'].set_color('w')

    ax.tick_params( which='both', direction='in', top=True, right=True, color='w' )




def hit_analysis():
    import glob

    # Take all analyis files from ./flusi --turbulence-analysis and put all
    # data in one csv file

    showed_header=False

    fid = open('complete_analysis.csv', 'w')

    for file in sorted( glob.glob('analysis_*.txt') ):
        print(file)
        # read entire file to list of lines
        with open (file, "r") as myfile:
            data = myfile.readlines()

        # remove header lines
        del data[0:2+1]
        del data[1]

        # fetch viscosity from remaining header
        header = data[0].split()
        nu = float(header[7])

        if not showed_header:
            showed_header = True
            # write header
            fid.write('name;')
            for line in data[1:]:
                fid.write("%15s; " % (line[20:-1]))
            fid.write('viscosity;\n')

        # read remaining data items
        fid.write("%s; " % (file))
        for line in data[1:]:
            fid.write("%e; " % (float(line[0:20])))

        fid.write("%e;" % (nu) )
        fid.write("\n")

    fid.close()


def plot_a_col( data, col ):
    D = stroke_average_matrix( data, tstroke=0.5 )
    plt.plot( data[:,0], data[:,col] )
    plt.plot( D[:,0], D[:,col], linestyle='None', marker='o', markerfacecolor='none', color=h[-1].get_color())



def forces_g2b( data, kinematics ):
    """ Transform timeseries data (forces.t) to body system defined by kinematics.t
    """

    # they are not necessarily at the same times t -> interpolation
    time = data[:,0]

    # interpolate kinematics to data time vector
    k = interp_matrix( kinematics, time )
    psi = k[:,4]
    beta  = k[:,5]
    gamma  = k[:,6]

    data_new = data.copy()

    for it in range(data.shape[0]):
        M_body = Rx(psi[it]) * Ry(beta[it]) * Rz(gamma[it])
        # forces
        Fg = vct( [data[it,1], data[it,2], data[it,3]] )
        Fb = M_body*Fg
        data_new[it,1:3+1] = Fb.transpose()

        # moments, if present
        if data.shape[1] >= 10:
            Mg = vct( [data[it,7],data[it,8],data[it,9]] )
            Mb = M_body*Mg
            data_new[it,7:9+1] = Mb.transpose()

        # unsteady corrections (rarely used)
        if data.shape[1] >= 6:
            Fg = vct( [data[it,4],data[it,5],data[it,6]] )
            Fb = M_body*Fg
            data_new[it,4:6+1] = Fb.transpose()

        if data.shape[1] >= 12:
            Mg = vct( [data[it,10],data[it,11],data[it,12]] )
            Mb = M_body*Mg
            data_new[it,10:12+1] = Mb.transpose()

    return(data_new)


def forces_g2wr( data, kinematics ):
    """ Transform timeseries data (forces.t) to right wing system defined by kinematics.t
    """
    import numpy as np

    # they are not necessarily at the same times t -> interpolation
    time = data[:,0]

    # interpolate kinematics to data time vector
    k = interp_matrix( kinematics, time )

    psi     = k[:,4]
    beta    = k[:,5]
    gamma   = k[:,6]
    eta_stroke = k[:,7]
    alpha_r    = k[:,11]
    phi_r      = k[:,12]
    theta_r    = k[:,13]

    data_new = data.copy()

    for it in range(data.shape[0]):

        M_body = Rx(psi[it])*Ry(beta[it])*Rz(gamma[it])
        # rotation matrix from body to stroke coordinate system:
        M_stroke_r = Rx(np.pi)*Ry(eta_stroke[it])

        # rotation matrix from body to wing coordinate system
        M_wing_r = Ry(-alpha_r[it])*Rz(+theta_r[it])*Rx(-phi_r[it])*M_stroke_r


        # usual forces
        Fg = vct( [data[it,1],data[it,2],data[it,3]] )
#        Mg = vct( [data[it,7],data[it,8],data[it,9]] )

        Fb = M_wing_r*M_body*Fg
#        Mb = M_wing_r*M_body*Mg

        data_new[it,1:3+1] = Fb.transpose()
#        data_new[it,7:9+1] = Mb.transpose()
#
#        # unsteady corrections (rarely used)
#        Fg = vct( [data[it,4],data[it,5],data[it,6]] )
#        Mg = vct( [data[it,10],data[it,11],data[it,12]] )
#
#        Fb = M_wing_r*M_body*Fg
#        Mb = M_wing_r*M_body*Mg
#
#        data_new[it,4:6+1] = Fb.transpose()
#        data_new[it,10:12+1] = Mb.transpose()

    return(data_new)



def read_flusi_HDF5( fname, dtype=np.float64):
    """  Read HDF5 file generated by FLUSI.
    Returns: time, box, origin, data
    """
    import flusi_tools
    time, box, origin, data = flusi_tools.read_flusi_HDF5( fname, dtype=dtype )
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
        data = np.zeros([nx,ny,nz], dtype=float32)

        # it is useful to now use the entire array, so python can crash here if
        # out of memory, and not after waiting a long time...
        data = data + 1.0


        for i in range(nz):
            sheet = load_image( files[i] )
            data[:,:,i] = sheet.copy()

        write_flusi_HDF5( outfile, 0.0, [float(nx)*dx,float(ny)*dx,float(nz)*dx], data, viscosity=0.0, origin=origin )


def integrated_L2_difference_signal( data1, data2, qty ):
    """ compute the integrated L2 difference between two signals.
    Data is assumed in matrix form d[:,0] is time. Normalization is done with data1 """

    # interpolate signal 2 to time vector of signal 1
    sig2 = np.interp( data1[:,0], data2[:,0], data2[:,qty])
    sig1 = data1[:,qty]

#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot( sig1 , label='ref')
#    plt.plot( sig2 , label='this')

    err = np.sqrt(np.trapz( (sig2-sig1)**2, x=data1[:,0] ) ) / np.sqrt(np.trapz( sig1**2, x=data1[:,0] ))
#    err = np.linalg.norm( sig2-sig1 ) / np.linalg.norm(sig1)

    return err


def forces_L2_error( data, data_ref, idx_data, idx_ref, normalized=True):
    """ Similar to suzuki_error, but more general. compute integrated L2 difference
    wrt a reference data set """
    import numpy as np

    # interpolate reference data (assumed to be high-precision) to
    # the time vector of the actual data
    data_ref = interp_matrix(data_ref, data[:,0])

    err = []
    for IDX_DATA, IDX_REF in zip(idx_data, idx_ref):
        if normalized:
            err.append( np.trapz( abs(data[:,IDX_DATA]-data_ref[:,IDX_REF]), x=data[:,0] ) /
                        np.trapz( abs(data_ref[:,IDX_REF]), x=data[:,0]) )
        else:
            err.append( np.trapz( abs(data[:,IDX_DATA]), x=data[:,0] ) )

    err = np.asarray(err)

#    return np.linalg.norm(err)
    return np.mean(err)

def suzuki_error( filename, component=None, reference='suzuki', T0=None ):
    """compute the error for suzukis test case"""


    import numpy as np
    import matplotlib.pyplot as plt

    # read reference data, digitized from suzukis paper
    reference_file = '/home/engels/Documents/Research/Insects/3D/projects/suzuki_validation/Digitize/LBM_lift.csv'
    dref_lift = np.loadtxt( reference_file, delimiter=',', skiprows=1 )

    reference_file = '/home/engels/Documents/Research/Insects/3D/projects/suzuki_validation/Digitize/LBM_drag.csv'
    dref_drag = np.loadtxt( reference_file, delimiter=',', skiprows=1 )

#    reference_file = '/home/engels/Documents/Research/Insects/3D/projects/suzuki_validation/Digitize/mediana_lift.csv'
#    dref_lift2 = np.loadtxt( reference_file, delimiter=',', skiprows=1 )
#
#    reference_file = '/home/engels/Documents/Research/Insects/3D/projects/suzuki_validation/Digitize/mediana_drag.csv'
#    dref_drag2 = np.loadtxt( reference_file, delimiter=',', skiprows=1 )

    # read actual data
    data = insect_tools.load_t_file( filename, T0=T0 )

    # Suzuki et al. eq. in appendix B.5.2
    L = 0.833
    c = 0.4167
    rho = 1
    utip = 2*np.pi*(80*np.pi/180)*(0.1667+0.833)/1.0
    fcoef = 0.5*rho*(utip**2)*(L*c)
    # apply normalization
    data[:,1:3+1] /= fcoef

    # after some experimentation, I can tell what suzuki calls 'drag': it is indeed the force
    # in x-direction, but the standard suzuki test includes gamma=45° hence some modification is required
    data[:,1] = (data[:,1] + data[:,2]) / np.sqrt( 2.0 )


    if reference == 'suzuki':
        data_flusi = load_t_file('/home/engels/Documents/Research/Insects/3D/projects/suzuki_validation/level3_small/forces.t')
        data_flusi[:,1:3+1] /= fcoef
        data_flusi[:,1] = (data_flusi[:,1] + data_flusi[:,2]) / np.sqrt( 2.0 )
        data_flusi = interp_matrix(data_flusi, dref_drag[:,0])

        # interpolate actual data on ref data points
        data1 = insect_tools.interp_matrix( data, dref_lift[:,0] )
        data2 = insect_tools.interp_matrix( data, dref_drag[:,0] )


        plt.figure()
    #    plt.plot(data[:,0], data[:,3])
        plt.plot(data1[:,0], data1[:,3],  label='this data')
        plt.plot(dref_lift[:,0], dref_lift[:,1], 'k', label='ref scan data')
    #    plt.plot(dref_lift2[:,0]+0.5, dref_lift2[:,1], 'c-',label='medina thesis')


        plt.plot(dref_drag[:,0], dref_drag[:,1],'k--', label='ref (scan)')
        plt.plot(data2[:,0], data2[:,1], '-.', label='this data')
        plt.plot(data_flusi[:,0], data_flusi[:,1],'-.',label='flusi-reference-data')
    #    plt.plot(dref_drag2[:,0], dref_drag2[:,1], 'c-', label='medina drag')

        plt.title(filename)
        plt.legend()


        err1 = np.trapz( abs(data1[:,3]-dref_lift[:,1]), x=data1[:,0] ) / np.trapz( abs(dref_lift[:,1]), x=data1[:,0] )
        err2 = np.trapz( abs(data2[:,1]-dref_drag[:,1]), x=data2[:,0] ) / np.trapz( abs(dref_drag[:,1]), x=data2[:,0] )

        return np.sqrt(err1**2 + err2**2)

    else:
        data_flusi = load_t_file( reference, T0=T0 )

        data_flusi[:,1:3+1] /= fcoef
        data_flusi[:,1] = (data_flusi[:,1] + data_flusi[:,2]) / np.sqrt( 2.0 )
        data_flusi = interp_matrix(data_flusi, data[:,0])

#        plt.figure()
#
#        plt.plot( data[:,0], data[:,1], label='this data (x)')
#        plt.plot( data[:,0], data[:,2], label='this data (y)')
#        plt.plot( data[:,0], data[:,3], label='this data (z)')
#
#        # reset color cycle
#        plt.gca().set_prop_cycle(None)
#
#        plt.plot( data_flusi[:,0], data_flusi[:,1], '--', label='reference (flusi,1024) (x)')
#        plt.plot( data_flusi[:,0], data_flusi[:,2], '--', label='reference (flusi,1024) (y)')
#        plt.plot( data_flusi[:,0], data_flusi[:,3], '--', label='reference (flusi,1024) (z)')
#
#        plt.grid()
#        plt.title(filename)
#        plt.legend()
#        plt.xlim((3.0, 4.0))
#        plt.ylim((-0.75, 0.75))


        err1 = np.trapz( abs(data[:,1]-data_flusi[:,1]), x=data_flusi[:,0] ) / np.trapz( abs(data_flusi[:,1]), x=data_flusi[:,0] )
        err2 = np.trapz( abs(data[:,2]-data_flusi[:,2]), x=data_flusi[:,0] ) / np.trapz( abs(data_flusi[:,2]), x=data_flusi[:,0] )
        err3 = np.trapz( abs(data[:,3]-data_flusi[:,3]), x=data_flusi[:,0] ) / np.trapz( abs(data_flusi[:,3]), x=data_flusi[:,0] )

        # error is magnitude of all 3 components
        return np.sqrt(err1**2 + err2**2 + err3**2)


def write_kinematics_ini_file(fname, alpha, phi, theta, nfft, header=['header goes here']):
    """
     given the angles alpha, phi and theta and a vector of numbers, perform
     Fourier series approximation and save result to INI file.

     Input:
         - fname: filename
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
    f.write('units=degree;\n')
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
    
  
    
def visualize_wing_shape_file(fname, fig=None):
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
    import wabbit_tools
    import matplotlib.pyplot as plt
    
    
    plt.rcParams["text.usetex"] = False

    # does the ini file exist?
    if not os.path.isfile(fname):
        raise ValueError("Inifile: %s not found!" % (fname))
        
    if not wabbit_tools.exists_ini_section(fname, "Wing"):
        raise ValueError("The ini file you specified does not contain the [Wing] section "+
                         "so maybe it is not a wing-shape file after all?")

    wtype = wabbit_tools.get_ini_parameter(fname, "Wing", "type", str)
    
    if wtype != "fourier":
        print(wtype)
        raise ValueError("Not a fourier wing. This function currently only supports a "+
                         "Fourier encoded wing (maybe with bristles)")
    
    # open the new figure:
    if fig is None:
        fig = plt.figure()    
        
    # -------------------------------------------------------------------------
    # damage (if present)
    # Drawn first as a background
    # -------------------------------------------------------------------------  
    damaged = wabbit_tools.get_ini_parameter(fname, "Wing", "damaged", bool, default=False)
    
    if damaged:
        # actual 0/1 damage mask:
        mask = wabbit_tools.get_ini_parameter( fname, 'Wing', 'damage_mask', dtype=float, vector=False, default=None, matrix=True )
        mask = mask.T
        
        # the bounding box of the mask is set here:
        bbox = wabbit_tools.get_ini_parameter( fname, 'Wing', 'corrugation_array_bbox', dtype=float, vector=True, default=None )
        
        n1, n2 = mask.shape[0], mask.shape[1]
        
        x1, x2 = np.linspace(bbox[0], bbox[1], num=n2, endpoint=True), np.linspace(bbox[2], bbox[3], num=n1, endpoint=True)
        X, Y = np.meshgrid(x2, x1)
        
        plt.pcolormesh(Y.T, X.T, mask, rasterized=True, cmap='gray_r')
        
        
    
        
    x0 = wabbit_tools.get_ini_parameter(fname, "Wing", "x0w", float)
    y0 = wabbit_tools.get_ini_parameter(fname, "Wing", "y0w", float)
    
    a0 = wabbit_tools.get_ini_parameter(fname, "Wing", "a0_wings", float)
    ai = wabbit_tools.get_ini_parameter(fname, "Wing", "ai_wings", float, vector=True)
    bi = wabbit_tools.get_ini_parameter(fname, "Wing", "bi_wings", float, vector=True)
    
    # compute outer wing shape (membraneous part)
    theta2 = np.linspace(-np.pi, np.pi, num=1024, endpoint=False)
    r_fft = Fserieseval(a0, ai, bi, (theta2 + np.pi) / (2.0*np.pi) )
    
    xc = x0+np.cos(theta2)*r_fft
    yc = y0+np.sin(theta2)*r_fft
    
    d = 0.1
    
    # plots wing outline
    plt.plot( xc, yc, 'r-', label='wing')
    
    plt.axis('equal')
    
    # plot the rotation axes
    plt.plot( [np.min(xc)-d, np.max(xc)+d], [0.0, 0.0], 'b--', label='rotation axis ($x^{(w)}$, $y^{(w)}$)')
    plt.plot( [0.0, 0.0], [np.min(yc)-d, np.max(yc)+d], 'b--')
    plt.grid()
    plt.legend()
    plt.title("wing shape visualization: \n"+fname)
    
    # -------------------------------------------------------------------------
    # bristles (if present)
    # -------------------------------------------------------------------------
    # if present, add bristles    
    bristles = wabbit_tools.get_ini_parameter(fname, "Wing", "bristles", bool, default=False)
    if bristles:
        bristles_coords = wabbit_tools.get_ini_parameter(fname, "Wing", "bristles_coords", matrix=True)
        print(bristles_coords.shape)
        for j in range( bristles_coords.shape[0]):
            plt.plot( [bristles_coords[j,0], bristles_coords[j,2]], [bristles_coords[j,1], bristles_coords[j,3]], 'r-')
            
            
    
        
    # plt.savefig( fname.replace('.ini','.pdf') )
    # plt.savefig( fname.replace('.ini','.svg') )
    plt.savefig( fname.replace('.ini','')+'_shape.png', dpi=300 )
    



def compute_inertial_power(file_kinematics='kinematics.t'):
    """
    Post-processing: Compute the insects inertial power. 
    
    The routine reads the kinematics.t file (for angular velocity and acceleration
    of the wings). The inertia tensor can be 
        (i) read from a PARAMS.INI file
        (ii) passed to this routine
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
