#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:41:23 2019

@author: tommy
"""
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


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


def RK4(t, u, rhs, dt, params):

    k1 = rhs(t, u, params)


    u2 = u + dt/2.0 * k1
    k2 = rhs(t + dt/2.0, u2, params)

    u3 =  u + dt/2.0 * k2
    k3 = rhs(t + dt/2.0, u3, params)

    u4 = u + dt*k3
    k4 = rhs(t + dt, u4, params)

    return( u + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4) )



def smoothstep( x, t, h):
    y = np.zeros( x.shape )
    y[ x<=t-h ] = 1.0
    y[ x>=t+h ] = 0.0
    y[ np.abs(x-t) < h ] = 0.5 * ( 1.0 + np.cos((x[ np.abs(x-t) < h ]-t+h)*np.pi/(2.0*h)) )

    return y



def chebychev(degree, x):
    """
    Return the value, first and second derivative of a chebychev polynomial
    of degree "degree" at position x

    Input:
        degree: (scalar, integer) degree of chebychev polynomial
        x: (scalar) position to be evaluated.

    Output:
        Tn, Tn_dx, Tn_dxdx: value, first and second derivative of chebychev polynomial

    """
    import scipy

    # chebychev polynomials of the first kind and degree "degree"
    cheby_first_kind = scipy.special.eval_chebyt

    # chebychev polynomials of the second kind and degree "degree"
    cheby_second_kind = scipy.special.eval_chebyu


    # chebychev polynomial at position s of degree degree
    # as well as derivatived in space,
    # see link: https://en.wikipedia.org/wiki/Chebyshev_polynomials#Differentiation_and_integration
    # for formulas

    # value
    Tn = cheby_first_kind( int(degree), x )

    # first derivative
    Tn_dx = degree * cheby_second_kind( int(degree)-1, x )

    # second derivative
    Tn_dxdx = degree * ( (degree+1.0)*cheby_first_kind(degree, x)  - cheby_second_kind(degree, x) ) / ( x**2 - 1 )

    return Tn, Tn_dx, Tn_dxdx

def cheby_first_kind_tommy(n, X):
    """
    Code adapted from the matlab script http://ceta.mit.edu/comp_spec_func/
    Acepts arrays as input, first flattens, than unflattens. (original code is for vectors)
    Code only computes chebychev of first kind and degree N (NOT second kind)

    Appears to be much more stable than scipy.special.eval_chebyt.

    we use this custom function instead of scipy-buildin because the latter does not
    yield stable results for large arguments to chebychev functions.

    """
    # flatten into vector
    x = X.copy()
    x = np.ndarray.flatten(x)

    a = 2.0
    b = 0.0
    c = 1.0
    y0 = 1.0
    y1 = 2.0 * x

    # the i'th position in pl corresponds to the i'th term
    # don't bother storing pl = 1
    pl = np.zeros( (x.shape[0], int(n)), dtype=np.complex128)

    y1 = x
    pl[:,0] = y1

    for k in np.arange(2,n-1+1):
        yn = (a*x + b) * y1 - c * y0
        pl[:, int(k-1)] = yn
        y0 = y1
        y1 = yn

    # unflatten into matrix
    y = np.reshape( pl[:,int(n-2)], X.shape )
    return (y)

def RKC_time_stepper(time, u, rhs, dt, params, s=20, eps=10.0):
    """
    Make one time step using an RKC scheme.

    Input:
    ------

        time : scalar float
            current time (at beginning of time step, old level)
        u :  numpy-ndarray
            solution at old time level
        rhs : numpy-ndarray
            function to be evaluated on. Assumes call of form RHS(time, u, params)
        dt : scalar-float
            time step size. no stability check performed here.
        params : dict
            parameter dict passed to RHS
        s : scalar, int
            number of stages for RKC scheme
        eps : scalar, float
            "damping" parameter for RKC scheme

    Output:
    -------
        u: solution at new time.
    """
    # ideally, we should do that only once.
    mu, mu_tilde, nu, gamma_tilde, c, eps = RKC_coefficients(s, eps)

    # changed: same indexing as verwer (not zero based, arrays are padded
    # with one leading zero)

    y00 = u
    y0 = u

    F0 = rhs(time, y00, params)
    y1 = y0 + mu_tilde[1] * dt * F0 # one-based indexing

    for j in np.arange(2, s+1):
        tau = time + c[j-1]*dt # one-based indexing, note this is j-1 not j
        F1 = rhs(tau, y1, params)

        y2 = (1.0-mu[j]-nu[j])*y00 + mu[j]*y1 + nu[j]*y0 + mu_tilde[j]*dt*F1 + gamma_tilde[j]*dt*F0

        y0 = y1
        y1 = y2

    return(y2)


def RKC_IMEX(u, rhs, dt, params, s=20, eps=10.0):
    """ RKC imex scheme. not tested: contains the bug (Feb 2020) with c-coefficients?
    """
    # ideally, we should do that only once.
    mu, mu_tilde, nu, gamma_tilde, c, eps = RKC_coefficients(s, eps)

    y00 = u
    y0 = u

    chi = -params['mask'] / params['C_eta']

    F0 = rhs(y00, params)
    y1 = y0 + mu_tilde[0] * dt * F0 # zero-based indexing

    # IMEX modification: eqn3.6 (Verwer SISC2004)
    # penalization applies to u only.
    y1[:,0] /= (1.0 - mu_tilde[0] * dt * chi)

    for i in np.arange(2, s+1): #, i = 2:s
#        tau = time + c(i)*dt;
        F1 = rhs(y1, params)

        y2 = (1.0-mu[i-1]-nu[i-1]) * y00
        y2 += mu[i-1] * y1
        y2 += nu[i-1] * y0
        y2 += mu_tilde[i-1]*dt*F1
        y2 += gamma_tilde[i-1]*dt*F0

        # IMEX: constant terms adding:
        y2[:,0] += (gamma_tilde[i-1] - (1.0-mu[i-1]-nu[i-1])*mu_tilde[i-1])*dt * (chi * y00[:,0])
        y2[:,0] += -nu[i-1] * mu_tilde[0] * dt * (chi*y0[:,0])

        # IMEX: implicit part
        y2[:,0] /= (1.0 - mu_tilde[0] * dt * chi)

        y0 = y1
        y1 = y2

    return(y2)


def RKC_coefficients( s, eps=10.0 ):
    """
    Return the coefficients for an explicit Runge-Kutta-Chebychev Scheme.

    Input:
    ------

        s : scalar, integer
            number of stages for the RKC scheme
        eps : scalar, float
            damping coefficient for RKC scheme. Not a real damping in the
            traditional sense. See Verwer et al JCP 2004.

    Damping:
    --------
    in the 1997 paper they set eps = 2 /13
    in the 2004 one (for advection) they set eps = 10

    Output:
    -------
        mu, mu_tilde, nu, gamma_tilde, c, eps
    """

    # allocation
    mu          = np.zeros( (s+1) )
    mu_tilde    = np.zeros( (s+1) )
    nu          = np.zeros( (s+1) )
    c           = np.zeros( (s+1) )
    gamma_tilde = np.zeros( (s+1) )
    b           = np.zeros( (s+1) )

    # the first element is the padding one for pythons zero-based indexing
    # set it to nan to notice if it is used by mistake.
    mu[0] = np.nan
    nu[0] = np.nan
    mu_tilde[0] = np.nan
    c[0] = np.nan
    gamma_tilde[0] = np.nan

    # change (29/01/2020): we now allocate arrays for
    # j = 0, ... , s (thus s+1 elements)
    # this way, we have a one-to-one correspondence with
    # ververs notation (i.e. in formula (2.1) of 1997 JCP,
    # we can really take mu[j] and NOT mu[j-1] )

    w0 = 1.0 + eps/(s**2)

    dummy, Ts_dx, Ts_dxdx = chebychev( s, w0 )
    w1 = Ts_dx / Ts_dxdx

    dummy, Tj_dx, Tj_dxdx = chebychev( 2, w0 )
    b[2] = Tj_dxdx / Tj_dx**2
    b[0] = b[2]

    # in the 1997 paper, this is b1 = b0
    # in the 2004 paper, this is b1 = 1/w0
    b[1] = 1.0 / w0

    # first entry of mu_tilde : w1 * b1
    mu_tilde[1] = w1 * b[1]

    # s+1 because it excludes the end point
    # Changed: we use the same indexing as Verwer did !
    # this just gives us a leading zero in the arrays
    for j in np.arange(2, s+1):
        # b_j
        dummy, Tj_dx, Tj_dxdx = chebychev( j, w0 )


        b[j] = Tj_dxdx / Tj_dx**2
        mu[j] = 2.0 * w0 * b[j] / b[j-1]
        nu[j] = -b[j] / b[j-2]
        mu_tilde[j] = 2.0 * b[j] * w1 / b[j-1]

        Tjm1, Tj_dx, Tj_dxdx = chebychev( j-1, w0 )
        gamma_tilde[j] = -(1.0 -b[j-1]*Tjm1) * mu_tilde[j]

        dummy, Ts_dx, Ts_dxdx = chebychev( s, w0 )
        dummy, Tj_dx, Tj_dxdx = chebychev( j, w0 )

        c[j] = Ts_dx * Tj_dxdx / (Ts_dxdx * Tj_dx)

    # this is taken from eqn (2.8) from the Verwer et al 2004 JCP.
    # other versions of this formula did not work with moving obstacles!
    c[1] = c[2]

    return mu, mu_tilde, nu, gamma_tilde, c, eps


def RKC_stability_map( s=4, eps=10.0, fig=None, color='k' ):
    """
    Plot a stability map of an RKC scheme with given parameters
    in a given figure (or open new figure)
    """
    s = np.float64(s)
    eps = np.float64(eps)

    if s<10:
        R = 100.0
    elif s<20 and s>10:
        R = 400.0
    elif s<40 and s>20:
        R = 800.0
    else:
        R = 2000.0

    im = np.linspace(-40.0, 40.0, 400, endpoint=True, dtype=np.float64)
    re = np.linspace(-R, 5.0, int(R), endpoint=True, dtype=np.float64)

    RE, IM = np.meshgrid(re, im)
    z = RE + 1j * IM


    w0 = 1.0 + eps/s**2
    Ts, Ts_dx, Ts_dxdx = chebychev( s, w0 )
    w1 = Ts_dx / Ts_dxdx

    bj = Ts_dxdx / Ts_dx**2
    aj = 1.0 - bj*Ts

    Pj = np.zeros( RE.shape, dtype=np.float64 )

    # chebychev polynomials of the first kind and degree "degree"
    # we use this custom function instead of scipy-buildin because the latter does not
    # yield stable results for large arguments to chebychev functions.
    cheby_first_kind = cheby_first_kind_tommy # scipy.special.eval_chebyt


    Ts = cheby_first_kind( s, w0 + w1*z )
    # this if the growth rate (<1 means stable)
    Pj = np.abs( aj + bj*Ts )


    if fig is None:
        fig = plt.figure()

    fig.gca().contour(RE, IM, Pj, levels=[1.0], colors=color)


def RK4_stability_map( fig=None ):
    """
    plot the stability map for a conventional RK4 scheme
    """
    im = np.linspace(-5.0, 5.0, 100, endpoint=True, dtype=np.float64)
    re = np.linspace(-5.0, 5.0, 100, endpoint=True, dtype=np.float64)

    RE, IM = np.meshgrid(re, im)

    z = RE + 1j * IM

    # this if the growth rate (<1 means stable)
    Pj = np.abs(  1 + z + 0.5*z**2 +(1/6)*z**3 + (1/24)*z**4 )


    if fig is None:
        fig = plt.figure()

    fig.gca().contour(RE, IM, Pj, levels=[1.0], colors='k',  linestyles='--', label='RK4')


def select_RKC_dt( eigenvalues, s=20, eps=10.0, RK4=False ):
    """
    Compute for a given RKC scheme and given eigenvalues of discrete operator
    the largest stable dt
    """
    s = np.float64(s)
    eps = np.float64(eps)
    # chebychev polynomials of the first kind and degree "degree"
    # we use this custom function instead of scipy-buildin because the latter does not
    # yield stable results for large arguments to chebychev functions.
    cheby_first_kind = cheby_first_kind_tommy

    # we need a good guess for dt (which will work), then we make it larger
    dt1 = 0.5 / np.max( np.imag(eigenvalues) ) # CFL type condition
    dt2 = -2.0 / np.min( np.real(eigenvalues) ) # real eigenvalues (2.0 is very strict!)
    dt = min( [dt1,dt2] )

    okay = True

    while okay:
        dt = dt*1.01

        z = dt*eigenvalues*1.1 # 10% security

        if not RK4:
            w0 = 1.0 + eps/s**2
            Ts, Ts_dx, Ts_dxdx = chebychev( s, w0 )
            w1 = Ts_dx / Ts_dxdx

            bj = Ts_dxdx / Ts_dx**2
            aj = 1.0 - bj*Ts

            Ts = cheby_first_kind( s, w0 + w1*z )
            # this if the growth rate (<1 means stable)
            Pj = np.abs( aj + bj*Ts )
        else:
            Pj = np.abs(  1 + z + 0.5*z**2 +(1/6)*z**3 + (1/24)*z**4 )

        if np.max(Pj) < 1.0:
            okay = True
        else:
            okay = False

    return dt

def select_RKC_scheme( eigenvalues, dt, plot=True ):
    """
    Given operator eigenvalues, select best stable RKC scheme.

    Input:
    ------
        eigenvalues : complex numpy array
            operator eigenvalues
        dt : float
            desired time step

    Output:
    -------
        s, eps : float
            parameters for RKC scheme. On screen, the copy-paste line for WABBIT
            *.ini files is printed.
    """
    # chebychev polynomials of the first kind and degree "degree"
    # we use this custom function instead of scipy-buildin because the latter does not
    # yield stable results for large arguments to chebychev functions.
    cheby_first_kind = cheby_first_kind_tommy

    eigenvalues *= dt
    z = eigenvalues

    S      = np.arange(4, 52+1, 1)
    EPS    = np.linspace(2.0/13.0, 20, 75)
    S, EPS = np.meshgrid(S, EPS)
    stable = S*0.0

    # check for each scheme (in the scanned range of s, eps) if it is stable
    # and store the result in a lookup table.
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            s = S[i,j]
            eps = EPS[i,j]

            w0 = 1.0 + eps/s**2
            Ts, Ts_dx, Ts_dxdx = chebychev( s, w0 )
            w1 = Ts_dx / Ts_dxdx

            bj = Ts_dxdx / Ts_dx**2
            aj = 1.0 - bj*Ts

            Ts = cheby_first_kind( s, w0 + w1*z )
            # this if the growth rate (<1 means stable)
            Pj = np.abs( aj + bj*Ts )

            if (np.max(Pj) < 1.0):
                # stable
                stable[i,j] = 1.0
            else:
                # unstable
                stable[i,j] = 0.0

    if np.max(stable)<1.0:
        raise ValueError("No stable scheme found!")

    # select the scheme. The "best" scheme is the one with smallest s (least stages)
    # and largest eps. The latter should make the choice more robust as larger eps
    # results in a more circle-like stability map with less dimpels.
    for i in range(S.shape[1]):
        s =  S[0,i]
        if np.max( stable[:,i] ) == 1.0:
            # find largest stable epsilon:
            for j in range(S.shape[0]):
                if stable[j,i] == 1.0:
                    eps_best = EPS[j,i]
                    s_best = S[j,i]
            break

    print(';-------------------')
    print('; Best RKC scheme given eigenvalues')
    print('; s=%2.1f eps=%3.3f' % (s_best, eps_best) )
    print('; dt=%e' % (dt))
    print('; cost = %5.1f [NRHS/T]' % (s_best/dt))
    print(';-------------------')


    mu, mu_tilde, nu, gamma_tilde, c, eps = RKC_coefficients(s_best, eps_best)

    # NOTE: the coefficients are padded with an first element due to pythons
    # 0-based indexing. This element is NAN for safety. It is cut here, as in
    # fortran, we use 1-based indexing

    def print_array(a, name):
        print('%s=' % (name), end="")
        for i in a[1:-1]:
            print("%e, " % (i), end="")
        print('%e;' % (a[-1]))

    print_array(mu, "RKC_mu")
    print_array(mu_tilde, "RKC_mu_tilde")
    print_array(nu, "RKC_nu")
    print_array(gamma_tilde, "RKC_gamma_tilde")
    print_array(c, "RKC_c")
    print("s=%i;" % (s_best))
    print(';-------------------')

    if plot:
        RKC_stability_map(s_best, eps_best)
        RK4_stability_map( fig=plt.gcf() )
        plt.plot( np.real(eigenvalues), np.imag(eigenvalues), 'o', mfc='none' )

    return s_best, eps_best



def piecewise_linear_universal( t, ti, ui ):
    """
    Piecewise linear interpolation. Given data points {ti, ui} (lists)
    return the function u(t).
    Periodization is applied.
    """
    # yes do include ti = 0.0    
    u = np.inf + np.zeros( t.shape )
    
    if len(ti) != len(ui):
        raise ValueError("not the same length??")
    
    for i in range(len(t)):
        T = t[i]        
        for j in np.arange(0, len(ti) ): 
            t1 = ti[j]           
            u1 = ui[j]
            
            if j == len(ti)-1:
                # periodization
                t2 = 1.0
                u2 = ui[0]
            else:                
                t2 = ti[j+1]
                u2 = ui[j+1]
                
            
            if T >= t1 and T < t2:
                # yes, this interval
                u[i] = u1 + (u2-u1) * (T-t1) / (t2-t1)           
        
    return u


def smooth(x, window_len=11, window='flat'):
    """
    smooth the data using a window with requested size.
    """
    import numpy

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # periodization
    s = numpy.r_[ x[-window_len-1:], x, x[0:window_len+1]]

    if window == 'flat': #moving average
        w = numpy.ones(window_len,'d')
    else:
        w = eval('numpy.'+window+'(window_len)')

    y = numpy.convolve( w/w.sum(), s, mode='same' )

    return y[window_len+1:-window_len-1]
