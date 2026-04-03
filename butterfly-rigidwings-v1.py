"""
Created on Tue Sep 23 13:11:38 2025

@author: engels
"""
import sys
#sys.path.append("E:\PhD_Schweitzer\CFD_Thomas\python-tools-master")
import numpy as np
from scipy.spatial import Delaunay
#import igl
import matplotlib.pyplot as plt
import pandas as pd
import insect_tools
from insect_tools import Rx, Ry, Rz
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import finite_differences
import fourier_tools
import os
from pathlib import Path
import scipy.ndimage
from scipy.interpolate import interp1d

import scipy

rad2deg = 180.0/np.pi


# dold = np.loadtxt('data/0g/frameData_2_simple_h__charithonia_0g.mat.csv.kineloader.old', skiprows=1)
# dnew = np.loadtxt('data/0g/frameData_2_simple_h__charithonia_0g.mat.csv.kineloader', skiprows=1)

# plt.figure()
# plt.plot( dold[:,0], dold[:,13]*rad2deg, 'r')
# plt.plot( dold[:,0], dold[:,15]*rad2deg, 'g')
# plt.plot( dold[:,0], dold[:,17]*rad2deg, 'b')

# plt.plot( dnew[:,0], dnew[:,13]*rad2deg, 'r--', label='psi new')
# plt.plot( dnew[:,0], dnew[:,15]*rad2deg, 'g--', label='beta new')
# plt.plot( dnew[:,0], dnew[:,17]*rad2deg, 'b--', label='gamma new')
# plt.legend()
# insect_tools.indicate_strokes(tstroke=2.0)
# raise






def best_rotation(A, B):
    """
    Compute best-fit transform that maps A -> B in least squares.
    A, B : (N,3) arrays of corresponding points (N >= 1).
    allow_scaling : if True, also estimate uniform scale c.
    Returns: R (3x3), t (3,), and c (scalar, 1.0 if not used)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    assert A.shape == B.shape and A.shape[1] == 3

    N = A.shape[0]
    if N == 0:
        raise ValueError("Need at least one correspondence")

    # covariance
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection correction
    if np.linalg.det(R) < 0:
        # fix reflection: flip sign of last column of Vt (or U)
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def rotation_align(V, ii, bc):
    """
    1) Compute rigid transform mapping V[b] -> bc
    2) Apply transform to entire mesh V -> V_aligned
    3) Return V_aligned, and the transform (R, t, c)
    """
    src = V[ii]            # source marker positions in rest pose
    tgt = np.asarray(bc)  # target measured positions
    R = best_rotation(src, tgt)
    # apply rotation to all vertices
    V_aligned = (V @ R.T)   # note: V @ R.T == (R @ V.T).T
    
    return V_aligned, (R)


def euler_angles_from_body_basis(ex_body, ey_body, ez_body):
    """
    Given orthonormal body basis vectors in the inertial frame,
    compute passive Euler angles (psi, beta, gamma)
    such that:
        x_body = Rx(psi) * Ry(beta) * Rz(gamma) * (x_global - x0)
    
    Returns
    -------
    psi, beta, gamma : floats
        Euler angles in radians.
    """
    # Build the rotation matrix (columns are body basis vectors)
    R = np.column_stack((ex_body, ey_body, ez_body))
    R = R.T
    
    # Extract angles from R = Rx(psi) * Ry(beta) * Rz(gamma)
    beta = np.arcsin(-R[0, 2])
    c_beta = np.cos(beta)
    
    
    # Handle potential singularity (gimbal lock)
    if abs(c_beta) > 1e-8:
        psi = np.arctan2(R[1, 2], R[2, 2])
        gamma = np.arctan2(R[0, 1], R[0, 0])
    else:
        # Gimbal lock: beta ≈ ±90°
        psi = 0.0
        gamma = np.arctan2(-R[1, 0], R[1, 1])
    
    return psi, beta, gamma



def euler_from_wing_basis(ex_wing, ey_wing, ez_wing, side='Left', eps=1e-12):
    """
    Compute Euler angles (alpha, theta, phi) for the wing coordinate system.

    Passive convention:
      Left:  x_wing = Ry(alpha) * Rz(theta) * Rx(phi) * (x_global - x0)
      Right: x_wing = Ry(-alpha) * Rz(theta) * Rx(-phi) * Rx(pi) * (x_global - x0)

    Parameters
    ----------
    ex_wing, ey_wing, ez_wing : array_like shape (3,)
        Body (wing) basis vectors expressed in the inertial frame (columns of R).
    side : {'Left', 'Right'}
        Which wing convention to follow.
    eps : float
        Small threshold to detect near-gimbal-lock (cos(theta) ~ 0).

    Returns
    -------
    alpha, theta, phi : floats (radians)
        Euler angles consistent with the chosen side convention.
    """
    ex = np.asarray(ex_wing, dtype=float)
    ey = np.asarray(ey_wing, dtype=float)
    ez = np.asarray(ez_wing, dtype=float)
    R = np.column_stack((ex, ey, ez))  # columns are body axes in global frame
    R = np.transpose(R)

    # For the Right wing, undo the final Rx(pi) so extraction matches the Left form:
    if side.lower() == 'right':
        # R_right = Ry(-alpha)*Rz(theta)*Rx(-phi)*Rx(pi)
        # so R_adj = R_right * Rx(pi).T = Ry(-alpha)*Rz(theta)*Rx(-phi)
        # We extract angles from R_adj and then flip alpha,phi signs.
        R = R @ Rx(np.pi).T

    # Make sure numerical noise doesn't push entries slightly outside [-1,1]
    # (not strictly necessary, but keeps sqrt/atan2 stable)
    R = np.clip(R, -1.0, 1.0)

    # Robust extraction:
    # R = Ry(alpha) * Rz(theta) * Rx(phi)
    # matrix elements we use:
    # R[0,0] = cos(alpha)*cos(theta)
    # R[1,0] = -sin(theta)
    # R[2,0] = sin(alpha)*cos(theta)
    # R[1,1] = cos(phi)*cos(theta)
    # R[1,2] = sin(phi)*cos(theta)

    # theta: use atan2 with hypotenuse for numerical stability
    denom = np.hypot(R[0,0], R[2,0])  # equals |cos(theta)| (up to numerical noise)
    theta = np.arctan2(-R[1,0], denom)

    # alpha: atan2(R[2,0], R[0,0]) (safe even if denom small, atan2 handles it)
    alpha = np.arctan2(R[2,0], R[0,0])

    # phi: from R[1,1] and R[1,2]: phi = atan2(R[1,2], R[1,1])
    # but when cos(theta) ~ 0 these entries are ~0 and phi becomes ambiguous (gimbal lock)
    if denom > eps:
        phi = np.arctan2(R[1,2], R[1,1])
    else:
        # Gimbal lock case: cos(theta) ~= 0 -> theta ~= +-pi/2
        # We can set phi = 0 and recover a combined angle for alpha (or choose a stable convention).
        # Here we choose phi = 0 and compute a representative alpha:
        phi = 0.0
        # When cos(theta) ~ 0, R simplifies and we can get alpha from other elements:
        # For theta ~ +pi/2 => R[1,0] ~ -1
        # Use elements R[0,1], R[0,2] or R[2,1], R[2,2] if desired.
        # We'll leave alpha as computed above (atan2(R[2,0], R[0,0])) which is still meaningful.
        # (Alternative strategies are possible depending on application.)

    # If we processed the right side, flip alpha and phi signs to match your parameterization
    if side.lower() == 'right':
        alpha = -alpha
        phi = -phi
        
        # verification
        M_b2w = (Ry(-alpha) @ Rz(theta) @ Rx(-phi) @ Rx(np.pi) )
        
        e = np.linalg.norm(ex-M_b2w[0,:]) + np.linalg.norm(ey-M_b2w[1,:]) + np.linalg.norm(ez-M_b2w[2,:])
        if (e>1e-12):
            raise ValueError("Wrong result> right")
    else:
        # verification
        M_b2w = (Ry(alpha) @ Rz(theta) @ Rx(phi) )
        
        e = np.linalg.norm(ex-M_b2w[0,:]) + np.linalg.norm(ey-M_b2w[1,:]) + np.linalg.norm(ez-M_b2w[2,:])
        if (e>1e-12):
            raise ValueError("Wrong result> left")
        
    

    return alpha, theta, phi



def symmetry_plane_normal(x_left, x_right):
    """
    Estimate the normal vector to the approximate symmetry plane
    between corresponding 3D points in x_right and x_left.
    
    The output normal points toward x_left.
    
    Parameters
    ----------
    x_right : (N, 3) array
        Points on the right side.
    x_left : (N, 3) array
        Points on the left side (corresponding to x_right).
    
    
    Returns
    -------
    normal : (3,) array
        Unit normal vector pointing toward x_left.
    plane_point : (3,) array
        A point on the estimated symmetry plane (centroid of midpoints).
    """

    # Ensure numpy arrays
    x_right = np.asarray(x_right)
    x_left = np.asarray(x_left)

    # Check sizes
    assert x_right.shape == x_left.shape, "x_right and x_left must have same shape"
    assert x_right.shape[1] == 3, "Points must be 3D"

    # Compute midpoints and difference vectors
    midpoints = 0.5 * (x_left + x_right)
    diffs = x_left - x_right

    # Estimate normal direction as dominant direction of differences
    # via SVD (principal component of the difference vectors)
    _, _, vh = np.linalg.svd(diffs, full_matrices=False)
    normal = vh[0]  # first right-singular vector (largest variance direction)
    normal /= np.linalg.norm(normal)

    # Make sure it points toward x_left
    mean_diff = np.mean(diffs, axis=0)
    if np.dot(normal, mean_diff) < 0:
        normal = -normal

    # Plane point (mean of midpoints)
    plane_point = np.mean(midpoints, axis=0)
    return normal, plane_point


def plot_line(name1, name2, color='k'):
    x1, x2 = df[name1+'.x'][it], df[name2+'.x'][it]
    y1, y2 = df[name1+'.y'][it], df[name2+'.y'][it]
    z1, z2 = df[name1+'.z'][it], df[name2+'.z'][it]
    
    plt.plot( [x1,x2], [y1,y2], [z1,z2], '-', color=color)
    
    

def fill_nan_linear(y):
    """Interpole linéairement les NaN. Si tout NaN -> renvoie tel quel."""
    y = np.asarray(y, dtype=float)
    n = y.size
    x = np.arange(n)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y
    f = interp1d(x[mask], y[mask], kind='linear', fill_value='extrapolate')
    y_filled = y.copy()
    y_filled[~mask] = f(x[~mask])
    return y_filled

def hampel_filter(y, k=21, n_sigmas=3.0):
    """
    Filtre Hampel 1D (robuste) : remplace les outliers par la médiane locale.
    k = taille de fenêtre.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    y_out = y.copy()
    half = k // 2

    for i in range(n):
        i0 = max(0, i-half)
        i1 = min(n, i+half+1)
        w = y[i0:i1]
        w = w[np.isfinite(w)]
        if w.size < 5:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))  # median absolute deviation
        if mad < 1e-12:
            continue
        # approx sigma from MAD for normal dist
        sigma = 1.4826 * mad
        if np.isfinite(y[i]) and np.abs(y[i] - med) > n_sigmas * sigma:
            y_out[i] = med
    return y_out

def smooth_point_in_df(df, tail_key='Tail', hampel_k=31, hampel_sig=3.0, smooth_size=151):
    """
    - Hampel pour virer les gros outliers
    - interpolation des NaN
    - lissage fort uniform_filter1d
    """
    for coord in ['x','y','z']:
        col = f'{tail_key}.{coord}'
        y = df[col].to_numpy(dtype=float)

        # 1) outliers robustes
        y = hampel_filter(y, k=hampel_k, n_sigmas=hampel_sig)

        # 2) fill NaN
        y = fill_nan_linear(y)

        # 3) lissage fort (moyenne glissante)
        # mode='nearest' évite les bords qui partent en vrille
        y = scipy.ndimage.uniform_filter1d(y, size=smooth_size, mode='nearest')

        df[col] = y

    return df



# list of all markers on the animal
all_keys = ['Head','LeftAntenna','LeftDelta','LeftLine1Bottom','LeftLine1Top','LeftLine2Bottom','LeftLine2Top','LeftLine3Bottom',
            'LeftLine3Middle','LeftLine4Bottom','LeftLine5Far','LeftLine5Close','LeftWingTip','RightAntenna','RightDelta','RightLine1Bottom',
            'RightLine1Top','RightLine2Bottom','RightLine2Top','RightLine3Bottom','RightLine3Middle','RightLine4Bottom','RightLine5Close',
            'RightLine5Far','RightWingTip','Tail','Torso']       

# note you have to use them with Left/right prefix:
wing_keys = ['Line3Middle', 'Line3Bottom', 'Line2Top', 'Line2Bottom', 'Line1Top', 'Line1Bottom', 'WingTip', 'Line5Far', 'Line4Bottom', 'Line5Close']



#%% SETTINGS                                        
# read in processed csv file (contains only the valid frames)
# NOTE: for the 1G case, the original data are left/right inverted.
"""
for DATA in ['0g', '1g', '2g-new']:

    root = '/home/engels/Documents/Research/Insects/3D/projects/butterflies_lyon/data/'+DATA+'/'
    if "2g/" in root:
        file = 'frameData_2_simple_h__charithonia_2g.mat.csv'
    elif "2g-new" in root:
        file = 'frameData_2_simple_h__charithonia_30_09_2025_vol_1_S3_991_10_991_85_fs77_fe3673.mat.csv'
    elif "1g" in root:
        file = 'frameData_simple_h__charithonia_1g.mat.csv'
    elif "0g" in root:
        file = 'frameData_2_simple_h__charithonia_0g.mat.csv'
    """
for DATA in ['0g']:

    root = 'E:/PhD_Schweitzer/CFD_Thomas/test_lissage_body/'
    file = 'frameData_simple_h__charithonia_02_10_2025_vol_3_S2_406_20_409_fs78_fe13799.mat.csv'
        
    d = np.loadtxt( root+file, skiprows=1, delimiter=';')
    df = pd.read_csv( root+file, delimiter=';')
    
    # lissage fort de Tail (sur données brutes)
    df = smooth_point_in_df(df,
                           tail_key='Tail',
                           hampel_k=31,
                           hampel_sig=3.0,
                           smooth_size=251) #smooth_size=251; hampel_k=31
    

    df = smooth_point_in_df(df,
                           tail_key='Torso',
                           hampel_k=31,
                           hampel_sig=3.0,
                           smooth_size=251)
    

    df = smooth_point_in_df(df,
                           tail_key='Head',
                           hampel_k=31,
                           hampel_sig=3.0,
                           smooth_size=251)
    
    # this can be merged into the matlab code, inlcuding the switch left/right renaming
    # Motion starts at 0,0,0
    for key in all_keys:
        # copying and translation            
        df[key+'.x'] = df[key+'.x'] - df.loc[0,'Torso.x']
        df[key+'.y'] = df[key+'.y'] - df.loc[0,'Torso.y']
        df[key+'.z'] = df[key+'.z'] - df.loc[0,'Torso.z']
    
    df_org = df.copy()
    df_body = df.copy()
    
    nt = d.shape[0]
    Npoints = int( (d.shape[1]-1)/3 )
    
    plot3d  = False
    translate = False # used only if project==False
    plot_markers = True
    force_recompute = True
    collison_check = False
    pin_body_output = False
    
    istart, iend = 0, nt #500, 5200 #340, 5000
    # istart, iend = 1270,1275
    step = 10 # only for plotting
    # This is the forewing length, but the same number is used for the hindwing
    # (=normalization done by forewing)
    R_wing = 3.76*10 # mm take the right R_wing (I rescaled with factor s = R_ref/R_true) with R_ref = 3.76 cm
    
    file_kineloader = root+file+'.kineloader'
    
    gray = 3*[0.8]
    norm = np.linalg.norm
    
    #%% Part 1 computation of body system
    # first pass: get body angles, apply agressive filter to roll angle
    # in the later parts, those angles are used to define the body coordinate system
    #
    # NOTE: regardless of how we define the body system, the wing trajectory in global space will
    # be exactly the same. We determine the wing angles relative to the body such that the orientation
    # is as measured, hence, applying smoothing to the body angles directly affects the wing angles (which
    # then result in exactly the same orientation as before.)
    #
    #
    if force_recompute or not os.path.isfile(file_kineloader):
        psi, beta, gamma = np.zeros(iend), np.zeros(iend), np.zeros(iend)
        
        for it in range(istart, iend, 1):    
            if it % 500 ==0:
                print( "pass 1: %i%%" % (100*it/(iend-istart+1)) )
    
            # computation of body system
            # ex_body_g = np.asarray( [df['Torso.x'][it], df['Torso.y'][it], df['Torso.z'][it]] ) -  np.asarray( [df['Head.x'][it], df['Head.y'][it], df['Head.z'][it]] )
            # ex_body_g /= -np.linalg.norm(ex_body_g)
            
            # Body x-vector: best approximation to the line Head-Torso-Tail 
            """
            p = np.asarray( [[df['Torso.x'][it],df['Torso.y'][it],df['Torso.z'][it]],
                             [df['Head.x'][it],df['Head.y'][it],df['Head.z'][it]],
                             [df['Tail.x'][it],df['Tail.y'][it],df['Tail.z'][it]]] )
            """
            p = np.asarray( [[df['Torso.x'][it],df['Torso.y'][it],df['Torso.z'][it]],
                             [df['Head.x'][it],df['Head.y'][it],df['Head.z'][it]]] )
            
            p -= np.mean(p, axis=0)
            # SVD yields the vector
            _, _, Vt = np.linalg.svd(p)
            # normalization and sign determination
            ex_body_g = Vt[0] / np.linalg.norm(Vt[0])
            if np.dot( ex_body_g, p[1] ) < 0: # points towards the head
                ex_body_g *= -1.0
            
            side = 'Right'
            x_right = []
            for wkey in wing_keys:
                x_right.append( np.array( [df[side+wkey+'.x'][it], df[side+wkey+'.y'][it], df[side+wkey+'.z'][it]] ) )
            x_right = np.asarray(x_right)
            
            side = 'Left'
            x_left = []
            for wkey in wing_keys:
                x_left.append( np.array( [df[side+wkey+'.x'][it], df[side+wkey+'.y'][it], df[side+wkey+'.z'][it]] ) )
            x_left = np.asarray(x_left)
            
            ey_body_g, _ = symmetry_plane_normal(x_left, x_right) # note left-right inversion
            # ey is not necessarily orthogonal to ex
            ey_body_g = ey_body_g - np.dot(ey_body_g, ex_body_g) * ex_body_g
            # re-normalize
            ey_body_g /= np.linalg.norm(ey_body_g)
                
            ez_body_g = np.cross(ex_body_g, ey_body_g)
            ez_body_g /= np.linalg.norm(ez_body_g)
            
            psi[it], beta[it], gamma[it] = euler_angles_from_body_basis(ex_body_g, ey_body_g, ez_body_g)    
            
            # error-checking: computation of body system        
            M_b2g = insect_tools.get_M_g2b(psi[it], beta[it], gamma[it])#.T    
        
            e = np.linalg.norm(ex_body_g-M_b2g[0,:]) + np.linalg.norm(ey_body_g-M_b2g[1,:]) + np.linalg.norm(ez_body_g-M_b2g[2,:])
            if (e>1e-12):
                raise
            
          
        # remove jumps by 2*pi
        # psi1, beta1, gamma1 = np.unwrap(psi), np.unwrap(beta), np.unwrap(gamma)
        psi1, beta1, gamma1 = psi, beta, gamma
        
            
        # apply the very agressive smoothing filter
        # we do this because from francois data, the roll angle cannot be estimated reliably
        fsize = 250
        psi = scipy.ndimage.uniform_filter1d(psi1, size=fsize) 
        beta = scipy.ndimage.uniform_filter1d(beta1, size=fsize) 
        gamma = scipy.ndimage.uniform_filter1d(gamma1, size=fsize) 
        
        
        #%% Part 2 projection and wings angles
        alpha_L, theta_L, phi_L = np.zeros(nt), np.zeros(nt), np.zeros(nt)
        alpha_R, theta_R, phi_R = np.zeros(nt), np.zeros(nt), np.zeros(nt)
        err_L, err_R = np.zeros(nt), np.zeros(nt)
        opening_angle_L, opening_angle_R = np.zeros(nt), np.zeros(nt)
            
        # performace tuning (attention needs to be done for fore- and hind wing 
        # individually so here works only for singlewing)
        dx = 2.5e-2
        xc, yc = insect_tools.get_wing_membrane_grid( 'E:/PhD_Schweitzer/CFD_Thomas/for_francois/simulations_raw_data/2g-new/singlewing.ini', dx=dx, dy=dx)
        x_wing_w = np.zeros( (xc.shape[0], 3) )
        x_wing_w[:,0] = xc*R_wing # mm 
        x_wing_w[:,1] = yc*R_wing # mm 
        
        # helper: find index of nearest vertex to a 2D coordinate
        from scipy.spatial import cKDTree
        kdt = cKDTree(x_wing_w)
        
        marker_w = np.asarray([[-0.011858,0.362999,0],
                                [-0.199664,0.606422,0],
                                [0.129061,0.487015,0],
                                [-0.132625,0.785551,0],
                                [0.121500,0.781098,0],
                                [-0.105160,0.913530,0],
                                [0.002797,1.000067,0],
                                [-0.349130,0.386808,0],
                                [-0.270419,0.438344,0],
                                [-0.288587,0.002673,0]]) * R_wing # mm
            
        _, ii = kdt.query(marker_w)    
        
        #---------------------------------------------------
        #---------------------------------------------------
        # code for adjusting hindwing opening angle
        marker_HW_w = np.asarray([[-0.349130,0.386808,0], [-0.270419,0.438344,0], [-0.288587,0.002673,0]]) * R_wing # mm
        
        dx = 2.5e-2
        xch, ych = insect_tools.get_wing_membrane_grid( 'E:/PhD_Schweitzer/CFD_Thomas/for_francois/simulations_raw_data/1g/singlewing.ini', dx=dx, dy=dx)
        # xch, ych = insect_tools.get_wing_membrane_grid( 'model/wing_model/singlewing.ini', dx=dx, dy=dx)
        x_HW_w = np.zeros( (xch.shape[0], 3) )
        x_HW_w[:,0] = xch*R_wing # mm 
        x_HW_w[:,1] = ych*R_wing # mm 
    
        
        kdt_HW = cKDTree(x_HW_w)
        _, ij = kdt_HW.query(marker_HW_w)  
        #---------------------------------------------------
        #---------------------------------------------------
            
        for it in range(istart, iend, 1): 
            if it % 250 == 0:
                print( "pass 2: %i%%" % (100*it/(iend-istart+1)) )
            
            # computation of body system
            # now psi is filtered
            M_b2g = insect_tools.get_M_g2b(psi[it], beta[it], gamma[it])
            # extract basis vectors from rotation matrix
            ex_body_g = M_b2g[0,:]
            ey_body_g = M_b2g[1,:] 
            ez_body_g = M_b2g[2,:]    
            
            # project data on the body coordinate system, store result in df
            for key in all_keys:
                # copying and translation            
                x_old, y_old, z_old = df_org[key+'.x'][it]-df_org['Torso.x'][it], df_org[key+'.y'][it]-df_org['Torso.y'][it], df_org[key+'.z'][it]-df_org['Torso.z'][it]
                # projection on new unit vectors          
                df_body.loc[it, key+'.x'] = x_old*ex_body_g[0] + y_old*ex_body_g[1] + z_old*ex_body_g[2]
                df_body.loc[it, key+'.y'] = x_old*ey_body_g[0] + y_old*ey_body_g[1] + z_old*ey_body_g[2]
                df_body.loc[it, key+'.z'] = x_old*ez_body_g[0] + y_old*ez_body_g[1] + z_old*ez_body_g[2]                
             
            for side in ['Right', 'Left']:
                # create target positions marker_measured_b (shape: #b x 3)
                marker_measured_b = x_wing_w[ii].copy()
                # new marker positions (in body system; we did the projection above)
                marker_measured_b[0, 0:2+1] = np.array( [df_body[side+'Line3Middle.x'][it], df_body[side+'Line3Middle.y'][it], df_body[side+'Line3Middle.z'][it]] )
                marker_measured_b[1, 0:2+1] = np.array( [df_body[side+'Line3Bottom.x'][it], df_body[side+'Line3Bottom.y'][it], df_body[side+'Line3Bottom.z'][it]] )
                marker_measured_b[2, 0:2+1] = np.array( [df_body[side+'Line2Top.x'][it]   , df_body[side+'Line2Top.y'][it]   , df_body[side+'Line2Top.z'][it]] )
                marker_measured_b[3, 0:2+1] = np.array( [df_body[side+'Line2Bottom.x'][it], df_body[side+'Line2Bottom.y'][it], df_body[side+'Line2Bottom.z'][it]] )
                marker_measured_b[4, 0:2+1] = np.array( [df_body[side+'Line1Top.x'][it]   , df_body[side+'Line1Top.y'][it]   , df_body[side+'Line1Top.z'][it]] )
                marker_measured_b[5, 0:2+1] = np.array( [df_body[side+'Line1Bottom.x'][it], df_body[side+'Line1Bottom.y'][it], df_body[side+'Line1Bottom.z'][it]] )
                marker_measured_b[6, 0:2+1] = np.array( [df_body[side+'WingTip.x'][it]    , df_body[side+'WingTip.y'][it]    , df_body[side+'WingTip.z'][it]] )
                marker_measured_b[7, 0:2+1] = np.array( [df_body[side+'Line5Far.x'][it]   , df_body[side+'Line5Far.y'][it]   , df_body[side+'Line5Far.z'][it]] )
                marker_measured_b[8, 0:2+1] = np.array( [df_body[side+'Line4Bottom.x'][it], df_body[side+'Line4Bottom.y'][it], df_body[side+'Line4Bottom.z'][it]] )
                marker_measured_b[9, 0:2+1] = np.array( [df_body[side+'Line5Close.x'][it] , df_body[side+'Line5Close.y'][it] , df_body[side+'Line5Close.z'][it]] )
            
                x_wing_b, (R) = rotation_align(x_wing_w, ii, marker_measured_b)
                
                
                
                # determine angles assuming fore-and hindwing coupled in reference configuration
                if side == 'Right':
                    alpha_R[it], theta_R[it], phi_R[it] = euler_from_wing_basis( R[:,0], R[:,1], R[:,2], 'right')
                    err_R[it] = np.mean( np.linalg.norm(x_wing_b[ii]-marker_measured_b, axis=1) ) / R_wing
                    
                    marker_HW_b = x_HW_w[ij].copy()
                    marker_HW_b[0, 0:2+1] = np.array( [df_body[side+'Line5Far.x'][it]   , df_body[side+'Line5Far.y'][it]   , df_body[side+'Line5Far.z'][it]] )
                    marker_HW_b[1, 0:2+1] = np.array( [df_body[side+'Line4Bottom.x'][it], df_body[side+'Line4Bottom.y'][it], df_body[side+'Line4Bottom.z'][it]] )
                    marker_HW_b[2, 0:2+1] = np.array( [df_body[side+'Line5Close.x'][it] , df_body[side+'Line5Close.y'][it] , df_body[side+'Line5Close.z'][it]] )
                    
                    num = 50
                    distance = np.zeros((num))
                    thetas = np.linspace(-20, 20, num)
                    for t, theta_opening in enumerate(thetas): # deg
                        theta_opening = theta_opening / rad2deg # rad
                        
                        M_b2w_R = insect_tools.get_M_b2w(alpha_L[it], theta_L[it], phi_L[it], 0.0, 'right') 
                        
                        # assumes hinge position to be [0,0,0] in body system
                        x_HW_b = ( (insect_tools.Rz(theta_opening) @ M_b2w_R.T) @ x_HW_w.T).T                    
                        
                        distance[t] = norm(x_HW_b[ij[0]] - marker_HW_b[0]) + norm(x_HW_b[ij[1]] - marker_HW_b[1]) + norm(x_HW_b[ij[2]] - marker_HW_b[2]) 
    
                    opening_angle_R[it] = thetas[np.argmin(distance)] # deg
                    
                else:
                    alpha_L[it], theta_L[it], phi_L[it] = euler_from_wing_basis( R[:,0], R[:,1], R[:,2], 'left')
                    err_L[it] = np.mean( np.linalg.norm(x_wing_b[ii]-marker_measured_b, axis=1) ) / R_wing
                    
                    # # if it == 500:
                        
                    # fig = plt.figure()
                    # ax = plt.gcf().add_subplot(1,1,1, projection='3d')
                    # plt.plot( marker_measured_b[:,0], marker_measured_b[:,1], marker_measured_b[:,2], 'c*')
                    # plt.plot( x_wing_b[:,0], x_wing_b[:,1], x_wing_b[:,2], 'ks', mfc='none')
                    # plt.plot( x_wing_b[ii,0], x_wing_b[ii,1], x_wing_b[ii,2], 'ro', mfc='none')
                    
    
                    
                    # plt.figure()
                    # plt.plot(x_HW_w[:,0], x_HW_w[:,1], "bo" )
                    # plt.plot(x_HW_w[ij,0], x_HW_w[ij,1], "ro" )
                    
                    marker_HW_b = x_HW_w[ij].copy()
                    marker_HW_b[0, 0:2+1] = np.array( [df_body[side+'Line5Far.x'][it]   , df_body[side+'Line5Far.y'][it]   , df_body[side+'Line5Far.z'][it]] )
                    marker_HW_b[1, 0:2+1] = np.array( [df_body[side+'Line4Bottom.x'][it], df_body[side+'Line4Bottom.y'][it], df_body[side+'Line4Bottom.z'][it]] )
                    marker_HW_b[2, 0:2+1] = np.array( [df_body[side+'Line5Close.x'][it] , df_body[side+'Line5Close.y'][it] , df_body[side+'Line5Close.z'][it]] )
                    
                    num = 50
                    distance = np.zeros((num))
                    thetas = np.linspace(-20, 20, num)
                    for t, theta_opening in enumerate(thetas): # deg
                        theta_opening = theta_opening / rad2deg # rad
                        
                        M_b2w_L = insect_tools.get_M_b2w(alpha_L[it], theta_L[it], phi_L[it], 0.0, 'left') 
                        
                        # assumes hinge position to be [0,0,0] in body system
                        x_HW_b = ( (insect_tools.Rz(theta_opening) @ M_b2w_L.T) @ x_HW_w.T).T
                        # x_HW_b = ( (insect_tools.Rz(theta_opening) @ R) @ x_HW_w.T).T
                        
                        
                        
                        distance[t] = norm(x_HW_b[ij[0]] - marker_HW_b[0]) + norm(x_HW_b[ij[1]] - marker_HW_b[1]) + norm(x_HW_b[ij[2]] - marker_HW_b[2]) 
                        
                    # plt.figure()
                    # plt.plot(thetas, distance)
                    
                    theta_final = thetas[np.argmin(distance)]
                    opening_angle_L[it] = theta_final # deg
                    # x_HW_b = ( (insect_tools.Rz(theta_final/rad2deg) @ M_b2w_L.T) @ x_HW_w.T).T
                    
                    # fig = plt.figure()
                    
                    # ax = plt.gcf().add_subplot(1,1,1, projection='3d')
                    # # singlewing:
                    # plt.plot( x_wing_b[:,0], x_wing_b[:,1], x_wing_b[:,2], 'ks', mfc='none')
                    # # the hindwing in best orientation
                    # plt.plot( x_HW_b[:,0], x_HW_b[:,1], x_HW_b[:,2], 'bo', mfc='none')
                    # plt.plot( x_HW_b[ij,0], x_HW_b[ij,1], x_HW_b[ij,2], 'ms')
                    # plt.plot( marker_HW_b[:,0], marker_HW_b[:,1], marker_HW_b[:,2], 'r*')
                    
                    
                    
                    
                    # # raise
        #%% Time vector  
        # all vectors same length
        nk = nt
          
        # compute time normalization using dominant frequency in phi
        u = phi_L - np.mean(phi_L)
        k, ek = fourier_tools.spectrum1(u)
        """
        k, ek = fourier_tools.spectrum1( phi_L )
        """
        f = (k[ np.argmax(ek) ]*2.0*np.pi/nt) / (2.0*np.pi) # frequency in 1/frames
        T = 1.0 / f # cycle time in frames
        # time vector
        time = np.arange(istart, iend) / T
        
        phi_L_1, alpha_L_1, theta_L_1 =  phi_L, alpha_L, theta_L
        phi_R_1, alpha_R_1, theta_R_1 = phi_R, alpha_R, theta_R
        
        phi_L, alpha_L, theta_L = np.unwrap(phi_L_1), np.unwrap(alpha_L_1), np.unwrap(theta_L_1)
        phi_R, alpha_R, theta_R = np.unwrap(phi_R_1), np.unwrap(alpha_R_1), np.unwrap(theta_R_1)


        
        print('Simulation time up to t=%f' % (time[-1]))
        print('Frequency in physical units f=%2.2f Hz' % (f*5000))
        
        
        
        #%% Output: kineloader
        fig1 = plt.figure()
        plt.subplot(231)
        plt.plot(time, psi1*rad2deg, "-", label='roll')
        plt.plot(time, beta1*rad2deg, "-", label='pitch')
        plt.plot(time, gamma1*rad2deg, "-", label="yaw")
        
        plt.plot(time, psi*rad2deg, "k--", label='roll filtered')
        plt.plot(time, beta*rad2deg, "k--", label='pitch filtered')
        plt.plot(time, gamma*rad2deg, "k--", label="yaw filtered")
        plt.legend()
        plt.grid(True)
        plt.title('Body angles')
        
        insect_tools.indicate_strokes(tstroke=2.0)
        
        plt.subplot(232)
        plt.plot(time,  phi_L*rad2deg, label='phi L')
        plt.plot(time,  alpha_L*rad2deg, label='alpha L')
        plt.plot(time,  theta_L*rad2deg, label='theta L')
        insect_tools.reset_colorcycle(plt.gca())
        plt.plot(time,  phi_R*rad2deg, label='phi R')
        plt.plot(time,  alpha_R*rad2deg, label='alpha R')
        plt.plot(time,  theta_R*rad2deg, label='theta R')
        plt.grid(True)
        plt.legend()
        plt.title('wing angles')
        
        insect_tools.indicate_strokes(tstroke=2.0)
        
        plt.subplot(236)
        plt.plot(time, err_L, label="left")
        plt.plot(time, err_R, label='right')
        plt.grid(True)
        plt.legend()
        plt.title('mean error (mean difference of imposed marker \nposition (rigidwing) to measured marker position')
        
        insect_tools.indicate_strokes(tstroke=2.0)
        
        plt.subplot(235)
        plt.plot(time, opening_angle_L, label="left wing opening angle")
        plt.plot(time, opening_angle_R, label="right wing opening angle")
        plt.grid(True)
        plt.legend()
        
        # d,h = insect_tools.load_t_file('data/1g/dry/kinematics.t', return_header=True)    
        # plt.plot( d[:,0], d[:,8]*rad2deg, 'o',label='alphaL2')
        # plt.plot( d[:,0], d[:,9]*rad2deg, 'o',label='phil2')
        # plt.plot( d[:,0], d[:,10]*rad2deg, 'o',label='theta L dry')    
        # plt.legend()
        
        insect_tools.indicate_strokes(tstroke=2.0)
        

    
        dt = time[1]-time[0]
        D = finite_differences.D12(nk, dt)
        
        
        # kineloader data file
        # open file, erase existing
        f = open( file_kineloader, 'w', encoding='utf-8' )
        
        phi_R_dt   = D @ phi_R
        alpha_R_dt = D @ alpha_R
        theta_R_dt = D @ theta_R
        
        phi_L_dt   = D @ phi_L
        alpha_L_dt = D @ alpha_L
        theta_L_dt = D @ theta_L
        
        psi_dt   = D @ psi
        gamma_dt = D @ gamma
        beta_dt  = D @ beta
        
        # relative to starting point and normalized by wing length
        x = (df_org['Torso.x'] - df_org.loc[0, 'Torso.x'])/R_wing
        y = (df_org['Torso.y'] - df_org.loc[0, 'Torso.y'])/R_wing
        z = (df_org['Torso.z'] - df_org.loc[0, 'Torso.z'])/R_wing
        
        x_dt = D @ x
        y_dt = D @ y
        z_dt = D @ z
        
        f.write('; header\n')
            
        for it in range(nk):        
            f.write( '%+.9e ' % (time[it])) #0
            
            if pin_body_output:
                if it < 7:
                    print('WARNING MOTION INHIBITED it=%i' %(it))
                for k in range(6):    
                    f.write( '%+.9e ' % (0.0)) 
            else:
                f.write( '%+.9e ' % (x[it]) )
                f.write( '%+.9e ' % (x_dt[it]))
                f.write( '%+.9e ' % (y[it]) )
                f.write( '%+.9e ' % (y_dt[it]))
                f.write( '%+.9e ' % (z[it]) )
                f.write( '%+.9e ' % (z_dt[it]))
            
            for k in range(6):    
                f.write( '%+.9e ' % (0.0)) # unused, was planned for velocity
            
            f.write( '%+.9e ' % (psi[it])) # psi
            f.write( '%+.9e ' % (psi_dt[it])) # psi
                           
            f.write( '%+.9e ' % (beta[it])) # beta
            f.write( '%+.9e ' % (beta_dt[it])) # beta
            
            f.write( '%+.9e ' % (gamma[it])) # gamma
            f.write( '%+.9e ' % (gamma_dt[it])) # gamma
            
            f.write( '%+.9e ' % (alpha_L[it]))
            f.write( '%+.9e ' % (alpha_L_dt[it]))
                    
            f.write( '%+.9e ' % (phi_L[it])) # newcode: radians!
            f.write( '%+.9e ' % (phi_L_dt[it]))
            
            f.write( '%+.9e ' % (theta_L[it]))
            f.write( '%+.9e ' % (theta_L_dt[it]))                 
    
            f.write( '%+.9e ' % (alpha_R[it]))
            f.write( '%+.9e ' % (alpha_R_dt[it]))
                    
            f.write( '%+.9e ' % (phi_R[it])) # newcode: radians!
            f.write( '%+.9e ' % (phi_R_dt[it]))
            
            f.write( '%+.9e ' % (theta_R[it]))
            f.write( '%+.9e ' % (theta_R_dt[it]))
                    
            # unused, reserved for hindwings
            for k in range(11):    
                f.write( '%+.9e ' % (0.0))
            f.write( '%+.9e' % (0.0)) # no space lastline
            
            # new line
            if it != alpha_L.shape[0]-1:
                f.write('\n')
                
        f.close()
        
        #%% Output: particles for paraview
        # -------------------------------------------------------------------------
        # for paraview
        # -------------------------------------------------------------------------
        # time vector in dry-run mode 
        time_dryrun =  np.round( np.linspace(0, 15.95, 320, endpoint=True), decimals=5)
        # add complete time vector (required for interpolation)
        df_org["t"] = time
        # interpolate to dry-run time vector
        df_dry = pd.DataFrame({"t": time_dryrun})
        for c in df.columns:
            if c == "t": 
                continue
            df_dry[c] = np.interp(time_dryrun, df_org["t"], df_org[c])
            
        if pin_body_output:
            for key in all_keys:
                if key != 'Torso':
                    df_dry[key+'.x'] -= df_dry['Torso.x']
                    df_dry[key+'.y'] -= df_dry['Torso.y']
                    df_dry[key+'.z'] -= df_dry['Torso.z']
                    
            key = 'Torso'
            df_dry[key+'.x'] *= 0
            df_dry[key+'.y'] *= 0
            df_dry[key+'.z'] *= 0    
        
        for key in all_keys:
            df_dry[key+'.x'] /= R_wing
            df_dry[key+'.y'] /= R_wing
            df_dry[key+'.z'] /= R_wing
            # origin of motion in middle of dmain
            df_dry[key+'.x'] += 16
            df_dry[key+'.y'] += 16
            df_dry[key+'.z'] += 16
        
        
        """
        from df_to_vtp_timeseries import df_to_vtp_timeseries
        df_to_vtp_timeseries(df_dry, output_dir=root+'particle_data/', keys=all_keys)
        """
        # -------------------------------------------------------------------------
        
        
        ax = fig1.add_subplot(2,3,3, projection='3d')
        
        # d = np.loadtxt(root+file+'.kineloader', skiprows=1, delimiter=' ')  
        
        plt.plot( df_org["Torso.x"]/R_wing, df_org["Torso.y"]/R_wing, df_org["Torso.z"]/R_wing, label='thorax trajectory')
        # plt.plot( d[:,2-1], d[:,4-1], d[:,6-1], label='thorax trajectory')
        ax.set_xlabel('x/R')
        ax.set_ylabel('y/R')
        ax.set_zlabel('z/R')
        plt.legend()
        plt.axis('equal')
        
    
    
    
        #%% Collision test 
        # -------------------------------------------------------------------------
        if collison_check:
            print("Starting collision test...")
            dx = 2.5e-2
            xc, yc = insect_tools.get_wing_membrane_grid( 'E:/PhD_Schweitzer/CFD_Thomas/for_francois/simulations_raw_data/2g-new/singlewing.ini', dx=dx, dy=dx)
            V = np.zeros( (xc.shape[0], 3) )
            V[:,0] = xc # mm 
            V[:,1] = yc # mm 
        
            collision = insect_tools.collision_test(time, V, alpha_L, theta_L, phi_L, np.asarray([0,+0.075,0]), V, alpha_R, theta_R, phi_R, np.asarray([0,-0.075,0]), hold_on_collision=False, verbose=False)
            
            plt.subplot(2,3,4)
            plt.plot( time, collision, 'o')
            plt.title('wing/wing collisions')
            insect_tools.indicate_strokes(tstroke=2.0)
    
    
        fig1.set_size_inches((20,15))
        fig1.savefig( root+file+'_11.pdf')
    
    
    #%% Part3: plotting
    plot3d  = False
    if plot3d:
        d = np.loadtxt(file_kineloader, skiprows=1)
        time = d[:,0]
        psi, beta, gamma = d[:,13], d[:,15], d[:,17]
        phi_R, alpha_R, theta_R =  d[:,27], d[:,25], d[:,29]
        phi_L, alpha_L, theta_L =  d[:,21], d[:,19], d[:,23]
          
        fig = plt.figure()
        ax = plt.gcf().add_subplot(1,1,1, projection='3d')
        fig.set_size_inches([25, 25])
    
        xcontour, ycontour, area = insect_tools.wing_contour_from_file( 'E:/PhD_Schweitzer/CFD_Thomas/for_francois/simulations_raw_data/1g/singlewing.ini', N=200)
        contour_w = np.zeros( (xcontour.shape[0], 3) )
        contour_w[:,0] = xcontour*R_wing # mm 
        contour_w[:,1] = ycontour*R_wing # mm 
        
        # undo the projection done before
        # TODO
        df = df_org.copy()
    
        box_size = 1.1* ( max([np.max(np.abs(df['Torso.x'])), np.max(np.abs(df['Torso.y'])), np.max(np.abs(df['Torso.z'])) ])  )
        
        for it in range(istart, iend, step): 
            if it % 100 == 0:
                print( "pass 3: %i%%" % (100*it/(iend-istart+1)) )
                
            ax.cla()
            ax.set_xlim([-box_size/2, box_size/2])
            ax.set_ylim([-box_size/2, box_size/2])
            ax.set_zlim([-box_size/2, box_size/2])
            # ax.set_xlim([-box_size/2 +np.mean(df['Torso.x']), box_size/2+np.mean(df['Torso.x'])])
            # ax.set_ylim([-box_size/2 +np.mean(df['Torso.y']), box_size/2+np.mean(df['Torso.y'])])
            # ax.set_zlim([-box_size/2 +np.mean(df['Torso.z']), box_size/2+np.mean(df['Torso.z'])])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title("frame=%i  t=%f"  % (df['frame'][it], time[it]))   
            
            # trajectory
            plt.plot( df['Torso.x'], df['Torso.y'], df['Torso.z'], 'k:')
            # shade of trajectory
            plt.plot( df['Torso.x'], df['Torso.y'], df['Torso.z']*0.0+ax.get_zlim()[0], 'k:', alpha=0.25)
            plt.plot( df['Torso.x'], df['Torso.y']*0.0+ax.get_ylim()[1], df['Torso.z'], 'k:', alpha=0.25)
                
            # computation of body system
            # now psi is filtered
            M_g2b = insect_tools.get_M_g2b(psi[it], beta[it], gamma[it])
            # extract basis vectors from rotation matrix ( for plotting them )
            ex_body_g = M_g2b[0,:]
            ey_body_g = M_g2b[1,:] 
            ez_body_g = M_g2b[2,:]
                
            M_b2w_L = insect_tools.get_M_b2w(alpha_L[it], theta_L[it], phi_L[it], 0.0, 'left') ##############
            M_b2w_R = insect_tools.get_M_b2w(alpha_R[it], theta_R[it], phi_R[it], 0.0, 'right')
            
            # assumes hinge position to be [0,0,0] in body system
            contour_L_b = (M_b2w_L.T @ contour_w.T).T
            contour_L_g = (M_g2b.T @ contour_L_b.T).T
            
            contour_R_b = (M_b2w_R.T @ contour_w.T).T
            contour_R_g = (M_g2b.T @ contour_R_b.T).T
            
            # add body translation
            for k, xyz in enumerate(['x','y','z']):
                contour_L_g[:,k] += df['Torso.'+xyz][it]
                contour_R_g[:,k] += df['Torso.'+xyz][it]
            
            # draw the wings
            ax.add_collection3d(Poly3DCollection(verts=[contour_L_g], color='b', edgecolor='k', alpha=0.5))
            ax.add_collection3d(Poly3DCollection(verts=[contour_R_g], color='r', edgecolor='k', alpha=0.5))
            
            # draw wing shades
            for contour_g, color in zip([contour_L_g, contour_R_g],['b','r']):
                # shade (floor)
                shade_g = contour_g.copy()
                shade_g[:,2] = ax.get_zlim()[0]
                ax.add_collection3d(Poly3DCollection(verts=[shade_g], color=color, edgecolor=gray, alpha=0.2))
                
                # shade (side)
                shade_g = contour_g.copy()
                shade_g[:,1] = ax.get_ylim()[1]
                ax.add_collection3d(Poly3DCollection(verts=[shade_g], color=color, edgecolor=gray, alpha=0.2))
                
                # shade (side)
                shade_g = contour_g.copy()
                shade_g[:,0] = ax.get_xlim()[0]
                ax.add_collection3d(Poly3DCollection(verts=[shade_g], color=color, edgecolor=gray, alpha=0.2))
                    
            # draw body parts later so they cover the wings - easier to see then                
            # body:
            names = ['Tail', 'Torso', 'Head']
            for j in names:
                # body
                plt.plot( df[j+'.x'][it],  df[j+'.y'][it],  df[j+'.z'][it], 'k*'  ) 
                # shade (bottom)
                plt.plot( df[j+'.x'][it],  df[j+'.y'][it],  ax.get_zlim()[0], '*', color=gray  ) 
                # shade (side)
                plt.plot( df[j+'.x'][it],  ax.get_ylim()[1],  df[j+'.z'][it], '*', color=gray  ) 
                
            names = ['LeftAntenna', 'RightAntenna']
            for j in names:
                # body
                plt.plot( df[j+'.x'][it],  df[j+'.y'][it],  df[j+'.z'][it], 'ko'  ) 
                # shade (bottom)
                plt.plot( df[j+'.x'][it],  df[j+'.y'][it],  ax.get_zlim()[0], 'o', color=gray  ) 
                # shade (side)
                plt.plot( df[j+'.x'][it],  ax.get_ylim()[1],  df[j+'.z'][it], 'o', color=gray  ) 
                
            # skeleton
            plot_line('Tail', 'Torso')
            plot_line('Head', 'Torso')
            plot_line('Head', 'LeftAntenna')
            plot_line('Head', 'RightAntenna')
            
            if plot_markers:
                for wing, color in zip(['Left','Right'],['b','r']):
                    names = [ 'Line1Top', 'Line1Bottom', 'Line2Top', 'Line2Bottom', 'Line3Middle', 'Line3Bottom', 'Line4Bottom', 'Line5Far', 'Line5Close', 'Line4Bottom', 'Line5Far', 'Line5Close']
                    for j in names:
                        plt.plot( df[wing+j+'.x'][it],  df[wing+j+'.y'][it],  df[wing+j+'.z'][it], color+'o', mfc='w'  )         
                        # shades:
                        plt.plot( df[wing+j+'.x'][it],  df[wing+j+'.y'][it],  ax.get_zlim()[0], color+'o', mfc='w', alpha=0.25  )   
                        plt.plot( df[wing+j+'.x'][it],  ax.get_ylim()[1],  df[wing+j+'.z'][it], color+'o', mfc='w', alpha=0.25  )   
                        plt.plot( ax.get_xlim()[0],  df[wing+j+'.y'][it],  df[wing+j+'.z'][it], color+'o', mfc='w', alpha=0.25  )   
                
            # plot body coordinate system vectors
            j, scale = 'Torso',  50.0
            x0, y0, z0 = df[j+'.x'][it],  df[j+'.y'][it],  df[j+'.z'][it] 
            plt.plot( [x0, x0+ex_body_g[0]*scale], [y0, y0+ex_body_g[1]*scale], [z0, z0+ex_body_g[2]*scale], 'r--', label="ex"  ) 
            plt.plot( [x0, x0+ey_body_g[0]*scale], [y0, y0+ey_body_g[1]*scale], [z0, z0+ey_body_g[2]*scale], 'g--', label="ey"  ) 
            plt.plot( [x0, x0+ez_body_g[0]*scale], [y0, y0+ez_body_g[1]*scale], [z0, z0+ez_body_g[2]*scale], 'b--', label="ez"  ) 
    
            # shades        
            plt.plot( [x0, x0+ex_body_g[0]*scale], [y0, y0+ex_body_g[1]*scale], [ax.get_zlim()[0], ax.get_zlim()[0]], 'r--'  ) 
            plt.plot( [x0, x0+ey_body_g[0]*scale], [y0, y0+ey_body_g[1]*scale], [ax.get_zlim()[0], ax.get_zlim()[0]], 'g--'  ) 
            plt.plot( [x0, x0+ez_body_g[0]*scale], [y0, y0+ez_body_g[1]*scale], [ax.get_zlim()[0], ax.get_zlim()[0]], 'b--'  ) 
            
            plt.plot( [x0, x0+ex_body_g[0]*scale], 2*[ax.get_ylim()[1]], [z0, z0+ex_body_g[2]*scale], 'r--'  ) 
            plt.plot( [x0, x0+ey_body_g[0]*scale], 2*[ax.get_ylim()[1]], [z0, z0+ey_body_g[2]*scale], 'g--'  ) 
            plt.plot( [x0, x0+ez_body_g[0]*scale], 2*[ax.get_ylim()[1]], [z0, z0+ez_body_g[2]*scale], 'b--'  ) 
            
            plt.plot( 2*[ax.get_xlim()[0]], [y0, y0+ex_body_g[1]*scale], [z0, z0+ex_body_g[2]*scale], 'r--', label="ex"  ) 
            plt.plot( 2*[ax.get_xlim()[0]], [y0, y0+ey_body_g[1]*scale], [z0, z0+ey_body_g[2]*scale], 'g--', label="ey"  ) 
            plt.plot( 2*[ax.get_xlim()[0]], [y0, y0+ez_body_g[1]*scale], [z0, z0+ez_body_g[2]*scale], 'b--', label="ez"  ) 
                
            plt.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()        
            fig.set_size_inches( (10, 10))        
            plt.savefig(root+file+'frame.singlewing.%04i.png' % (it))
    
