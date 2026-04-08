#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:11:38 2025

@author: engels
"""
import sys
import numpy as np
from scipy.spatial import Delaunay
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
from scipy.signal import savgol_filter

plt.ion()
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


#%%


def best_rotation_only(A, B):
    """
    Best rotation only mapping A -> B in least squares, without translation.
    Returns R such that:
        B ~ A @ R.T
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("A and B must have shape (N,3)")
    if A.shape[0] < 1:
        raise ValueError("Need at least one correspondence")

    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def rotation_align_no_translation(V, ii, bc):
    """
    Align full mesh V to measured markers bc using rotation only.
    """
    V = np.asarray(V, dtype=np.float64)
    src = V[ii]
    tgt = np.asarray(bc, dtype=np.float64)

    R = best_rotation_only(src, tgt)
    V_aligned = V @ R.T
    return V_aligned, R

def best_rotation_translation(A, B):
    """
    Best rigid transform mapping A -> B in least squares.
    Returns R, t such that:
        B ~ A @ R.T + t
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("A and B must have shape (N,3)")
    if A.shape[0] < 1:
        raise ValueError("Need at least one correspondence")

    cA = A.mean(axis=0)
    cB = B.mean(axis=0)

    Ac = A - cA
    Bc = B - cB

    H = Ac.T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cB - cA @ R.T
    return R, t


def rotation_align(V, ii, bc):
    """
    Align full mesh V to measured markers bc using rigid transform.
    """
    V = np.asarray(V, dtype=np.float64)
    src = V[ii]
    tgt = np.asarray(bc, dtype=np.float64)

    R, t = best_rotation_translation(src, tgt)
    V_aligned = V @ R.T + t

    return V_aligned, (R, t)



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



# ---------------------------------------------------------
# Helper: normalize safely
# ---------------------------------------------------------
def safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n

# ---------------------------------------------------------
# Helper: enforce sign continuity relative to previous vector
# (avoid +/- flips)
# ---------------------------------------------------------
def enforce_continuity(v, v_prev):
    if v_prev is None:
        return v
    if np.dot(v, v_prev) < 0:
        return -v
    return v

# ---------------------------------------------------------
# Your symmetry plane normal (as you already have)
# ---------------------------------------------------------
def symmetry_plane_normal(x_left, x_right):
    x_right = np.asarray(x_right)
    x_left  = np.asarray(x_left)
    assert x_right.shape == x_left.shape and x_right.shape[1] == 3

    diffs = x_left - x_right  # vectors from right to left

    # SVD principal direction of diffs
    _, _, vh = np.linalg.svd(diffs, full_matrices=False)
    normal = vh[0]
    normal = normal / np.linalg.norm(normal)

    # orient toward left on average
    mean_diff = np.mean(diffs, axis=0)
    if np.dot(normal, mean_diff) < 0:
        normal = -normal

    plane_point = np.mean(0.5*(x_left + x_right), axis=0)
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

# ---------------------------------------------------------
# Extract Euler angles from basis (you already have yours)
# Here we assume you have:
#   psi[it], beta[it], gamma[it] = euler_angles_from_body_basis(ex,ey,ez)
# ---------------------------------------------------------


# =========================================================
# PASS 1: Compute RAW body basis per frame (NO smoothing here)
# =========================================================
def compute_body_basis_raw(df, wing_keys, istart=0, iend=None, verbose_every=500):
    if iend is None:
        iend = len(df)

    nt = iend
    ex_all = np.full((nt, 3), np.nan)
    ey_all = np.full((nt, 3), np.nan)
    ez_all = np.full((nt, 3), np.nan)

    ex_prev = None
    ey_prev = None

    for it in range(istart, iend):
        if verbose_every and it % verbose_every == 0:
            print(f"pass 1 raw basis: {100*it/max(1,(iend-istart)):.1f}%")

        # -------------------------
        # ex = Head-Torso (robust, no tail)
        # -------------------------
        """
        p = np.asarray( [[df['Torso.x'][it],df['Torso.y'][it],df['Torso.z'][it]],
                         [df['Head.x'][it],df['Head.y'][it],df['Head.z'][it]],
                         [df['Tail.x'][it],df['Tail.y'][it],df['Tail.z'][it]]] )
        

        
        p -= np.mean(p, axis=0)
        # SVD yields the vector
        _, _, Vt = np.linalg.svd(p)
        # normalization and sign determination
        ex = Vt[0] / np.linalg.norm(Vt[0])
        if np.dot( ex, p[1] ) < 0: # points towards the head
            ex *= -1.0
            
        """
        head = np.array([df['Head.x'][it],  df['Head.y'][it],  df['Head.z'][it]])
        torso= np.array([df['Torso.x'][it], df['Torso.y'][it], df['Torso.z'][it]])
        ex = head - torso
        
        ex = safe_normalize(ex)
        if ex is None:
            # cannot define -> keep NaN
            continue
        
        # continuity on ex
        # ex = enforce_continuity(ex, ex_prev)
        ex_prev = ex.copy()

        # -------------------------
        # ey_raw from wing symmetry plane normal
        # -------------------------
        x_right = np.array([[df['Right'+w+'.x'][it], df['Right'+w+'.y'][it], df['Right'+w+'.z'][it]]
                            for w in wing_keys])
        x_left  = np.array([[df['Left'+w+'.x'][it],  df['Left'+w+'.y'][it],  df['Left'+w+'.z'][it]]
                            for w in wing_keys])

        ey_raw, _ = symmetry_plane_normal(x_left, x_right)

        # project ey to be orthogonal to ex
        ey = ey_raw - np.dot(ey_raw, ex) * ex
        ey = safe_normalize(ey)

        # fallback if degenerate (e.g. bad wings)
        if ey is None:
            if ey_prev is None:
                # pick any vector not parallel to ex
                tmp = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(tmp, ex)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0])
                ey = tmp - np.dot(tmp, ex)*ex
                ey = safe_normalize(ey)
            else:
                ey = ey_prev.copy()

        # continuity on ey
        # ey = enforce_continuity(ey, ey_prev)
        ey_prev = ey.copy()

        # -------------------------
        # ez
        # -------------------------
        ez = np.cross(ex, ey)
        ez = safe_normalize(ez)
        if ez is None:
            continue

        # re-orthonormalize ey (to kill drift)
        ey = np.cross(ez, ex)
        ey = safe_normalize(ey)
        if ey is None:
            continue

        ex_all[it] = ex
        ey_all[it] = ey
        ez_all[it] = ez

    return ex_all, ey_all, ez_all


# =========================================================
# PASS 2: Offline smoothing of basis (vector-wise) + renormalization
# =========================================================
def smooth_body_basis_offline(ex_all, ey_all, window=1001, poly=3):
    """
    Smooth ex and ey with Savitzky-Golay, then re-orthonormalize per frame.
    window must be odd.
    Choose window ~ a few wingbeats in frames.
    """

    nt = ex_all.shape[0]
    ex_f = ex_all.copy()
    ey_f = ey_all.copy()

    # handle NaNs: simple fill forward/backward so filter works
    def fill_nans(arr):
        out = arr.copy()
        for k in range(3):
            x = out[:, k]
            good = np.isfinite(x)
            if not np.any(good):
                continue
            idx = np.arange(len(x))
            out[:, k] = np.interp(idx, idx[good], x[good])
        return out

    ex_ff = fill_nans(ex_f)
    ey_ff = fill_nans(ey_f)

    # SavGol filtering (zero-phase, offline)
    if window % 2 == 0:
        window += 1
    window = min(window, nt - (1 - nt % 2))  # ensure <= nt and odd
    if window < 5:
        # too short to filter meaningfully
        window = 5 if nt >= 5 else nt

    ex_s = savgol_filter(ex_ff, window_length=window, polyorder=poly, axis=0, mode="interp")
    ey_s = savgol_filter(ey_ff, window_length=window, polyorder=poly, axis=0, mode="interp")

    # re-orthonormalize each frame
    ex_out = np.zeros_like(ex_s)
    ey_out = np.zeros_like(ey_s)
    ez_out = np.zeros_like(ey_s)

    ex_prev = None
    ey_prev = None

    for i in range(nt):
        ex = safe_normalize(ex_s[i])
        if ex is None:
            continue
        # ex = enforce_continuity(ex, ex_prev)
        ex_prev = ex.copy()

        ey = ey_s[i] - np.dot(ey_s[i], ex)*ex
        ey = safe_normalize(ey)
        if ey is None:
            # fallback: keep previous
            if ey_prev is None:
                tmp = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(tmp, ex)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0])
                ey = tmp - np.dot(tmp, ex)*ex
                ey = safe_normalize(ey)
            else:
                ey = ey_prev.copy()

        # ey = enforce_continuity(ey, ey_prev)
        ey_prev = ey.copy()

        ez = safe_normalize(np.cross(ex, ey))
        if ez is None:
            continue

        # final orthonormal correction
        ey = safe_normalize(np.cross(ez, ex))
        if ey is None:
            continue

        ex_out[i] = ex
        ey_out[i] = ey
        ez_out[i] = ez

    return ex_out, ey_out, ez_out

def estimate_T_seconds_from_phi(t0_sec, phi, fmin_hz=5.0, fmax_hz=40.0):
    """
    Estime T (s) à partir de phi(t0).
    - Interpole sur une grille uniforme (dt médian)
    - Nettoie grossièrement phi (unwrap + suppression d’outliers)
    - Cherche le pic spectral DANS [fmin_hz, fmax_hz]
    """
    t0_sec = np.asarray(t0_sec, dtype=float)
    phi = np.asarray(phi, dtype=float)

    mask = np.isfinite(t0_sec) & np.isfinite(phi)
    if mask.sum() < 50:
        return np.nan

    t = t0_sec[mask]
    p = phi[mask]
    idx = np.argsort(t)
    t = t[idx]
    p = p[idx]

    # unwrap (mais attention aux gros flips)
    p = np.unwrap(p)

    # dt uniforme ~ dt médian
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan

    tu = np.arange(t.min(), t.max(), dt)

    interp = interp1d(t, p, kind="linear", fill_value="extrapolate", bounds_error=False)
    pu = interp(tu)

    # retire moyenne + outliers simples (Hampel light)
    u = pu - np.mean(pu)
    u = hampel_filter(u, k=31, n_sigmas=3.0)  # réutilise ta fonction

    # FFT (via ta routine)
    k, ek = fourier_tools.spectrum1(u)

    # conversion en Hz :
    # fourier_tools.spectrum1 renvoie k en "indice de mode"
    # fréquence en cycles/sample = k_max / N
    N = len(u)
    f_cyc_per_sample = (k * 2.0 * np.pi / N) / (2.0 * np.pi)
    f_hz = f_cyc_per_sample / dt

    # ignore DC + restreint à la bande
    band = (f_hz >= fmin_hz) & (f_hz <= fmax_hz)
    if not np.any(band):
        return np.nan

    k_band = k[band]
    ek_band = ek[band]
    f_hz_band = f_hz[band]

    f0 = f_hz_band[np.argmax(ek_band)]
    if f0 <= 0:
        return np.nan

    return 1.0 / f0

def smooth_angle_series(y, window=151, poly=3, amplitude_rescale=True):
        """
        Smooth angle signal with Savitzky-Golay while preserving amplitude.
    
        Parameters
        ----------
        y : array-like
            Angle signal
        window : int
            Savitzky-Golay window length (must be odd)
        poly : int
            Polynomial order
        amplitude_rescale : bool
            If True, rescales amplitude to match raw signal
    
        Returns
        -------
        y_smooth : ndarray
            Smoothed signal
        """
    
        y = np.asarray(y, dtype=float).copy()
    
        mask = np.isfinite(y)
    
        if np.sum(mask) < window:
            return y
    
        # interpolation des NaN
        if not np.all(mask):
            x = np.arange(len(y))
            y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    
        # fenêtre impaire obligatoire
        if window % 2 == 0:
            window += 1
    
        # si fenêtre trop grande
        if window >= len(y):
            window = len(y) - 1
            if window % 2 == 0:
                window -= 1
    
        # =========================
        # SAVITZKY-GOLAY
        # =========================
        y_smooth = savgol_filter(
            y,
            window_length=window,
            polyorder=poly,
            mode="interp"
        )
    
        # =========================
        # AMPLITUDE RESCALE
        # =========================
        if amplitude_rescale:
    
            mu_raw = np.nanmean(y)
            mu_smooth = np.nanmean(y_smooth)
    
            p_raw = np.nanpercentile(y, [2, 98])
            p_smooth = np.nanpercentile(y_smooth, [2, 98])
    
            amp_raw = (p_raw[1] - p_raw[0]) / 2
            amp_smooth = (p_smooth[1] - p_smooth[0]) / 2
    
            if amp_smooth > 0 and np.isfinite(amp_smooth):
    
                scale = amp_raw / amp_smooth
    
                y_smooth = mu_raw + scale * (y_smooth - mu_smooth)
    
        return y_smooth

#%%

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
for DATA in ['0g']:
    
    # --- SETTINGS: give full path only
    csv_path = Path(r"E:\PhD_Schweitzer\CFD_Thomas\smoothing_kinematics\test\frameData_h__charithonia_02_10_2025_vol_3_S2_402_30_403_30_fs78_fe4921_smoothing11.mat.csv")
    
    root = str(csv_path.parent)            # Path object (folder)
    file = '/' + str(csv_path.name)         # filename only (string)

    """
    root = 'E:/PhD_Schweitzer/CFD_Thomas/test_lissage_body/'
    file = 'frameData_simple_h__charithonia_02_10_2025_vol_3_S6_151_20_152_50_fs82_fe6350.mat.csv'
    """
        
    d = np.loadtxt( root+file, skiprows=1, delimiter=',')
    df = pd.read_csv( root+file, delimiter=',')
    
    # lissage fort de Tail (sur données brutes)
    df = smooth_point_in_df(df,
                           tail_key='Tail',
                           hampel_k=31,
                           hampel_sig=3.0,
                           smooth_size=151) #smooth_size=251; hampel_k=31
    

    df = smooth_point_in_df(df,
                           tail_key='Torso',
                           hampel_k=31,
                           hampel_sig=3.0,
                           smooth_size=151)
    

    df = smooth_point_in_df(df,
                           tail_key='Head',
                           hampel_k=31,
                           hampel_sig=3.0,
                           smooth_size=151)
    
    # this can be merged into the matlab code, inlcuding the switch left/right renaming
    # Motion starts at 0,0,0
    for key in all_keys:
        # copying and translation            
        df[key+'.x'] = df[key+'.x'] - df.loc[0,'Torso.x']
        df[key+'.y'] = df[key+'.y'] - df.loc[0,'Torso.y']
        df[key+'.z'] = df[key+'.z'] - df.loc[0,'Torso.z']
    
    df_org = df.copy()
    df_body = df.copy()
    
    df_org = df.copy()
    df_body = df.copy()

    frames = df["frame"].to_numpy(dtype=float)
    frame_start = frames[0]
    nt = len(df)
    fps = 5000
    # --------------------------
    # TIMES YOU WANT
    # --------------------------
    # IMPORTANT: seconds are frame / fps (not frame * fps)
    t_raw = frames / fps
    t0 = (frames - frame_start) / fps
    
    nt = d.shape[0]
    Npoints = int( (d.shape[1]-1)/3 )
    
    plot3d  = False
    translate = False # used only if project==False
    plot_markers = True
    force_recompute = True
    collison_check = True
    pin_body_output = False
    
    istart, iend = 0, nt #500, 5200 #340, 5000
    # istart, iend = 1270,1275
    step = 10 # only for plotting
    # This is the forewing length, but the same number is used for the hindwing
    # (=normalization done by forewing)
    R_wing = 4.01*10 # mm take the right R_wing (I rescaled with factor s = R_ref/R_true) with R_ref = 3.76 cm
    
    file_kineloader = root+file+'_corrected_translation.kineloader'
    
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
    # =========================================================
    # USAGE inside your script
    # =========================================================
    # PASS 1
    ex_raw, ey_raw, ez_raw = compute_body_basis_raw(df, wing_keys, istart=istart, iend=iend)
    
    # Choose window in FRAMES ~ a few wingbeats
    # Example: if wingbeat ~ 400 frames, take 3-5 wingbeats => 1200-2000 (must be odd)
    window = 1501
    poly = 3
    
    # PASS 2
    ex_s, ey_s, ez_s = smooth_body_basis_offline(ex_raw, ey_raw, window=window, poly=poly)
    
    # Now compute Euler angles from smoothed basis
    psi = np.zeros(iend)
    beta = np.zeros(iend)
    gamma = np.zeros(iend)
    for it in range(istart, iend):
        ex = ex_s[it]; ey = ey_s[it]; ez = ez_s[it]
        if not np.all(np.isfinite(ex)) or not np.all(np.isfinite(ey)) or not np.all(np.isfinite(ez)):
            psi[it] = np.nan; beta[it] = np.nan; gamma[it] = np.nan
            continue
        psi[it], beta[it], gamma[it] = euler_angles_from_body_basis(ex, ey, ez)
    
    # optional: unwrap + gentle smoothing of angles (low)
    # psi = np.unwrap(psi); beta = np.unwrap(beta); gamma = np.unwrap(gamma)
    # psi = scipy.ndimage.uniform_filter1d(psi, size=51)
                
              
    # remove jumps by 2*pi
    psi1, beta1, gamma1 = np.unwrap(psi), np.unwrap(beta), np.unwrap(gamma)
    # psi1, beta1, gamma1 = psi, beta, gamma
    
        
    # apply the very agressive smoothing filter
    # we do this because from francois data, the roll angle cannot be estimated reliably
    fsize = 250
    psi = scipy.ndimage.uniform_filter1d(psi1, size=fsize) 
    beta = scipy.ndimage.uniform_filter1d(beta1, size=fsize) 
    gamma = scipy.ndimage.uniform_filter1d(gamma1, size=fsize) 
        
        
    #%% Part 2 projection and wings angles
    
    marker_names = [
    "Line3Middle",
    "Line3Bottom",
    "Line2Top",
    "Line2Bottom",
    "Line1Top",
    "Line1Bottom",
    "WingTip",
    "Line5Far",
    "Line4Bottom",
    "Line5Close",
    ]

    
    n_markers = len(marker_names)
    
    # residuals in same units as coordinates (likely mm)
    residuals_R = np.full((nt, n_markers), np.nan)
    residuals_L = np.full((nt, n_markers), np.nan)
    
    # optional normalized residuals
    residuals_R_rel = np.full((nt, n_markers), np.nan)
    residuals_L_rel = np.full((nt, n_markers), np.nan)

    alpha_L, theta_L, phi_L = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    alpha_R, theta_R, phi_R = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    err_L, err_R = np.zeros(nt), np.zeros(nt)
    opening_angle_L, opening_angle_R = np.zeros(nt), np.zeros(nt)

    alpha_L_not, theta_L_not, phi_L_not = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    alpha_R_not, theta_R_not, phi_R_not = np.zeros(nt), np.zeros(nt), np.zeros(nt)

    err_L_not, err_R_not = np.zeros(nt), np.zeros(nt)

    hinge_L_b = np.full((nt, 3), np.nan)
    hinge_R_b = np.full((nt, 3), np.nan)

    hinge_L_g = np.full((nt, 3), np.nan)
    hinge_R_g = np.full((nt, 3), np.nan)

    x_wing_b_store_L = []
    x_wing_b_store_R = []
    frames_store = []
    
        
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
    xch, ych = insect_tools.get_wing_membrane_grid( 'E:/PhD_Schweitzer/CFD_Thomas/for_francois/simulations_raw_data/2g-new/singlewing.ini', dx=dx, dy=dx)
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
        
            x_wing_b, (R, t) = rotation_align(x_wing_w, ii, marker_measured_b)
            x_wing_b_not, R_not = rotation_align_no_translation(x_wing_w, ii, marker_measured_b)
            hinge_b = t.copy()   # because model hinge is at [0,0,0] in wing coordinates
            if it % 100 == 0:
                frames_store.append(it)
                if side == 'Right':
                    x_wing_b_store_R.append(x_wing_b.copy())
                else:
                    x_wing_b_store_L.append(x_wing_b.copy())

            #x_wing_b, (R, t) = rotation_align(x_wing_w, ii, marker_measured_b)
            # pointwise residuals between rigidly aligned model markers and measured markers
            residuals_pts = np.linalg.norm(x_wing_b[ii] - marker_measured_b, axis=1)   # shape (10,)
            residuals_pts_rel = residuals_pts / R_wing
            
            residuals_pts_not = np.linalg.norm(x_wing_b_not[ii] - marker_measured_b, axis=1)
            
            
            # determine angles assuming fore-and hindwing coupled in reference configuration
            if side == 'Right':
                alpha_R[it], theta_R[it], phi_R[it] = euler_from_wing_basis(R[:,0], R[:,1], R[:,2], 'right')
                residuals_R[it, :] = residuals_pts
                residuals_R_rel[it, :] = residuals_pts_rel
                err_R[it] = np.mean( np.linalg.norm(x_wing_b[ii]-marker_measured_b, axis=1) ) / R_wing

                alpha_R_not[it], theta_R_not[it], phi_R_not[it] = euler_from_wing_basis(R_not[:,0], R_not[:,1], R_not[:,2], 'right')

                err_R_not[it] = np.mean(np.linalg.norm(x_wing_b_not[ii] - marker_measured_b, axis=1)) / R_wing
                
                marker_HW_b = x_HW_w[ij].copy()
                marker_HW_b[0, 0:2+1] = np.array( [df_body[side+'Line5Far.x'][it]   , df_body[side+'Line5Far.y'][it]   , df_body[side+'Line5Far.z'][it]] )
                marker_HW_b[1, 0:2+1] = np.array( [df_body[side+'Line4Bottom.x'][it], df_body[side+'Line4Bottom.y'][it], df_body[side+'Line4Bottom.z'][it]] )
                marker_HW_b[2, 0:2+1] = np.array( [df_body[side+'Line5Close.x'][it] , df_body[side+'Line5Close.y'][it] , df_body[side+'Line5Close.z'][it]] )
                
                num = 50
                distance = np.zeros((num))
                thetas = np.linspace(-20, 20, num)
                for t, theta_opening in enumerate(thetas): # deg
                    theta_opening = theta_opening / rad2deg # rad
                    
                    M_b2w_R = insect_tools.get_M_b2w(alpha_R[it], theta_R[it], phi_R[it], 0.0, 'right') 
                    
                    # assumes hinge position to be [0,0,0] in body system
                    x_HW_b = ( (insect_tools.Rz(theta_opening) @ M_b2w_R.T) @ x_HW_w.T).T                    
                    
                    distance[t] = norm(x_HW_b[ij[0]] - marker_HW_b[0]) + norm(x_HW_b[ij[1]] - marker_HW_b[1]) + norm(x_HW_b[ij[2]] - marker_HW_b[2]) 

                opening_angle_R[it] = thetas[np.argmin(distance)] # deg

                hinge_R_b[it, :] = hinge_b

                # body -> global
                torso_g = np.array([
                    df_org['Torso.x'][it],
                    df_org['Torso.y'][it],
                    df_org['Torso.z'][it]
                ])
                hinge_R_g[it, :] = M_b2g.T @ hinge_b + torso_g

                
            else:
                alpha_L[it], theta_L[it], phi_L[it] = euler_from_wing_basis(R[:,0], R[:,1], R[:,2], 'left')
                residuals_L[it, :] = residuals_pts
                residuals_L_rel[it, :] = residuals_pts_rel
                err_L[it] = np.mean( np.linalg.norm(x_wing_b[ii]-marker_measured_b, axis=1) ) / R_wing
                
                alpha_L_not[it], theta_L_not[it], phi_L_not[it] = euler_from_wing_basis(R_not[:,0], R_not[:,1], R_not[:,2], 'left')

                err_L_not[it] = np.mean(np.linalg.norm(x_wing_b_not[ii] - marker_measured_b, axis=1)) / R_wing
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

                hinge_L_b[it, :] = hinge_b

                torso_g = np.array([
                    df_org['Torso.x'][it],
                    df_org['Torso.y'][it],
                    df_org['Torso.z'][it]
                ])
                hinge_L_g[it, :] = M_b2g.T @ hinge_b + torso_g
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

    
    phi_L_1, alpha_L_1, theta_L_1 =  phi_L, alpha_L, theta_L
    phi_R_1, alpha_R_1, theta_R_1 = phi_R, alpha_R, theta_R
    
    phi_L, alpha_L, theta_L = np.unwrap(phi_L_1), np.unwrap(alpha_L_1), np.unwrap(theta_L_1)
    phi_R, alpha_R, theta_R = np.unwrap(phi_R_1), np.unwrap(alpha_R_1), np.unwrap(theta_R_1)
    
    phi_L_not, alpha_L_not, theta_L_not = np.unwrap(phi_L_not), np.unwrap(alpha_L_not), np.unwrap(theta_L_not)
    phi_R_not, alpha_R_not, theta_R_not = np.unwrap(phi_R_not), np.unwrap(alpha_R_not), np.unwrap(theta_R_not)
    
    windowsize = 151
    
    phi_L_s   = smooth_angle_series(phi_L,   window=windowsize, poly=3)
    alpha_L_s = smooth_angle_series(alpha_L, window=windowsize, poly=3)
    theta_L_s = smooth_angle_series(theta_L, window=windowsize, poly=3)
    
    phi_R_s   = smooth_angle_series(phi_R,   window=windowsize, poly=3)
    alpha_R_s = smooth_angle_series(alpha_R, window=windowsize, poly=3)
    theta_R_s = smooth_angle_series(theta_R, window=windowsize, poly=3)

    phi_L_not_s   = smooth_angle_series(phi_L_not,   window=windowsize, poly=3)
    alpha_L_not_s = smooth_angle_series(alpha_L_not, window=windowsize, poly=3)
    theta_L_not_s = smooth_angle_series(theta_L_not, window=windowsize, poly=3)

    phi_R_not_s   = smooth_angle_series(phi_R_not,   window=windowsize, poly=3)
    alpha_R_not_s = smooth_angle_series(alpha_R_not, window=windowsize, poly=3)
    theta_R_not_s = smooth_angle_series(theta_R_not, window=windowsize, poly=3)
    # all vectors same length
    nk = nt
      
    # compute time normalization using dominant frequency in phi
    u = phi_L_s - np.mean(phi_L_s)
    k, ek = fourier_tools.spectrum1(u)

    f = (k[ np.argmax(ek) ]*2.0*np.pi/nt) / (2.0*np.pi) # frequency in 1/frames
    T = 1.0 / f # cycle time in frames
    # time vector
    time = np.arange(istart, iend) / T


    
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
    plt.plot(time,  phi_L_s*rad2deg, label='phi L')
    plt.plot(time,  alpha_L_s*rad2deg, label='alpha L')
    plt.plot(time,  theta_L_s*rad2deg, label='theta L')
    insect_tools.reset_colorcycle(plt.gca())
    plt.plot(time,  phi_R_s*rad2deg, label='phi R')
    plt.plot(time,  alpha_R_s*rad2deg, label='alpha R')
    plt.plot(time,  theta_R_s*rad2deg, label='theta R')
    # plt.ylim(-60, 60)
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
    
    phi_R_dt   = D @ phi_R_s
    alpha_R_dt = D @ alpha_R_s
    theta_R_dt = D @ theta_R_s
    
    phi_L_dt   = D @ phi_L_s
    alpha_L_dt = D @ alpha_L_s
    theta_L_dt = D @ theta_L_s
    
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
        
        f.write( '%+.9e ' % (alpha_L_s[it]))
        f.write( '%+.9e ' % (alpha_L_dt[it]))
                
        f.write( '%+.9e ' % (phi_L_s[it])) # newcode: radians!
        f.write( '%+.9e ' % (phi_L_dt[it]))
        
        f.write( '%+.9e ' % (theta_L_s[it]))
        f.write( '%+.9e ' % (theta_L_dt[it]))                 

        f.write( '%+.9e ' % (alpha_R_s[it]))
        f.write( '%+.9e ' % (alpha_R_dt[it]))
                
        f.write( '%+.9e ' % (phi_R_s[it])) # newcode: radians!
        f.write( '%+.9e ' % (phi_R_dt[it]))
        
        f.write( '%+.9e ' % (theta_R_s[it]))
        f.write( '%+.9e ' % (theta_R_dt[it]))
                
        # unused, reserved for hindwings
        for k in range(11):    
            f.write( '%+.9e ' % (0.0))
        f.write( '%+.9e' % (0.0)) # no space lastline
        
        # new line
        if it != alpha_L.shape[0]-1:
            f.write('\n')
            
    f.close()
    
    # --------------------------
    # Save angles + times
    # --------------------------
    T_sec = estimate_T_seconds_from_phi(t0, phi_L_s)   # your function
    if not np.isfinite(T_sec):
        T_sec = estimate_T_seconds_from_phi(t0, phi_R_s)
    
    if not np.isfinite(T_sec) or T_sec <= 0:
        print("WARNING: could not estimate T from phi. Using T=1s.")
        T_sec = 1.0
    
    time1 = t0 / T_sec
    out_df = pd.DataFrame({
        "frame": frames.astype(int),
        "t_raw": t_raw,     # seconds (absolute)
        "t0": t0,           # seconds shifted to start at 0
        "time": time1,       # t0 / T
        "T_sec": np.full(nt, T_sec),

        "psi": psi, "beta": beta, "gamma": gamma,
        "phi_L": phi_L_s, "alpha_L": alpha_L_s, "theta_L": theta_L_s,
        "phi_R": phi_R_s, "alpha_R": alpha_R_s, "theta_R": theta_R_s,
        "err_L": err_L, "err_R": err_R,
    })
    
    # add pointwise residual columns
    for j, name in enumerate(marker_names):
        out_df[f"res_L_{name}"] = residuals_L[:, j]
        out_df[f"res_R_{name}"] = residuals_R[:, j]
    
        out_df[f"res_L_{name}_rel"] = residuals_L_rel[:, j]
        out_df[f"res_R_{name}_rel"] = residuals_R_rel[:, j]

    #out_path = csv_path + ".angles.csv"
    out_path = csv_path.with_suffix(csv_path.suffix + "_corrected_translation.angles.csv")
    out_df.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    
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
    plt.plot(hinge_L_g[:,0]/R_wing, hinge_L_g[:,1]/R_wing, hinge_L_g[:,2]/R_wing,
         label='left hinge trajectory')

    plt.plot(hinge_R_g[:,0]/R_wing, hinge_R_g[:,1]/R_wing, hinge_R_g[:,2]/R_wing,
            label='right hinge trajectory')
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
    
        collision = insect_tools.collision_test(time, V, alpha_L_s, theta_L_s, phi_L_s, np.asarray([0,+0.075,0]), V, alpha_R_s, theta_R_s, phi_R_s, np.asarray([0,-0.075,0]), hold_on_collision=False, verbose=True)
        
        plt.subplot(2,3,4)
        plt.plot( time, collision, 'o')
        plt.title('wing/wing collisions')
        insect_tools.indicate_strokes(tstroke=2.0)


    fig1.set_size_inches((20,15))
    fig1.savefig( root+file+'_corrected_translation_2.pdf')
    plt.show()

    #%% Plot hinges in body frame 

    def smooth_series_nan_safe(y, window=151, poly=3):
        y = np.asarray(y, dtype=float).copy()
        mask = np.isfinite(y)
        if np.sum(mask) < max(poly + 2, 5):
            return y

        x = np.arange(len(y))
        if not np.all(mask):
            y[~mask] = np.interp(x[~mask], x[mask], y[mask])

        if window % 2 == 0:
            window += 1
        if window >= len(y):
            window = len(y) - 1
            if window % 2 == 0:
                window -= 1
        if window < poly + 2:
            return y

        return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


    def amplitude_pp(y):
        y = np.asarray(y, dtype=float)
        if np.sum(np.isfinite(y)) < 2:
            return np.nan
        return np.nanmax(y) - np.nanmin(y)


    def centered_rms(y):
        y = np.asarray(y, dtype=float)
        mu = np.nanmean(y)
        return np.sqrt(np.nanmean((y - mu)**2))


    def dominant_freq_hz_from_signal(t_sec, y, fmin_hz=5.0, fmax_hz=40.0):
        t_sec = np.asarray(t_sec, dtype=float)
        y = np.asarray(y, dtype=float)

        mask = np.isfinite(t_sec) & np.isfinite(y)
        if mask.sum() < 50:
            return np.nan, None, None

        t = t_sec[mask]
        s = y[mask]
        idx = np.argsort(t)
        t = t[idx]
        s = s[idx]

        dt = np.median(np.diff(t))
        if not np.isfinite(dt) or dt <= 0:
            return np.nan, None, None

        tu = np.arange(t.min(), t.max(), dt)
        fu = interp1d(t, s, kind='linear', fill_value='extrapolate', bounds_error=False)
        su = fu(tu)

        su = su - np.mean(su)

        k, ek = fourier_tools.spectrum1(su)
        N = len(su)
        f_cyc_per_sample = (k * 2.0 * np.pi / N) / (2.0 * np.pi)
        f_hz = f_cyc_per_sample / dt

        band = (f_hz >= fmin_hz) & (f_hz <= fmax_hz)
        if not np.any(band):
            return np.nan, f_hz, ek

        f_band = f_hz[band]
        ek_band = ek[band]
        f0 = f_band[np.argmax(ek_band)]
        return f0, f_hz, ek


    # ---------- filtered hinges ----------
    hinge_window = 151   
    hinge_poly = 3

    hinge_L_b_f = np.full_like(hinge_L_b, np.nan)
    hinge_R_b_f = np.full_like(hinge_R_b, np.nan)

    for j in range(3):
        hinge_L_b_f[:, j] = smooth_series_nan_safe(hinge_L_b[:, j], window=hinge_window, poly=hinge_poly)
        hinge_R_b_f[:, j] = smooth_series_nan_safe(hinge_R_b[:, j], window=hinge_window, poly=hinge_poly)

    hinge_L_b_mean = np.nanmean(hinge_L_b, axis=0)
    hinge_R_b_mean = np.nanmean(hinge_R_b, axis=0)

    hinge_L_b_const = np.tile(hinge_L_b_mean, (nt, 1))
    hinge_R_b_const = np.tile(hinge_R_b_mean, (nt, 1))


    # ---------- print quantitative summary ----------
    comp_names = ['x', 'y', 'z']
    print("\n================ HINGE DIAGNOSTIC (BODY FRAME, mm) ================\n")

    for side_name, H in [('Left', hinge_L_b), ('Right', hinge_R_b)]:
        print(f"--- {side_name} hinge ---")
        mu = np.nanmean(H, axis=0)
        rms = np.array([centered_rms(H[:, j]) for j in range(3)])
        pp  = np.array([amplitude_pp(H[:, j]) for j in range(3)])

        for j, cname in enumerate(comp_names):
            print(f"{cname}: mean = {mu[j]: .4f} mm, centered RMS = {rms[j]: .4f} mm, peak-to-peak = {pp[j]: .4f} mm, "
                f"mean/R = {mu[j]/R_wing: .5f}, rms/R = {rms[j]/R_wing: .5f}")
        print("")


    # frequency estimate from phi
    f_phi_L, _, _ = dominant_freq_hz_from_signal(t0, phi_L_s)
    f_phi_R, _, _ = dominant_freq_hz_from_signal(t0, phi_R_s)

    print(f"Dominant wingbeat frequency from phi_L: {f_phi_L:.3f} Hz")
    print(f"Dominant wingbeat frequency from phi_R: {f_phi_R:.3f} Hz\n")

    for side_name, H in [('Left', hinge_L_b), ('Right', hinge_R_b)]:
        print(f"--- {side_name} hinge dominant frequencies ---")
        for j, cname in enumerate(comp_names):
            f0, _, _ = dominant_freq_hz_from_signal(t0, H[:, j])
            print(f"{cname}: dominant freq = {f0:.3f} Hz")
        print("")


    # ---------- plots ----------
    fig_hinge = plt.figure(figsize=(18, 14))

    # 1) Left hinge x,y,z
    for j, cname in enumerate(comp_names):
        ax = fig_hinge.add_subplot(4, 3, j + 1)
        ax.plot(t0, hinge_L_b[:, j], label=f'L {cname} raw')
        ax.plot(t0, hinge_L_b_f[:, j], '--', label=f'L {cname} filtered')
        ax.axhline(hinge_L_b_mean[j], linestyle=':', color='k', label='mean')
        ax.set_title(f'Left hinge {cname}(t) [mm]')
        ax.set_xlabel('t0 [s]')
        ax.set_ylabel(f'{cname} [mm]')
        ax.grid(True)
        ax.legend()

    # 2) Right hinge x,y,z
    for j, cname in enumerate(comp_names):
        ax = fig_hinge.add_subplot(4, 3, j + 4)
        ax.plot(t0, hinge_R_b[:, j], label=f'R {cname} raw')
        ax.plot(t0, hinge_R_b_f[:, j], '--', label=f'R {cname} filtered')
        ax.axhline(hinge_R_b_mean[j], linestyle=':', color='k', label='mean')
        ax.set_title(f'Right hinge {cname}(t) [mm]')
        ax.set_xlabel('t0 [s]')
        ax.set_ylabel(f'{cname} [mm]')
        ax.grid(True)
        ax.legend()

    # 3) Left vs Right comparison
    for j, cname in enumerate(comp_names):
        ax = fig_hinge.add_subplot(4, 3, j + 7)
        ax.plot(t0, hinge_L_b[:, j], label=f'L {cname}')
        ax.plot(t0, hinge_R_b[:, j], label=f'R {cname}')
        ax.axhline(0.0, linestyle=':', color='k')
        ax.set_title(f'Left vs Right hinge {cname}(t) [mm]')
        ax.set_xlabel('t0 [s]')
        ax.set_ylabel(f'{cname} [mm]')
        ax.grid(True)
        ax.legend()

    # 4) difference / sum diagnostics
    ax = fig_hinge.add_subplot(4, 3, 10)
    ax.plot(t0, hinge_L_b[:,1] - hinge_R_b[:,1], label='L_y - R_y')
    ax.axhline(np.nanmean(hinge_L_b[:,1] - hinge_R_b[:,1]), linestyle=':', color='k', label='mean')
    ax.set_title('L_y - R_y [mm]')
    ax.set_xlabel('t0 [s]')
    ax.grid(True)
    ax.legend()

    ax = fig_hinge.add_subplot(4, 3, 11)
    ax.plot(t0, hinge_L_b[:,1] + hinge_R_b[:,1], label='L_y + R_y')
    ax.axhline(np.nanmean(hinge_L_b[:,1] + hinge_R_b[:,1]), linestyle=':', color='k', label='mean')
    ax.set_title('L_y + R_y [mm]')
    ax.set_xlabel('t0 [s]')
    ax.grid(True)
    ax.legend()

    ax = fig_hinge.add_subplot(4, 3, 12)
    ax.plot(t0, phi_L_s * rad2deg, label='phi_L [deg]')
    ax.plot(t0, phi_R_s * rad2deg, label='phi_R [deg]')
    ax.set_title('Wingbeat reference')
    ax.set_xlabel('t0 [s]')
    ax.grid(True)
    ax.legend()

    fig_hinge.tight_layout()
    fig_hinge.savefig(root + file + '_hinge_diagnostic_bodyframe.pdf')


    # ---------- spectra ----------
    fig_spec = plt.figure(figsize=(16, 8))

    for idx, (side_name, H) in enumerate([('Left', hinge_L_b), ('Right', hinge_R_b)]):
        for j, cname in enumerate(comp_names):
            ax = fig_spec.add_subplot(2, 3, idx*3 + j + 1)
            f0, f_hz, ek = dominant_freq_hz_from_signal(t0, H[:, j])
            if f_hz is not None and ek is not None:
                band = (f_hz >= 0) & (f_hz <= 60)
                ax.plot(f_hz[band], ek[band])
                if np.isfinite(f0):
                    ax.axvline(f0, linestyle='--', color='k', label=f'f0={f0:.2f} Hz')
                if np.isfinite(f_phi_L):
                    ax.axvline(f_phi_L, linestyle=':', color='r', label=f'f_phiL={f_phi_L:.2f} Hz')
                if np.isfinite(f_phi_R):
                    ax.axvline(f_phi_R, linestyle=':', color='g', label=f'f_phiR={f_phi_R:.2f} Hz')
            ax.set_title(f'{side_name} hinge {cname}: spectrum')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Energy')
            ax.grid(True)
            ax.legend()

    fig_spec.tight_layout()
    fig_spec.savefig(root + file + '_hinge_spectra.pdf')


    # ---------- local-frame check on singlewing ----------
    fig_local, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(x_wing_w[:,0], x_wing_w[:,1], 'k.', ms=2, alpha=0.25, label='singlewing mesh')
    axs[0].plot(0.0, 0.0, 'ro', ms=8, label='origin (0,0,0)')
    axs[0].set_title('Left/Right hinge mean position in BODY frame\n')
    axs[0].set_xlabel('x_wing mesh [mm]')
    axs[0].set_ylabel('y_wing mesh [mm]')
    axs[0].grid(True)
    axs[0].axis('equal')
    axs[0].legend()

    axs[1].plot(hinge_L_b[:,0], hinge_L_b[:,1], label='Left hinge (body)')
    axs[1].plot(hinge_R_b[:,0], hinge_R_b[:,1], label='Right hinge (body)')
    axs[1].plot(hinge_L_b_mean[0], hinge_L_b_mean[1], 'bo', ms=8, label='Left mean')
    axs[1].plot(hinge_R_b_mean[0], hinge_R_b_mean[1], 'ro', ms=8, label='Right mean')
    axs[1].axhline(0, color='k', linestyle=':')
    axs[1].axvline(0, color='k', linestyle=':')
    axs[1].set_title('Hinge trajectories in BODY frame (x-y)')
    axs[1].set_xlabel('x_body [mm]')
    axs[1].set_ylabel('y_body [mm]')
    axs[1].grid(True)
    axs[1].axis('equal')
    axs[1].legend()

    plt.tight_layout()
    fig_local.savefig(root + file + '_hinge_xy_bodyframe.pdf')


    
    #%% Rotation + translation vs translation only
    fig_cmp = plt.figure(figsize=(18, 12))

    # LEFT
    ax = fig_cmp.add_subplot(3, 2, 1)
    ax.plot(t0, phi_L_s * rad2deg, label='phi L with t')
    ax.plot(t0, phi_L_not_s * rad2deg, '--', label='phi L no t')
    ax.set_title('Left phi')
    ax.set_xlabel('t0 [s]')
    ax.set_ylabel('[deg]')
    ax.grid(True)
    ax.legend()

    ax = fig_cmp.add_subplot(3, 2, 3)
    ax.plot(t0, alpha_L_s * rad2deg, label='alpha L with t')
    ax.plot(t0, alpha_L_not_s * rad2deg, '--', label='alpha L no t')
    ax.set_title('Left alpha')
    ax.set_xlabel('t0 [s]')
    ax.set_ylabel('[deg]')
    ax.grid(True)
    ax.legend()

    ax = fig_cmp.add_subplot(3, 2, 5)
    ax.plot(t0, theta_L_s * rad2deg, label='theta L with t')
    ax.plot(t0, theta_L_not_s * rad2deg, '--', label='theta L no t')
    ax.set_title('Left theta')
    ax.set_xlabel('t0 [s]')
    ax.set_ylabel('[deg]')
    ax.grid(True)
    ax.legend()
    plt.show()

    # RIGHT
    ax = fig_cmp.add_subplot(3, 2, 2)
    ax.plot(t0, phi_R_s * rad2deg, label='phi R with t')
    ax.plot(t0, phi_R_not_s * rad2deg, '--', label='phi R no t')
    ax.set_title('Right phi')
    ax.set_xlabel('t0 [s]')
    ax.set_ylabel('[deg]')
    ax.grid(True)
    ax.legend()

    ax = fig_cmp.add_subplot(3, 2, 4)
    ax.plot(t0, alpha_R_s * rad2deg, label='alpha R with t')
    ax.plot(t0, alpha_R_not_s * rad2deg, '--', label='alpha R no t')
    ax.set_title('Right alpha')
    ax.set_xlabel('t0 [s]')
    ax.set_ylabel('[deg]')
    ax.grid(True)
    ax.legend()

    ax = fig_cmp.add_subplot(3, 2, 6)
    ax.plot(t0, theta_R_s * rad2deg, label='theta R with t')
    ax.plot(t0, theta_R_not_s * rad2deg, '--', label='theta R no t')
    ax.set_title('Right theta')
    ax.set_xlabel('t0 [s]')
    ax.set_ylabel('[deg]')
    ax.grid(True)
    ax.legend()
    plt.show()

    fig_cmp.tight_layout()
    fig_cmp.savefig(root + file + '_angles_compare_translation_vs_no_translation.pdf')

    fig_diff = plt.figure(figsize=(18, 12))
    def wrap_pi(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    dphi_L   = wrap_pi(phi_L_s   - phi_L_not_s) * rad2deg
    dalpha_L = wrap_pi(alpha_L_s - alpha_L_not_s) * rad2deg
    dtheta_L = wrap_pi(theta_L_s - theta_L_not_s) * rad2deg

    dphi_R   = wrap_pi(phi_R_s   - phi_R_not_s) * rad2deg
    dalpha_R = wrap_pi(alpha_R_s - alpha_R_not_s) * rad2deg
    dtheta_R = wrap_pi(theta_R_s - theta_R_not_s) * rad2deg

    for k, (y, ttl) in enumerate([
        (dphi_L,   'Delta phi L'),
        (dalpha_L, 'Delta alpha L'),
        (dtheta_L, 'Delta theta L'),
        (dphi_R,   'Delta phi R'),
        (dalpha_R, 'Delta alpha R'),
        (dtheta_R, 'Delta theta R'),
    ]):
        ax = fig_diff.add_subplot(3, 2, k+1)
        ax.plot(t0, y)
        ax.axhline(0, color='k', linestyle=':')
        ax.set_title(ttl + ' [deg]')
        ax.set_xlabel('t0 [s]')
        ax.grid(True)
    plt.show()

    fig_diff.tight_layout()
    fig_diff.savefig(root + file + '_angle_differences_translation_vs_no_translation.pdf')

    def rms(x):
        x = np.asarray(x, dtype=float)
        return np.sqrt(np.nanmean(x**2))

    print("\n================ ANGLE COMPARISON: WITH vs WITHOUT TRANSLATION ================\n")
    print(f"phi_L   RMS diff [deg] = {rms(dphi_L):.4f}")
    print(f"alpha_L RMS diff [deg] = {rms(dalpha_L):.4f}")
    print(f"theta_L RMS diff [deg] = {rms(dtheta_L):.4f}")
    print(f"phi_R   RMS diff [deg] = {rms(dphi_R):.4f}")
    print(f"alpha_R RMS diff [deg] = {rms(dalpha_R):.4f}")
    print(f"theta_R RMS diff [deg] = {rms(dtheta_R):.4f}")

    print("\n================ FIT ERRORS: WITH vs WITHOUT TRANSLATION ================\n")
    print(f"err_L    mean with t    = {np.nanmean(err_L):.6f}")
    print(f"err_L    mean no t      = {np.nanmean(err_L_not):.6f}")
    print(f"err_R    mean with t    = {np.nanmean(err_R):.6f}")
    print(f"err_R    mean no t      = {np.nanmean(err_R_not):.6f}")

    # Phase shift diagnostic between with/without translation

    def estimate_time_shift(x, y, dt):
        """
        Returns time shift tau such that y(t) ~ x(t - tau)
        Positive tau means y is delayed relative to x.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        x = x - np.mean(x)
        y = y - np.mean(y)

        corr = np.correlate(y, x, mode='full')
        lags = np.arange(-len(x)+1, len(x))
        lag_best = lags[np.argmax(corr)]
        tau = lag_best * dt
        return tau, lag_best, corr, lags

    dt_sec = np.median(np.diff(t0))
    fbeat = f_phi_L if np.isfinite(f_phi_L) else f_phi_R

    for name, sig_with, sig_without in [
        ("phi_L", phi_L_s, phi_L_not_s),
        ("phi_R", phi_R_s, phi_R_not_s),
        ("alpha_L", alpha_L_s, alpha_L_not_s),
        ("alpha_R", alpha_R_s, alpha_R_not_s),
        ("theta_L", theta_L_s, theta_L_not_s),
        ("theta_R", theta_R_s, theta_R_not_s),
    ]:
        tau, lag_best, corr, lags = estimate_time_shift(sig_with, sig_without, dt_sec)
        phase_deg = 360.0 * fbeat * tau if np.isfinite(fbeat) else np.nan
        print(f"{name}: time shift = {tau*1000:.3f} ms, lag = {lag_best} frames, phase shift = {phase_deg:.2f} deg of cycle")

    plt.ioff()
    plt.show()

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

