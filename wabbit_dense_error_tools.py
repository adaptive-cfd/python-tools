"""
This file contains functions that deal with flusi vs wabbit and dense grids
"""

from wabbit_tools import WabbitHDF5file, tca_2_level, tca_2_tcb
import numpy as np
import bcolors

#
def dense_matrix(  x0, dx, data, level, dim=2, verbose=True, new_format=False ):

    import math
    """ Convert a WABBIT grid to a full dense grid in a single matrix.

    We asssume here that interpolation has already been performed, i.e. all
    blocks are on the same (finest) level.

    returns the full matrix and the domain size. Note matrix is periodic and can
    directly be compared to FLUSI-style results (x=L * 0:nx-1/nx)
    """
    
    # number of blocks
    N = data.shape[0]
    # size of each block
    Bs = np.asarray(data.shape[1:])
    
    # if np.any(Bs % 2 != 0) and new_format:
    #     # Note: skipping of redundant points, hence the -1
    #     # Note: this is still okay after Apr 2020: we save one extra point in the HDF5 file
    #     # for visualization. However, now, the Bs must be odd!
    #     raise ValueError("For the new code without redundant points, the block size should be even!")

    # check if all blocks are on the same level or not
    jmin, jmax = np.min(level), np.max(level)
    if jmin != jmax:
        raise ValueError(bcolors.FAIL + "ERROR! not an equidistant grid yet..." + bcolors.ENDC)

    
    if dim==2:
        # in both uniqueGrid and redundantGrid format, a redundant point is included (it is the first ghost 
        # node in the uniqueGrid format!)
        nx = [int( np.sqrt(N)*(Bs[d]-1) ) for d in range(np.size(Bs))]
    else:
        nx = [int( round( (N)**(1.0/3.0)*(Bs[d]-1) ) ) for d in range(np.size(Bs))]


    # all spacings should be the same - it does not matter which one we use.
    ddx = dx[0,:]
    
    if verbose:
        print("Nblocks :" , (N))
        print("Bs      :" , Bs[::-1])
        print("Spacing :" , ddx)
        print("Domain  :" , ddx*nx[::-1])
        print("Dense field resolution :", nx[::-1] )

    if dim==2:
        # allocate target field
        field = np.zeros(nx)

        # domain size
        box = ddx*nx

        for i in range(N):
            # get starting index of block
            ix0 = int( round(x0[i,0]/dx[i,0]) )
            iy0 = int( round(x0[i,1]/dx[i,1]) )

            # copy block content to data field. Note we skip the last points, which
            # are the redundant nodes (or the first ghost node).
            field[ ix0:ix0+Bs[0]-1, iy0:iy0+Bs[1]-1 ] = data[i, 0:-1 ,0:-1]

    else:
        # allocate target field
        field = np.zeros([nx[0],nx[1],nx[2]])


        # domain size
        box = np.asarray([ddx[0]*nx[0], ddx[1]*nx[1], ddx[2]*nx[2]])

        for i in range(N):
            # get starting index of block
            ix0 = int( round(x0[i,0]/dx[i,0]) )
            iy0 = int( round(x0[i,1]/dx[i,1]) )
            iz0 = int( round(x0[i,2]/dx[i,2]) )

            # copy block content to data field. Note we skip the last points, which
            # are the redundant nodes (or the first ghost node).
            field[ ix0:ix0+Bs[0]-1, iy0:iy0+Bs[1]-1, iz0:iz0+Bs[2]-1 ] = data[i, 0:-1, 0:-1, 0:-1]

    return(field, box)



#
def wabbit_error_vs_flusi(fname_wabbit, fname_flusi, norm=2, dim=2):
    """ Compute the error (in some norm) wrt a flusi field.
    Useful for example for the half-swirl test where no exact solution is available
    at mid-time (the time of maximum distortion)

    NOTE: We require the wabbit-field to be already full (but still in block-data) so run
    ./wabbit-post 2D --sparse-to-dense input_00.h5 output_00.h5
    first
    """
    import numpy as np
    import insect_tools
    import matplotlib.pyplot as plt

    if dim==3:
        print('I think due to fft2usapmle, this routine works only in 2D')
        raise ValueError

    # read in flusi's reference solution
    time_ref, box_ref, origin_ref, data_ref = insect_tools.read_flusi_HDF5( fname_flusi )
    print(data_ref.shape)
    ny = data_ref.shape[1]

    # wabbit field to be analyzed: note has to be full already
    wabbit_obj = WabbitHDF5file()
    wabbit_obj.read(fname_wabbit)
    x0 = wabbit_obj.coords_origin
    dx = wabbit_obj.coords_spacing
    data = wabbit_obj.blocks
    level = wabbit_obj.level
    Bs = data.shape[1]
    Jflusi = (np.log2(ny/(Bs-1)))
    print("Flusi resolution: %i %i %i so desired level is Jmax=%f" % (data_ref.shape[0], data_ref.shape[2], data_ref.shape[2], Jflusi) )

    if dim==2:
        # squeeze 3D flusi field (where dim0 == 1) to true 2d data
        data_ref = data_ref[0,:,:].copy().transpose()
        box_ref = box_ref[1:2].copy()

    # convert wabbit to dense field
    data_dense, box_dense = dense_matrix( x0, dx, data, level, dim )
    
    if data_dense.shape[0] < data_ref.shape[0]:
        # both datasets have different size
        s = int( data_ref.shape[0] / data_dense.shape[0] )
        data_ref = data_ref[::s, ::s].copy()
        raise ValueError(bcolors.FAIL + "ERROR! Both fields are not a the same resolutionn" + bcolors.ENDC)

    if data_dense.shape[0] > data_ref.shape[0]:
        bcolors.warn("WARNING! The reference solution is not fine enough for the comparison! UPSAMPLING!")
        import fourier_tools
        print(data_ref.shape)
        data_ref = fourier_tools.fft2_resample( data_ref, data_dense.shape[1] )

    err = np.ndarray.flatten(data_ref-data_dense)
    exc = np.ndarray.flatten(data_ref)

    err = np.linalg.norm(err, ord=norm) / np.linalg.norm(exc, ord=norm)
    print( "error was e=%e" % (err) )

    return err


#
def flusi_error_vs_flusi(fname_flusi1, fname_flusi2, norm=2, dim=2):
    """ compute error given two flusi fields
    """
    import numpy as np
    import insect_tools

    # read in flusi's reference solution
    time_ref, box_ref, origin_ref, data_ref = insect_tools.read_flusi_HDF5( fname_flusi1 )

    time, box, origin, data_dense = insect_tools.read_flusi_HDF5( fname_flusi2 )

    if len(data_ref) is not len(data_dense):
        raise ValueError(bcolors.FAIL + "ERROR! Both fields are not a the same resolution" + bcolors.ENDC)

    err = np.ndarray.flatten(data_dense-data_ref)
    exc = np.ndarray.flatten(data_ref)

    err = np.linalg.norm(err, ord=norm) / np.linalg.norm(exc, ord=norm)

    print( "error was e=%e" % (err) )

    return err

def wabbit_error_vs_wabbit(fname_ref_list, fname_dat_list, norm=2, dim=2):
    """
    Read two wabbit files, which are supposed to have all blocks at the same
    level. Then, we re-arrange the data in a dense matrix (wabbit_tools.dense_matrix)
    and compute the relative error:
        
        err = || u2 - u1 || / || u1 ||
        
    The dense array is flattened before computing the error (np.ndarray.flatten)
    
    New: if a list of files is passed instead of a single file,
    then we compute the vector norm.
    
    Input:
    ------
    
        fname_ref : scalar or list of string 
            file to read u1 from
            
        fname_dat : scalar or list of string
            file to read u2 from
            
        norm : scalar, float
            Can be either 2, 1, or np.inf (passed to np.linalg.norm)
    
    Output:
    -------
        err
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if  not isinstance(fname_ref_list, list):
        fname_ref_list = [fname_ref_list]
    
    if  not isinstance(fname_dat_list, list):
        fname_dat_list = [fname_dat_list]
    
    assert len(fname_dat_list) == len(fname_ref_list) 
        
    for k, (fname_ref, fname_dat) in enumerate (zip(fname_ref_list,fname_dat_list)):
        wabbit_ref = WabbitHDF5file()
        wabbit_ref.read(fname_ref)
        wabbit_dat = WabbitHDF5file()
        wabbit_dat.read(fname_dat)
        time1, time2 = wabbit_ref.time, wabbit_dat.time
        x01, x02 = wabbit_ref.coords_origin, wabbit_dat.coords_origin
        dx1, dx2 = wabbit_ref.coords_spacing, wabbit_dat.coords_spacing
        box1, box2 = wabbit_ref.domain_size, wabbit_dat.domain_size
        data1, data2 = wabbit_ref.blocks, wabbit_dat.blocks
        treecode_num1, treecodenum2 = wabbit_ref.block_treecode_num, wabbit_dat.block_treecode_num
        level1, level2 = wabbit_ref.level, wabbit_dat.level
    
        data1, box1 = dense_matrix( x01, dx1, data1, level1, dim )
        data2, box2 = dense_matrix( x02, dx2, data2, level2, dim )
        
        if (len(data1) != len(data2)) or (np.linalg.norm(box1-box2)>1e-15):
           raise ValueError(bcolors.FAIL + "ERROR! Both fields are not a the same resolution" + bcolors.ENDC)

        if k==0:
            err = np.ndarray.flatten(data1-data2)
            exc = np.ndarray.flatten(data1)
        else:
            err = np.concatenate((err,np.ndarray.flatten(data1-data2)))
            exc = np.concatenate((exc,np.ndarray.flatten(data1)))
        

    err = np.linalg.norm(err, ord=norm) / np.linalg.norm(exc, ord=norm)

    print( "error was e=%e" % (err) )

    return err


#
def to_dense_grid( fname_in, fname_out = None, dim=2 ):
    """ Convert a WABBIT grid to a full dense grid in a single matrix.

    We asssume here that interpolation has already been performed, i.e. all
    blocks are on the same (finest) level.
    """
    import numpy as np
    import insect_tools
    import matplotlib.pyplot as plt

    # read data
    wabbit_obj = WabbitHDF5file()
    wabbit_obj.read(fname_in)
    time = wabbit_obj.time
    x0 = wabbit_obj.coords_origin
    dx = wabbit_obj.coords_spacing
    box = wabbit_obj.domain_size
    data = wabbit_obj.blocks
    treecode_num = wabbit_obj.block_treecode_num
    level = wabbit_obj.level

    # convert blocks to complete matrix
    field, box = dense_matrix(  x0, dx, data, level, dim=dim )

    # write data to FLUSI-type hdf file
    if fname_out:
        insect_tools.write_flusi_HDF5( fname_out, time, box, field)
    else:        
        dx = [b/(np.size(field,k)) for k,b in enumerate(box)]
        X = [np.arange(0,np.size(field,k))*dx[k] for k,b in enumerate(box)]
        return field, box, dx, X
    

#
def flusi_to_wabbit_dir(dir_flusi, dir_wabbit , *args, **kwargs ):
    """
    Convert directory with flusi *h5 files to wabbit *h5 files
    """
    import re
    import os
    import glob

    if not os.path.exists(dir_wabbit):
        os.makedirs(dir_wabbit)
    if not os.path.exists(dir_flusi):
        bcolors.err("The given directory does not exist!")

    files = glob.glob(dir_flusi+'/*.h5')
    files.sort()
    for file in files:

        fname_wabbit = dir_wabbit + "/" + re.split("_\\d+.h5",os.path.basename(file))[0]

        flusi_to_wabbit(file, fname_wabbit ,  *args, **kwargs )

#
def flusi_to_wabbit(fname_flusi, fname_wabbit , level, dim=2, dtype=np.float64 ):

    """
    Convert flusi data file to wabbit data file.
    """
    import numpy as np
    import insect_tools
    import matplotlib.pyplot as plt


    # read in flusi's reference solution
    time, box, origin, data_flusi = insect_tools.read_flusi_HDF5( fname_flusi, dtype=dtype )
    box = box[1:]
    
    data_flusi = np.squeeze(data_flusi).T
    Bs = field_shape_to_bs(data_flusi.shape,level)
    dense_to_wabbit_hdf5(data_flusi, fname_wabbit , Bs, box, time, dtype=dtype)


#
def dense_to_wabbit_hdf5(ddata, name , Bs, box_size = None, time = 0, iteration = 0, dtype=np.float64):

    """
    This function creates a <name>_<time>.h5 file with the wabbit
    block structure from a given dense data matrix.
    Therefore the dense data is divided into equal blocks,
    similar as sparse_to_dense option in wabbit-post.

    Input:
    ======
         - ddata... 2d/3D array of the data you want to write to a file
                     Note ddata.shape=[Ny,Nx] !!
         - name ... prefix of the name of the datafile (e.g. "rho", "p", "Ux")
         - Bs   ... number of grid points per block
                    is a 2D/3D dimensional array with Bs[0] being the number of
                    grid points in x direction etc.
                    The data size in each dimension has to be dividable by Bs.
                    
    Optional Input:
    =============
        - box_size... 2D/3D array of the size of your box [Lx, Ly, Lz]
        - time    ... time of the data
        - iteration ... iteration of the time snappshot
        
    Output:
    =======
        - filename of the hdf5 output

    """
    # concatenate filename in the same style as wabbit does
    fname = name + "_%12.12d" % int(time*1e6) + ".h5"
    Ndim = ddata.ndim
    Nsize = np.asarray(ddata.shape)
    level = 0
    Bs = np.asarray(Bs)# make sure Bs is a numpy array
    Bs = Bs[::-1] # flip Bs such that Bs=[BsY, BsX] the order is the same as for Nsize=[Ny,Nx]
    
    #########################################################
    # do some initial checks on the input data
    # 1) check if the size of the domain is given
    if box_size is None:
        box = np.ones(Ndim)
    else:
        box = np.asarray(box_size)

    if (type(Bs) is int):
        Bs = [Bs]*Ndim
        
    # 2) check if number of lattice points is block decomposable
    # loop over all dimensions
    for d in range(Ndim):
        # check if Block is devidable by Bs
        if (np.remainder(Nsize[d], Bs[d]-1) == 0):
            if(is_power2(Nsize[d]//(Bs[d]-1))):
                level = int(max(level, np.log2(Nsize[d]/(Bs[d]-1))))
            else:
                bcolors.err("Number of Intervals must be a power of 2!")
        else:
            bcolors.err("datasize must be multiple of Bs!")
            
    # 3) check dimension of array:
    if Ndim < 2 or Ndim > 3:
        bcolors.err("dimensions are wrong")
    #########################################################

    # assume periodicity:
    data = np.zeros(Nsize+1,dtype=dtype)
    if Ndim == 2:
        data[:-1, :-1] = ddata
        # copy first row and column for periodicity
        data[-1, :] = data[0, :]
        data[:, -1] = data[:, 0]
    else:
        data[:-1, :-1, :-1] = ddata
        # copy for periodicity
        data[-1, :, :] = data[0, :, :]
        data[:, -1, :] = data[:, 0, :]
        data[:, :, -1] = data[:, :, 0]

    # number of intervals in each dimension
    Nintervals = [int(2**level)]*Ndim  # note [val]*3 means [val, val , val]
    Lintervals = box[:Ndim]/np.asarray(Nintervals)
    Lintervals = Lintervals[::-1]
    

    x0 = []
    treecode = []
    dx = []
    bdata = []
    if Ndim == 3:
        for ibx in range(Nintervals[0]):
            for iby in range(Nintervals[1]):
                for ibz in range(Nintervals[2]):
                    x0.append([ibx, iby, ibz]*Lintervals)
                    dx.append(Lintervals/(Bs-1))

                    lower = [ibx, iby, ibz]* (Bs - 1)
                    lower = np.asarray(lower, dtype=int)
                    upper = lower + Bs

                    treecode.append(blockindex2treecode([ibx, iby, ibz], 3, level))
                    bdata.append(data[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
    else:
        for ibx in range(Nintervals[0]):
            for iby in range(Nintervals[1]):
                x0.append([ibx, iby]*Lintervals)
                dx.append(Lintervals/(Bs-1))
                # lower = [ibx, iby]* (Bs - 1)
                lower = np.asarray(lower, dtype=int)
                upper = lower + Bs
                treecode.append(blockindex2treecode([ibx, iby], 2, level))
                bdata.append(data[lower[0]:upper[0], lower[1]:upper[1]])


    x0 = np.asarray(x0,dtype=dtype)
    dx = np.asarray(dx,dtype=dtype)
    treecode = np.asarray(treecode, dtype=int)
    block_data = np.asarray(bdata, dtype=dtype)

    treecode_num = tca_2_tcb(treecode, dim=Ndim, max_level=treecode.shape[1])
    # blocks are dense so level is the same everywhere
    level = tca_2_level(treecode)

    w_obj = WabbitHDF5file()
    w_obj.fill_vars(box, block_data, treecode_num, level, time, iteration)
    w_obj.coords_origin = x0
    w_obj.coords_spacing = dx
    w_obj.write(fname)

    return fname


def is_power2(num):
    'states if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)

###
def field_shape_to_bs(Nshape, level):
    """
     For a given shape of a dense field and maxtreelevel return the
     number of points per block wabbit uses
    """

    n = np.asarray(Nshape)
    
    for d in range(n.ndim):
        # check if Block is devidable by Bs
        if (np.remainder(n[d], 2**level) != 0):
            bcolors.err("Number of Grid points has to be a power of 2!")
            
    # Note we have to flip  n here because Bs = [BsX, BsY]
    # The order of Bs is choosen like it is in WABBIT.
    # NB: while this definition is the one from a redundant grid,
    # it is used the same in the uniqueGrid ! The funny thing is that in the latter
    # case, we store the 1st ghost node to the H5 file - this is required for visualization.
    return n[::-1]//2**level + 1

#
# calculates treecode array from the index of the block
# Note: other then in fortran we start counting from 0
def blockindex2treecode(ix, dim, treeN):

    treecode = np.zeros(treeN)
    for d in range(dim):
        # convert block index to binary
        binary = list(format(ix[d],"b"))
        # flip array and convert to numpy
        binary = np.asarray(binary[::-1],dtype=int)
        # sum up treecodes
        lt = np.size(binary)
        treecode[:lt] = treecode[:lt] + binary *2**d

    # flip again befor returning array
    return treecode[::-1]