#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:41:48 2017

@author: engels
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import h5py
import bcolors
import copy
from inifile_tools import *
from analytical_functions import *
import logging


def read_wabbit_hdf5_legacy(fname_wabbit):
    """ Read a wabbit-type HDF5 of block-structured data, without using fancy objects.

    Returns:
    =======
    ``time, x0, dx, box, data, treecode``
    
    Get number of blocks and blocksize as

    ``N, Bs = data.shape[0], data.shape[1]``
    
    A tale of two kingdoms:
    =======================
    Ordering:
        
        - The **WABBIT** kingdom stores data as [ix,iy,iz,icomponent,ib], for a scalar (and this is the usual case in python, since
          wabbit stores each component in a separate file), this reduces to [ix,iy,iz,ib]. Last index is block index.
    
        - The **PYTHON** kingdom reads those arrays using HDF5, and the resulting order is [ib,iz,iy,ix]. First index is block index.
    
    Block size:
    
        - The **WABBIT** kingdom defines the block size BS as the interior nodes of a block. As of 09/2025, this is an EVEN number.
          When storing HDF5 output, we noticed paraview has gaps between the blocks - this is tedious. Hence, WABBIT kingdom now
          stores BS+1 point, including the first ghost node. This is an ODD number.
    
        - The **PYTHON** kingdom reads in the file, and it defines the block size as BS_PYTHON = BS_WABBIT + 1, which is the size of the array. This is an ODD 
          number. Both numbers [BS_PYTHON=33 & BS_WABBIT=32] describe THE SAME GRID AND SAME BLOCK SIZE. Confused yet?
    
    """
    # create wabbit file object ...
    wabbit_obj = WabbitHDF5file()
    # ... and read in the actual file
    wabbit_obj.read(fname_wabbit)
    
    x0       = wabbit_obj.coords_origin
    dx       = wabbit_obj.coords_spacing
    data     = wabbit_obj.blocks
    treecode = wabbit_obj.treecode
    time     = wabbit_obj.time
    box      = wabbit_obj.domain_size
    
    return time, x0, dx, box, data, treecode


#------------------------------------------------------------------------------
# compute some keyvalues to compare the files
#------------------------------------------------------------------------------
def keyvalues(domain_size, dx, data):
    max1, min1 = np.max(data), np.min(data)
    mean1, squares1 = 0.0, 0.0
    
    # loop over all blocks, ignore last point to only use unique grid
    for i in range( data.shape[0] ):
        if len(data.shape) == 3: sum_block = np.sum(data[i,:-1,:-1])  # 2D
        else: sum_block = np.sum(data[i,:-1,:-1, :-1])  ## 3D
        mean1 = mean1 + np.prod(dx[i,:]) * sum_block
        squares1 = squares1 + np.prod(dx[i,:]) * sum_block**2
    # divide integrals by area to get mean value
    mean1 /= np.prod(domain_size)
    squares1 /= np.prod(domain_size)
        
    return(max1, min1, mean1, squares1 )


# this object contains all important details about a read in wabbit-file
# everything is neat and at one position and simplifies things
# A grid is uniquely defined by its dimension, block size, domain size
# The individual grid partition is uniquely defined by the number of blocks, treecode and level arrays
class WabbitHDF5file:
    # lets define all objects it can possess
    # first all attributes, those will always be set
    # should be consistent with write attribute list in saveHDF5_tree
    periodic_BC = symmetry_BC = version = block_size = time = \
    iteration = total_number_blocks = max_level = dim = domain_size = []

    # lets define all fields
    blocks = np.array([])
    blocks_order = np.array([])  # in case we are reordering, this is used for reading back the blocks in the original order
    coords_origin = np.array([])
    coords_spacing = np.array([])
    block_treecode_num = np.array([])
    block_treecode = np.array([])
    procs = np.array([])
    refinement_status = np.array([])
    level = np.array([])
    lgt_ids = np.array([])

    # some helping constructs
    tc_dict = []
    tc_find = []
    orig_file = ""

    """
        python has no init overloading so I keep this empty. Always make sure to call read or set vars afterwards!
    """
    def __init__(self):
        pass

    """
        reads in variables, we can either define to read in "all", "meta" (all but blocks) or only a single variable
    """    
    def read(self, file, read_var='all', verbose=True):
        if verbose:
            print(f"Reading {read_var} of file= {file}")
        fid = h5py.File(file,'r')
        dset_id = fid.get('blocks')

        # read attributes - always read all attributes
        version = dset_id.attrs.get('version', default=[0])
        if isinstance(version, (list, np.ndarray)):
            self.version = version[0]
        else:
            self.version = version
        self.periodic_BC = dset_id.attrs.get('periodic_BC')
        self.symmetry_BC = dset_id.attrs.get('symmetry_BC')
        self.block_size = dset_id.attrs.get('block-size')
        # for all scalar entries they can be read in as scalars or lists
        
        self.time = dset_id.attrs.get('time', default=[None])[0]
        iteration = dset_id.attrs.get('iteration', default=[None])
        if isinstance(iteration, (list, np.ndarray)):
            self.iteration = iteration[0]
        else:
            self.iteration = iteration
        total_number_blocks = dset_id.attrs.get('total_number_blocks')
        if isinstance(total_number_blocks, (list, np.ndarray)):
            self.total_number_blocks = total_number_blocks[0]
        else:
            self.total_number_blocks = total_number_blocks
        self.max_level = dset_id.attrs.get('max_level', default=[None])[0]
        self.dim = dset_id.attrs.get('dim', default=[None])[0]
        self.domain_size = dset_id.attrs.get('domain-size')

        # read fields, only if requested
        if read_var in ["coords_origin", "all", "meta"]:
            self.coords_origin = np.array(fid['coords_origin'][:])
        if read_var in ["coords_spacing", "all", "meta"]:
            self.coords_spacing = np.array(fid['coords_spacing'][:])
        if read_var in ["blocks", "all"]:
            self.blocks = np.array(fid['blocks'], dtype=np.float64)
        self.blocks_order = np.arange(self.total_number_blocks)  # default order is the order in the file
        
        # very old versions do not have fancy variables
        if self.version <= 20200902:
            self.refinement_status = [0]*self.total_number_blocks
            self.procs = [-1]*self.total_number_blocks
            self.lgt_ids = [-1]*self.total_number_blocks
            self.block_size = self.blocks.shape[1:]
        else:
            if read_var in ["refinement_status", "all", "meta"]:
                self.refinement_status = np.array(fid['refinement_status'])
                # # JB HACK for float numbers saved in refinement status
                # self.refinement_status = np.frombuffer(self.refinement_status.tobytes(), dtype=np.float32)
            if read_var in ["procs", "all", "meta"]:
                self.procs = np.array(fid['procs'])
            if read_var in ["lgt_ids", "all", "meta"]:
                self.lgt_ids = np.array(fid['lgt_ids'])
        
        # read in treecode - dependent on version
        # older version - treecode array, create dim, max_level and fields level and block_treecode_num
        if self.version < 20240410:
            # fill in attributes
            try:
                self.dim = 3 - (self.block_size[2] == 1)  # 2D if only one point in z-direction
            except IndexError:
                self.dim = 2  # self.block_size[2] is not set
                self.block_size = np.array(list(self.block_size) + [1])
            if read_var in ["block_treecode", "all", "meta"]:
                self.block_treecode = np.array(fid['block_treecode'])
                self.max_level = self.block_treecode.shape[1]
            if read_var in ["block_treecode_num", "all", "meta"]:
                block_treecode = np.array(fid['block_treecode'])
                self.max_level = self.block_treecode.shape[1]
                self.block_treecode_num = tca_2_tcb(block_treecode, dim=self.dim, max_level=self.max_level)
            if read_var in ["level", "all", "meta"]:
                block_treecode = np.array(fid['block_treecode'])
                self.max_level = self.block_treecode.shape[1]
                self.level = tca_2_level(block_treecode).astype(int)
        # new version - reconstruct block_treecode for compatibility
        else:
            if read_var in ["block_treecode", "all", "meta"]:
                block_treecode_num = np.array(fid['block_treecode_num'])
                level = np.array(fid['level'])
                self.block_treecode = tcb_level_2_tcarray(block_treecode_num, level, self.max_level, self.dim)
            if read_var in ["block_treecode_num", "all", "meta"]:
                self.block_treecode_num = np.array(fid['block_treecode_num'])
            if read_var in ["level", "all", "meta"]:
                self.level = np.array(fid['level'])
        # watch for block_size
        if self.version == 20200408 or self.version >= 20231602:
            self.block_size[:self.dim] += 1
            #print("!!!Warning old (old branch: newGhostNodes) version of wabbit format detected!!!")
        else:
            print("This file includes redundant points")
        
        # close and we are happy
        fid.close()

        # create objects which are handy to have
        # dictionary to quickly check if a block exists
        self.tc_dict = {(self.block_treecode_num[j], self.level[j]): True for j in range(self.total_number_blocks)}
        self.tc_find = {(tc, lvl): idx for idx, (tc, lvl) in enumerate(zip(self.block_treecode_num, self.level))}
        self.orig_file = file
        
        
        # consistency check: does treecode and (dx,x0) match for each block?
        for i in range(self.blocks.shape[0]):
            # origin, as stored in the file
            x0_this = self.coords_origin[i,:]            
            # origin, computed from treecode (should, of course, be the same!)
            x0_comp = treecode2origin(self.block_treecode_num[i], max_level=self.max_level, dim=self.dim, domain_size=self.domain_size)
            
            if np.max(np.abs(x0_this-x0_comp)) > 1.0e-13:
                raise ValueError('Inconsistency found: block spacing stored in coords_spacing and the one computed from block treecode do not match!')

    # init a wabbit state by data
    # A grid is uniquely defined by its dimension (from blocks), block size (from blocks), domain size
    # The individual grid partition is uniquely defined by the number of blocks (from blocks), treecode and level arrays
    # for time knowledge we set the time and iteration as well
    def fill_vars(self, domain_size, blocks, treecode, level, time, iteration, max_level=21):
        self.dim = len(blocks.shape[1:])
        self.block_size = np.array(blocks.shape[1:])
        if self.dim==2: self.block_size = np.append(self.block_size, 1)
        self.domain_size = np.array(domain_size)
        self.blocks = blocks.copy()
        self.block_treecode_num = treecode.copy()
        self.level = level.copy()
        self.total_number_blocks = blocks.shape[0]
        self.time = time
        self.iteration = iteration
        self.blocks_order = np.arange(self.total_number_blocks)  # default order is the order in the file

        # compute attrs which are not set
        self.version = 20240410
        self.periodic_BC = np.array([1, 1, 1])
        self.symmetry_BC = np.array([0, 0, 0])
        self.max_level = max_level  # for now set to max

        # set fields for meta data of blocks
        self.lgt_ids = np.arange(1,self.total_number_blocks+1)
        self.procs = np.zeros(self.total_number_blocks)
        self.refinement_status = np.zeros(self.total_number_blocks)
        self.block_treecode = tcb_level_2_tcarray(treecode, level, self.max_level, self.dim)
        self.coords_origin = np.zeros([self.total_number_blocks, self.dim])
        self.coords_spacing = np.zeros([self.total_number_blocks, self.dim])
        for i_b in range(self.total_number_blocks):
            self.coords_origin[i_b, :] = treecode2origin(self.block_treecode_num[i_b], self.max_level, self.dim, self.domain_size)
            self.coords_spacing[i_b, :] = level2spacing(self.level[i_b], self.dim, self.block_size, self.domain_size)
    
        # create objects which are handy to have
        # dictionary to quickly check if a block exists
        self.tc_dict = {(self.block_treecode_num[j], self.level[j]): True for j in range(self.total_number_blocks)}
        self.tc_find = {(tc, lvl): idx for idx, (tc, lvl) in enumerate(zip(self.block_treecode_num, self.level))}

    # init values from a matrix and set them into a grid on uniform level
    def fill_from_matrix(self, block_values, bs, domain_size=[1,1,1], dim=3, max_level=21, time=0.0, iteration=0, includes_g=False):
        # extract level from size of array
        level_num = np.log2((block_values.shape[0]-includes_g)/bs[0])
        if int(level_num) != level_num:
            print(f"Input array has wrong size: {block_values.shape[0]-includes_g}. Ensure size/bs is a power of 2")
            return
        level_num = int(level_num)

        # alter values, we need to copy first line as we have redundant setting - this assumes periodicity
        block_red = np.zeros(np.array(block_values.shape)+1-includes_g)
        if not includes_g:
            if dim == 2:
                block_red[:-1,:-1] = block_values[:,:]   # copy interior
                block_red[ -1,:-1] = block_values[0,:]   # copy x-line
                block_red[:-1, -1] = block_values[:,0]   # copy y-line
                block_red[ -1, -1] = block_values[0,0]   # copy last corner
            else:
                block_red[:-1,:-1,:-1] = block_values[:,:,:]   # copy interior
                block_red[ -1,:-1,:-1] = block_values[0,:,:]   # copy x-face
                block_red[:-1, -1,:-1] = block_values[:,0,:]   # copy y-face
                block_red[:-1,:-1, -1] = block_values[:,:,0]   # copy z-face
                block_red[ -1, -1,:-1] = block_values[0,0,:]   # copy xy-edge
                block_red[ -1,:-1, -1] = block_values[0,:,0]   # copy xz-edge
                block_red[:-1, -1, -1] = block_values[:,0,0]   # copy yz-edge
                block_red[ -1, -1, -1] = block_values[0,0,0]   # copy last corner
        else:
            if dim == 2: block_red[:,:] = block_values[:,:]   # copy interior
            if dim == 3: block_red[:,:,:] = block_values[:,:,:]   # copy interior

        number_blocks = 2**(level_num*dim)
        treecode = np.zeros(number_blocks)
        level = np.ones(number_blocks)*level_num
        if dim == 2:
            blocks = np.zeros([number_blocks, bs[0]+1, bs[1]+1])
        else:
            blocks = np.zeros([number_blocks, bs[0]+1, bs[1]+1, bs[2]+1])

        # prepare treecode
        for i_b in range(number_blocks):
            # encoding is 1-based
            if dim == 2:
                ix, iy = i_b//(2**level_num)+1, i_b%(2**level_num)+1
                tc = tc_encoding([ix, iy], level=level_num, max_level=max_level, dim=dim)
            else:
                ix, iy, iz = i_b//(2**(2*level_num))+1, (i_b//(2**level_num))%(2**level_num)+1, i_b%(2**(level_num))+1
                tc = tc_encoding([ix, iy, iz], level=level_num, max_level=max_level, dim=dim)
            treecode[i_b] = int(tc)
        
        # fill blocks array by transcribing part of array
        for i_b in range(number_blocks):
            if dim == 2:
                ix, iy = i_b//(2**level_num), i_b%(2**level_num)
                ix, iy = ix*bs[0], iy*bs[1]
                blocks[i_b, :, :] = block_red[ix:ix+bs[0]+1, iy:iy+bs[1]+1].transpose(1, 0)
            else:
                ix, iy, iz = i_b//(2**(2*level_num)), (i_b//(2**level_num))%(2**level_num), i_b%(2**(level_num))
                ix, iy, iz = ix*bs[0], iy*bs[1], iz*bs[2]
                blocks[i_b, :, :, :] = block_red[ix:ix+bs[0]+1, iy:iy+bs[1]+1, iz:iz+bs[2]+1].transpose(2, 1, 0)
        
        self.fill_vars(domain_size, blocks, treecode, level, time, iteration, max_level)


    # let it write itself
    def write(self, file, verbose=True):
        """ Write data from wabbit to an HDF5 file
            Note: hdf5 saves the arrays in [Nz, Ny, Nx] order!
            So: data.shape = Nblocks, Bs[3], Bs[2], Bs[1]
        """
        if verbose:
            print(f"Writing file= {file}")
        
        fid = h5py.File( file, 'w')
        
        
        # consistency check: does treecode and (dx,x0) match for each block?
        for i in range(self.blocks.shape[0]):
            # origin as stored in the file
            x0_this = self.coords_origin[i,:]            
            # origin, computed from treecode (should, of course, be the same!)
            x0_comp = treecode2origin(self.block_treecode_num[i], max_level=self.max_level, dim=self.dim, domain_size=self.domain_size)
            
            if np.max(np.abs(x0_this-x0_comp)) > 1.0e-13:
                raise ValueError('Inconsistency found: block spacing stored in coords_spacing and the one computed from block treecode do not match!')
        

        # those are necessary for wabbit
        fid.create_dataset( 'blocks', data=self.blocks, dtype=np.float64)
        fid.create_dataset( 'block_treecode_num', data=self.block_treecode_num, dtype=np.int64)
        fid.create_dataset( 'level', data=self.level, dtype=np.int32)

        # those are optional and not read in from wabbit
        fid.create_dataset( 'coords_origin', data=self.coords_origin, dtype=np.float64)
        fid.create_dataset( 'coords_spacing', data=self.coords_spacing, dtype=np.float64)
        fid.create_dataset( 'block_treecode', data=self.block_treecode, dtype=np.int32)
        fid.create_dataset( 'refinement_status', data=self.refinement_status, dtype=np.int32)
        fid.create_dataset( 'procs', data=self.procs, dtype=np.int32)
        fid.create_dataset( 'lgt_ids', data=self.lgt_ids, dtype=np.int32)

        fid.close()

        # watch for block_size
        if self.version == 20200408 or self.version >= 20231602:
            self.block_size[:self.dim] -= 1

        # write attributes
        # those are necessary for wabbit
        fid = h5py.File(file,'a')
        dset_id = fid.get( 'blocks' )
        dset_id.attrs.create( "version", [20240410], dtype=np.int32) # this is used to distinguish wabbit file formats
        dset_id.attrs.create( "block-size", self.block_size, dtype=np.int32) # this is used to distinguish wabbit file formats
        dset_id.attrs.create('time', [self.time], dtype=np.float64)
        dset_id.attrs.create('iteration', [self.iteration], dtype=np.int32)
        dset_id.attrs.create('max_level', [self.max_level], dtype=np.int32)
        dset_id.attrs.create('dim', [self.dim], dtype=np.int32)
        dset_id.attrs.create('domain-size', self.domain_size, dtype=np.float64)
        dset_id.attrs.create('total_number_blocks', [self.total_number_blocks], dtype=np.int32)
        dset_id.attrs.create('periodic_BC', self.periodic_BC, dtype=np.int32)
        dset_id.attrs.create('symmetry_BC', self.symmetry_BC, dtype=np.int32)

        # repair block_size
        if self.version == 20200408 or self.version >= 20231602:
            self.block_size[:self.dim] += 1

        # those are optional and not read in from wabbit
        # currently none
        fid.close()


    # for large data, we do not want to read all block values at once, as it is simply not feasible
    # however, we might still want to read or write single blocks, so those functions deal with that
    def block_read(self, i_b, file=None):
        # some safety checks
        file_read = self.orig_file if not file else file
        if not file_read:
            print("Tried to access a single block but no file is given")
            return None

        if not self.total_number_blocks:
            print("Tried to access a single block before the WabbitStateOject was initialized?")
            return None

        if i_b < 0 or i_b >= self.total_number_blocks:
            print("Tried to access a block outside block range")
            return None

        fid = h5py.File(file_read,'r')
        block = fid['blocks'][self.blocks_order[i_b], :]
        fid.close()
        return block


    def block_write(self, i_b, block, file=None):
        # some safety checks
        file_write = self.orig_file if not file else file
        if not file_write:
            print("Tried to access a single block but no file is given")
            return None

        if not self.total_number_blocks:
            print("Tried to access a single block before the WabbitStateOject was initialized?")
            return None

        if i_b < 0 or i_b >= self.total_number_blocks:
            print("Tried to access a block outside block range")
            return None

        fid = h5py.File(file_write,'r')
        fid['blocks'][self.blocks_order[i_b], :] = block  # we assume here that block sizes are equal
        fid.close()
        return
            

    # define the == operator for objects
    # this is only true if objects are 100% similar
    # this is not the case for different simulations so use other function for that
    def __eq__(self, other):
        # check if both are wabbit objects
        isWabbit = isinstance(other, self.__class__)
        if not isWabbit:
            return False
        
        # literally check attributes
        if not np.all(self.block_size == other.block_size): return False
        if not self.time == other.time: return False
        if not self.iteration == other.iteration: return False
        if not self.max_level == other.max_level: return False
        if not self.dim == other.dim: return False
        if not np.all(self.domain_size == other.domain_size): return False
        if not self.total_number_blocks == other.total_number_blocks: return False
        if not np.all(self.periodic_BC == other.periodic_BC): return False
        if not np.all(self.symmetry_BC == other.symmetry_BC): return False
    
        # check important fields
        if np.linalg.norm(self.blocks - other.blocks) != 0: return False
        if np.linalg.norm(self.block_treecode_num - other.block_treecode_num) != 0: return False
        if np.linalg.norm(self.level - other.level) != 0: return False

        # puhhh everything is passed so both are equal
        return True
    
    # overwrite + operator, can handle other wabbitstatefiles or scalars as integers/floats
    def __add__(self, other):

        new_obj = copy.deepcopy(self)
        if isinstance(other, WabbitHDF5file):
            equal_grid = self.compareGrid(other)
            equal_attr = self.compareAttr(other)
            if not equal_grid:
                print(bcolors.FAIL + f"WARNING: Grids are not equal, operation interpolated for non-consistent blocks- This might take a while" + bcolors.ENDC)
                grid_interpolator = other.create_interpolator()
            if not equal_attr:
                print(bcolors.WARNING + f"ERROR: Attributes are not equal" + bcolors.ENDC)
                return None

            # blocks are not structured similarly so we have to apply blockwise
            for i_blocks in range(new_obj.total_number_blocks):
                i_other = other.get_block_id(self.block_treecode_num[i_blocks], self.level[i_blocks])
                if (i_other != -1):
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] + other.blocks[i_other, :]
                else:
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] + \
                        other.interpolate_block(new_obj.blocks[i_blocks, :], new_obj.coords_origin[i_blocks], new_obj.coords_spacing[i_blocks], grid_interpolator)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            new_obj.blocks[:] += other

        return new_obj

    # overwrite - operator, can handle other wabbitstatefiles or scalars as integers/floats
    def __sub__(self, other):

        new_obj = copy.deepcopy(self)
        if isinstance(other, WabbitHDF5file):
            equal_grid = self.compareGrid(other)
            equal_attr = self.compareAttr(other)
            if not equal_grid:
                print(bcolors.FAIL + f"WARNING: Grids are not equal, operation interpolated for non-consistent blocks- This might take a while" + bcolors.ENDC)
                grid_interpolator = other.create_interpolator()
            if not equal_attr:
                print(bcolors.WARNING + f"ERROR: Attributes are not equal" + bcolors.ENDC)
                return None

            # blocks are not structured similarly so we have to apply blockwise
            for i_blocks in range(new_obj.total_number_blocks):
                i_other = other.get_block_id(self.block_treecode_num[i_blocks], self.level[i_blocks])
                if (i_other != -1):
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] - other.blocks[i_other, :]
                else:
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] - \
                        other.interpolate_block(new_obj.blocks[i_blocks, :], new_obj.coords_origin[i_blocks], new_obj.coords_spacing[i_blocks], grid_interpolator)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            new_obj.blocks[:] -= other
        return new_obj
    
    # overwrite * operator, can handle other wabbitstatefiles or scalars as integers/floats
    def __mul__(self, other):
        new_obj = copy.deepcopy(self)
        if isinstance(other, WabbitHDF5file):
            equal_grid = self.compareGrid(other)
            equal_attr = self.compareAttr(other)
            if not equal_grid:
                print(bcolors.FAIL + f"WARNING: Grids are not equal, operation interpolated for non-consistent blocks- This might take a while" + bcolors.ENDC)
                grid_interpolator = other.create_interpolator()
            if not equal_attr:
                print(bcolors.WARNING + f"ERROR: Attributes are not equal" + bcolors.ENDC)
                return None

            new_obj = copy.deepcopy(self)
            # blocks are not structured similarly so we have to apply blockwise
            for i_blocks in range(new_obj.total_number_blocks):
                i_other = other.get_block_id(self.block_treecode_num[i_blocks], self.level[i_blocks])
                if (i_other != -1):
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] * other.blocks[i_other, :]
                else:
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] * \
                        other.interpolate_block(new_obj.blocks[i_blocks, :], new_obj.coords_origin[i_blocks], new_obj.coords_spacing[i_blocks], grid_interpolator)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            new_obj.blocks[:] *= other
        return new_obj
    
    # overwrite / operator, can handle other wabbitstatefiles or scalars as integers/floats
    def __truediv__(self, other):
        new_obj = copy.deepcopy(self)
        if isinstance(other, WabbitHDF5file):
            equal_grid = self.compareGrid(other)
            equal_attr = self.compareAttr(other)
            if not equal_grid:
                print(bcolors.FAIL + f"WARNING: Grids are not equal, operation interpolated for non-consistent blocks- This might take a while" + bcolors.ENDC)
                grid_interpolator = other.create_interpolator()
            if not equal_attr:
                print(bcolors.WARNING + f"ERROR: Attributes are not equal" + bcolors.ENDC)
                return None

            new_obj = copy.deepcopy(self)
            # blocks are not structured similarly so we have to apply blockwise
            for i_blocks in range(new_obj.total_number_blocks):
                i_other = other.get_block_id(self.block_treecode_num[i_blocks], self.level[i_blocks])
                if (i_other != -1):
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] / other.blocks[i_other, :]
                else:
                    new_obj.blocks[i_blocks, :] = new_obj.blocks[i_blocks, :] / \
                        other.interpolate_block(new_obj.blocks[i_blocks, :], new_obj.coords_origin[i_blocks], new_obj.coords_spacing[i_blocks], grid_interpolator)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            new_obj.blocks[:] /= other
        return new_obj

    # given a level and treecode, give me the block ID
    def get_block_id(self, treecode, level):
        return self.tc_find.get((treecode, level), -1)


    # returns sort list where the elements are in ascending order for first treecode and then level
    def sort_list(self, do_resorting):
        combined_list = list(zip(self.block_treecode_num, self.level, range(self.total_number_blocks)))
        # Sort the list based on treecode first, then level
        sorted_combined_list = sorted(combined_list, key=lambda x: (x[0], x[1]))

        # if we want to resort, we do that now
        if do_resorting:
            self.blocks_order = np.array(sorted_combined_list)[:,2]
            self.coords_origin, self.coords_spacing = self.coords_origin[self.blocks_order, :], self.coords_spacing[self.blocks_order, :]
            self.level, self.block_treecode_num = self.level[self.blocks_order], self.block_treecode_num[self.blocks_order]
            if len(self.blocks) > 0: self.blocks = self.blocks[self.blocks_order, :]
            if len(self.lgt_ids) > 0: self.lgt_ids = self.lgt_ids[self.blocks_order]
            if len(self.procs) > 0: self.procs = self.procs[self.blocks_order]
            if len(self.refinement_status) > 0: self.refinement_status = self.refinement_status[self.blocks_order]
            if len(self.block_treecode) > 0: self.block_treecode = self.block_treecode[self.blocks_order, :]
            self.tc_find = {(tc, lvl): idx for idx, (tc, lvl) in enumerate(zip(self.block_treecode_num, self.level))}

        # Extract the sorted indices
        return np.array(sorted_combined_list)[:,2]


    # check if logically two objects are considered to be close to equal
    def isClose(self, other, verbose=True, logger=None, return_norm=False):
        # check if grid attributes are equal
        attr_similarity = self.compareAttr(other, logger=logger)
        if not attr_similarity:
            if verbose:
                text_now = bcolors.FAIL + f"ERROR: Grid attributes are note qual" + bcolors.ENDC
                if logger==None: print(text_now)
                else: logger.info(text_now)
            return False
        
        # check if grids are equal
        grid_similarity = self.compareGrid(other, logger=logger)
        grid_interpolator = ()
        if not grid_similarity:
            if verbose: 
                text_now = bcolors.FAIL + f"ERROR: Grid is not equal, interpolating the difference. This might take a while" + bcolors.ENDC
                if logger==None: print(text_now)
                else: logger.info(text_now)    
            grid_interpolator = other.create_interpolator()
            # return False
        
        # check key values of data
        max1, min1, mean1, squares1 = keyvalues(self.domain_size, self.coords_spacing, self.blocks)
        max2, min2, mean2, squares2 = keyvalues(other.domain_size, other.coords_spacing, other.blocks)
        
        #------------------------------------------------------------------------------
        # compute L2 norm of difference, but only if the grids are identical
        #------------------------------------------------------------------------------
        diff_L2 = 0.0
        diff_LInf = 0.0
        norm_L2 = 0.0
        error_L2 = np.nan
        error_LInf = np.nan

        for i in range(self.total_number_blocks):
            # normalization is norm of data1
            norm_L2 = norm_L2 + np.linalg.norm( np.ndarray.flatten(self.blocks[i,:]) )
        
            # L2 and Linfty difference, last point is ignored as this is a redundant point
            j = other.get_block_id(self.block_treecode_num[i], self.level[i])
            if j != -1:
                if self.dim==2:
                    diff_L2 = diff_L2 + np.linalg.norm( np.ndarray.flatten(self.blocks[i,:-1,:-1]-other.blocks[j,:-1,:-1]) )
                    diff_LInf = np.max([diff_LInf, np.linalg.norm( np.ndarray.flatten(self.blocks[i,:-1,:-1]-other.blocks[j,:-1,:-1]) , ord=np.inf)])
                else:
                    diff_L2 = diff_L2 + np.linalg.norm( np.ndarray.flatten(self.blocks[i,:-1,:-1,:-1]-other.blocks[j,:-1,:-1,:-1]) )
                    diff_LInf = np.max([diff_LInf, np.linalg.norm( np.ndarray.flatten(self.blocks[i,:-1,:-1,:-1]-other.blocks[j,:-1,:-1,:-1]) , ord=np.inf)])
            else:
                if self.dim==2:
                    diff_block = self.blocks[i, :-1,:-1] - other.interpolate_block(self.blocks[i, :-1,:-1], self.coords_origin[i], self.coords_spacing[i], grid_interpolator)
                else:
                    diff_block = self.blocks[i, :-1,:-1,:-1] - other.interpolate_block(self.blocks[i, :-1,:-1,:-1], self.coords_origin[i], self.coords_spacing[i], grid_interpolator)
                diff_L2 = diff_L2 + np.linalg.norm( np.ndarray.flatten(diff_block) )
                diff_LInf = np.max([diff_LInf, np.linalg.norm( np.ndarray.flatten(diff_block) , ord=np.inf)])
                    
            if norm_L2 >= 1.0e-10:
                # relative error
                error_L2 = diff_L2 / norm_L2
                error_LInf = diff_LInf  # not normed
            else:
                # absolute error
                error_L2 = diff_L2
                error_LInf = diff_LInf
        
        if verbose:
            text_now = f"First : max={max1:12.5e}, min   ={min1:12.5e}, mean={mean1:12.5e}, squares={squares1:12.5e}"
            if logger==None: print(text_now)
            else: logger.info(text_now)
            text_now = f"Second: max={max2:12.5e}, min   ={min2:12.5e}, mean={mean2:12.5e}, squares={squares2:12.5e}"
            if logger==None: print(text_now)
            else: logger.info(text_now)
            text_now = f"Error : L2 ={error_L2:12.5e}, LInfty={error_LInf:12.5e}"
            if logger==None: print(text_now)
            else: logger.info(text_now)
            if error_L2 <= 1.0e-13: 
                text_now = "GREAT: The files can be deemed as equal"
                if logger==None: print(text_now)
                else: logger.info(text_now)
            else:
                text_now = bcolors.FAIL + "ERROR: The files do not match" + bcolors.ENDC
                if logger==None: print(text_now)
                else: logger.info(text_now)
        if return_norm: return error_L2 <= 1.0e-13, error_L2, error_LInf
        return error_L2 <= 1.0e-13


    # check if grid is equal or not, with fractional we compute the fraction of treecodes which are different
    def compareGrid(self, other, fractional=False, verbose=True, logger=None):
        if self.total_number_blocks != other.total_number_blocks:
            if verbose:
                text_now = bcolors.FAIL + f"ERROR: We have a different number of blocks - {self.total_number_blocks} vs {other.total_number_blocks}" + bcolors.ENDC
                if logger==None: print(text_now)
                else: logger.info(text_now)
            return False
                
        mismatch_count = 0
        # Iterate through self once, checking against the dictionary
        # ToDo: Treecode might be similar but with different max level, this is not taken into account here
        for i in range(self.block_treecode_num.shape[0]):
            if (self.block_treecode_num[i], self.level[i]) not in other.tc_dict:
                mismatch_count += 1
                if not fractional:
                    if verbose:
                        text_now = bcolors.FAIL + f"ERROR: treecode not matching" + bcolors.ENDC
                        if logger==None: print(text_now)
                        else: logger.info(text_now)
                    return False  # Early exit if not computing fractional and a mismatch is found
        
        if fractional:
            return 1 - mismatch_count / self.block_treecode_num.shape[0]
        else:
            return True
        
    # check if position and other details about the grid are equal
    # A grid is uniquely defined by its dimension, block size, domain size
    # The individual grid partition is uniquely defined by the number of blocks, treecode and level arrays
    def compareAttr(self, other, verbose=True, logger=None):
        # check global grid attributes
        if self.dim != other.dim:
            if verbose:
                text_now = bcolors.FAIL + f"ERROR: Grids are not in the same dimension, we have to leave the matrix - {self.dim} vs {other.dim}" + bcolors.ENDC
                if logger==None: print(text_now)
                else: logger.info(text_now)
            return False
        if not np.all(self.block_size[1:self.dim] == other.block_size[1:other.dim]):
            if verbose:
                text_now = bcolors.FAIL + f"ERROR: Block sizes are different - {self.block_size} vs {other.block_size}" + bcolors.ENDC
                if logger==None: print(text_now)
                else: logger.info(text_now)
            return False
        if np.any(self.domain_size[1:self.dim] != other.domain_size[1:other.dim]):
            if verbose:
                text_now = bcolors.FAIL + f"ERROR: Domain size is different - {self.domain_size} vs {other.domain_size}" + bcolors.ENDC
                if logger==None: print(text_now)
                else: logger.info(text_now)
            return False
        return True
    
    # check if objects are at the same time instant, pretty simple but why not have a function for it
    # round_digits is needed as floating points do not like direct comparisons
    def compareTime(self, other, verbose=True, round_digits=12, logger=None):
        similar_time = (np.round(self.time, round_digits) == np.round(other.time, round_digits))
        if not similar_time and verbose:
            text_now = bcolors.FAIL + f"ERROR: times are not equal" + bcolors.ENDC
            if logger==None: print(text_now)
            else: logger.info(text_now)
        return similar_time
    
    # try to parse the variable name from the file it was read in with
    def var_from_filename(self, verbose=True):
        fname = self.orig_file
        # only get actualy filename not path
        if "/" in fname: fname = fname.split("/")[-1]
        # var name is in beginning
        fname = fname.split("_")[0]


        # basic fail-safe: check if there was any splitting done at all
        if fname == self.orig_file:
            if verbose: print(bcolors.FAIL + f"ERROR: I do not know how to parse the variable name from this file-name - {self.orig_file}" + bcolors.ENDC)
            return []
        else: return fname
    
    # try to parse the variable name from the file it was read in with
    def time_from_filename(self, out_str=True, verbose=True):
        fname = self.orig_file
        # only get actualy filename not path
        if "/" in fname: fname = fname.split("/")[-1]
        # var name is in beginning
        if not "_" in fname:
            if verbose: print(bcolors.FAIL + f"ERROR: I do not know how to parse the variable name from this file-name - {self.orig_file}" + bcolors.ENDC)
            return
        fname = fname.split("_")[1]

        # basic fail-safe: check if there was any splitting done at all
        if fname == self.orig_file:
            if verbose: print(bcolors.FAIL + f"ERROR: I do not know how to parse the variable name from this file-name - {self.orig_file}" + bcolors.ENDC)
            return
        
        # output either as str or number, convention is that last 6 digits are after dot
        if out_str: return fname
        else: return int(fname) / 1e6

    def get_min_max_level(self):
        '''
            Retrieve the minimum and maximum level in the grid
        '''
        return np.min(self.level), np.max(self.level)
        

    # interpolate the values of a block given by its position and spacing to compute norm difference with different grids
    def interpolate_block(self, block, coords_origin, coords_spacing, interpolator):
        block_out = np.zeros_like(block)

        # compute ends of blocks to find in which block a point lays
        self_coords_end = self.coords_origin + self.coords_spacing * (np.array(self.blocks.shape[1:]) -1)

        for i_x in range(block.shape[0]):
            for i_y in range(block.shape[1]):
                if len(block.shape) == 3: range_z = range(block.shape[2])
                else: range_z = [0]
                for i_z in range_z:
                    # coord of this point
                    if len(block.shape) == 3: i_xyz = np.array((i_x, i_y, i_z))
                    else: i_xyz = np.array((i_x, i_y))
                    coords_point = coords_origin + i_xyz * coords_spacing
                    # find corresponding block
                    b_id = np.where(np.all(coords_point >= self.coords_origin, axis=1) & np.all(coords_point <= self_coords_end, axis=1))[0]
                    # interpolate in this block
                    if len(block.shape) == 3: block_out[i_x, i_y, i_z] = interpolator[b_id[0]](coords_point)
                    else: block_out[i_x, i_y] = interpolator[b_id[0]](coords_point)
        return block_out
    
    # create regular grid interpolators for every single block so that we can reuse it
    def create_interpolator(self):
        # compute ends of blocks to find in which block a point lays
        self_coords_end = self.coords_origin + self.coords_spacing * (np.array(self.blocks.shape[1:]) -1)

        interpolators = []
        for i_block in range(self.total_number_blocks):
            x_coords = []
            for i_dim in range(self.dim):
                x_coords.append(np.linspace(self.coords_origin[i_block, i_dim], self_coords_end[i_block, i_dim], self.blocks.shape[1+i_dim]))

            interpolators.append(RegularGridInterpolator(x_coords, self.blocks[i_block, :]))
        
        return interpolators
    
    # in order to create analytical results on the same grid we might want to replace all values with that to study it
    def replace_values_with_function(self, function):
        for i_block in range(self.total_number_blocks):
            block = self.blocks[i_block, :]
            for i_x in range(block.shape[0]):
                for i_y in range(block.shape[1]):
                    if len(block.shape) == 3: range_z = range(block.shape[2])
                    else: range_z = [0]
                    for i_z in range_z:
                        # coord of this point
                        if len(block.shape) == 3: i_xyz = np.array((i_x, i_y, i_z))
                        else: i_xyz = np.array((i_x, i_y))
                        coords_point = self.coords_origin[i_block] + i_xyz * self.coords_spacing[i_block]
                        
                        # replace function values
                        fun_val = function(coords_point)

                        if len(block.shape) == 3: self.blocks[i_block, i_x, i_y, i_z] = fun_val
                        else: self.blocks[i_block, i_x, i_y] = fun_val


#
def block_level_distribution( wabbit_obj: WabbitHDF5file ):
    """ Read a 2D/3D wabbit file and return a list of how many blocks are at the different levels
    """
    counter = np.zeros(wabbit_obj.max_level)

    # fetch level for each block and count
    for i_level in range(1, wabbit_obj.max_level+1):
        counter[i_level - 1] = np.sum(wabbit_obj.level == i_level)
    return counter.astype(int)



def block_proc_level_distribution( wabbit_obj: WabbitHDF5file ):
    """ Read a 2D/3D wabbit file and return a 2D list of how many blocks are at the different levels for each proc
    """
    counter = np.zeros([np.max(wabbit_obj.procs), wabbit_obj.max_level])

    # loop over all blocks and add counter of proc and level
    for i_block in range(1, wabbit_obj.total_number_blocks):
        counter[wabbit_obj.procs[i_block]-1, wabbit_obj.level[i_block]-1] += 1
    return counter.astype(int)



def read_wabbit_hdf5_dir(dir):
    """ Read all h5 files in directory dir.

    Return time, x0, dx, box, data, treecode.

    Use data["phi"][it] to reference quantity phi at iteration it
    """
    import numpy as np
    import re
    import ntpath
    import os

    it=0
    w_obj_list, var_list = [], []
    # we loop over all files in the given directory
    for file in os.listdir(dir):
        # filter out the good ones (ending with .h5)
        if file.endswith(".h5"):
            # from the file we can get the fieldname
            fieldname=re.split('_',file)[0]
            print(fieldname)
            wabbit_obj = WabbitHDF5file()
            wabbit_obj.read(os.path.join(dir, file))
            w_obj_list.append(wabbit_obj)
            var_list.append(wabbit_obj.var_from_filename())
    return w_obj_list, var_list



def add_convergence_labels(dx, er, ax=None):
    """
    This generic function adds the local convergence rate as nice labels between
    two datapoints of a convergence rate (see https://arxiv.org/abs/1912.05371 Fig 3E)
    
    Input:
    ------
    
        dx: np.ndarray 
            The x-coordinates of the plot
        dx: np.ndarray 
            The y-coordinates of the plot
    
    Output:
    -------
        Print to figure.
    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()

    for i in range(len(dx)-1):
        x = 10**( 0.5 * ( np.log10(dx[i]) + np.log10(dx[i+1]) ) )
        y = 10**( 0.5 * ( np.log10(er[i]) + np.log10(er[i+1]) ) )
        order = "%2.1f" % ( convergence_order(dx[i:i+1+1],er[i:i+1+1]) )
        ax.text(x, y, order, horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='w', alpha=0.75, edgecolor='none'), fontsize=7 )


def convergence_order(N, err):
    """ This is a small function that returns the convergence order, i.e. the least
    squares fit to the log of the two passed lists.
    """
    import numpy as np

    if len(N) != len(err):
        raise ValueError('Convergence order args do not have same length')

    A = np.ones([len(err), 2])
    B = np.ones([len(err), 1])
    # ERR = A*N + B
    for i in range( len(N) ) :
        A[i,0] = np.log(N[i])
        B[i] = np.log(err[i])

    x, residuals, rank, singval  = np.linalg.lstsq(A, B, rcond=None)

    return x[0]

def logfit(N, err):
    """ This is a small function that returns the logfit, i.e. the least
    squares fit to the log of the two passed lists.
    """
    import numpy as np

    if len(N) != len(err):
        raise ValueError('Convergence order args do not have same length')

    A = np.ones([len(err), 2])
    B = np.ones([len(err), 1])
    # ERR = A*N + B
    for i in range( len(N) ) :
        A[i,0] = np.log10(N[i])
        B[i] = np.log10(err[i])

    x, residuals, rank, singval  = np.linalg.lstsq(A, B, rcond=None)

    return x

def linfit(N, err):
    """ This is a small function that returns the linfit, i.e. the least
    squares fit to the log of the two passed lists.
    """
    import numpy as np

    if len(N) != len(err):
        raise ValueError('Convergence order args do not have same length')

    A = np.ones([len(err), 2])
    B = np.ones([len(err), 1])
    # ERR = A*N + B
    for i in range( len(N) ) :
        A[i,0] = N[i]
        B[i] = err[i]

    x, residuals, rank, singval  = np.linalg.lstsq(A, B, rcond=None)

    return x


def plot_wabbit_dir(d, **kwargs):
    import glob

    files = glob.glob(os.path.join(d,'*.h5'))
    files.sort()
    for file in files:
        plot_wabbit_file(file, **kwargs)


# extract from numerical treecode the digit at a specific level
def tc_get_digit_at_level(tc_b, level, max_level=21, dim=3):
    result = (tc_b // (2**(dim*(max_level - level))) % (2**dim))
    if isinstance(tc_b, np.ndarray):
        return result.astype(int)
    else:
        return int(result)

# set for numerical treecode the digit at a specific level
def tc_set_digit_at_level(tc_b, digit, level, max_level=21, dim=3):
    # compute digit that is currently there
    result = (tc_b // (2**(dim*(max_level - level))) % (2**dim))
    # subtract old digit, add new one
    tc_b += (-result + digit) * (2**(dim*(max_level - level))) 

    if isinstance(tc_b, np.ndarray):
        return tc_b.astype(int)
    else:
        return int(tc_b)

# similar to encoding function in wabbit, ixyz are one-based ([1,1] or [1,1,1] give TC 0)
def tc_encoding(ixyz, level=21, max_level=21, dim=3):
    if np.any(np.array(ixyz) < 1) or np.any(np.array(ixyz) > 2**level):
        print(f"Invalid coordinates {ixyz}, ensure they are 1 <= ixyz <= {2**level}")
        return -1
    # one-based encoding, so subtract the 1
    ixyz_n = np.array(ixyz) - 1
    tc = 0
    p_arr = [1, 0, 2]  # y and x are interchanged for tc encoding
    # Loop over all bits set in index
    for i_dim in range(len(ixyz_n)):
        p_dim = p_arr[i_dim]
        for i_level in range(int(ixyz_n[i_dim]).bit_length()):
            bit = (ixyz_n[i_dim] >> i_level) & 1
            if bit:
                # max for if one forgets to set the level
                tc += bit << ((i_level) * dim + p_dim + max(max_level-level, 0)*dim)
    # print(f"{ixyz} - {tc}")
    return tc

def tc_decoding(treecode, level=None, dim=3, max_level=21):
    """
    Obtain block position coordinates from numerical binary treecode.
    Works for 2D and 3D. Considers each digit and adds their level-shift to each coordinate.
    
    Parameters:
    - treecode: int, treecode value
    - level: int, level at which to encode, can be negative to set from max_level
    - dim: int, dimension (2 or 3), defaults to 3
    - max_level: int, max level possible, should be set after params%Jmax
    
    Returns:
    - ix: list of int, block position coordinates
    """
    
    n_level = max_level if level is None else level
    if n_level < 0: n_level = max_level + n_level + 1
    
    # 1-based
    ix = [1] * dim
    
    for i_level in range(n_level):
        for i_dim in range(dim):
            shift = (i_level + max_level - n_level) * dim + i_dim
            bit = (treecode >> shift) & 1
            ix[i_dim] += bit << i_level
    
    ix[0], ix[1] = ix[1], ix[0]
    
    return ix

# get string representation of binary treecode
def tc_to_str(tc_b, level, max_level=21, dim=3):
    tc_str = ""
    for i_level in np.arange(level)+1:
        tc_str += str(tc_get_digit_at_level(tc_b, i_level, max_level, dim))
    return tc_str

# take level and numerical treecode and convert to treecode array
def tcb_level_2_tcarray(tc_b, level, max_level=21, dim=3):
    tc_array = np.zeros((tc_b.shape[0], max_level))
    # extract number of each level
    # level <= i_level ensures -1 values are inserted for unset levels
    for i_level in np.arange(max_level)+1:
        tc_array[:, i_level-1] = tc_get_digit_at_level(tc_b, i_level, max_level=max_level, dim=dim) - (level <= i_level)
    return tc_array

# extract level from treecode array, assume field
def tca_2_level(tca):
    level = np.zeros(tca.shape[0]).astype(int)
    # increase level by one if number is not -1
    for i_level in range(0, tca.shape[1]):
        level[tca[:, i_level] != -1] += 1
    return level


def tca_2_tcb(tca, dim=3, max_level=21):
    """
    Convert array-treecode (TCA= treecode_array, 1D array for a single treecode) to binary treecode (=TCB, single number for a single treecode).
    Given is an array of array-treecodes (of size [N,J]), which is converted 
    to [N] TCB values, which is a single number for each treecode (and not an 1D array). 
    
    This routine possibly also works for a single treecode (in which case a single int is returned).
    NOTE: this functionality may not work - I don't understand JBs python magic. -TE 05092025
    
    NOTE: This routine seems to work only correctly if max_level=21 is given.
    """
    # allocation
    tcb = np.zeros(tca.shape[0])
    
    # maximum level is length of TCA array
    Jmax = tca.shape[1]
    
    # loop over the levels (j); in the TCA array, each level has its own int
    for j in range(0, Jmax):
        # extract the digit for all TCA in the array in one go (slice)
        tc_digit = tca[:, j]
        
        # increase level where digit is not -1
        tcb[tc_digit != -1] += tc_digit[tc_digit != -1] * 2**( dim*(max_level-1 - j) )
        
    if isinstance(tcb, np.ndarray):
        # multiple treecodes used, so return array of binary treecodes
        return tcb.astype(int)
    else:
        # single treecode used, return just one int
        return int(tcb)

# given a treecode tc, return its level
def treecode_level( tc ):
    level = 0
    for k in range(len(tc)):        
        if (tc[k] >= 0):
            level += 1
        else:
            break
    return(level)


# return coords_origin from treecode
def treecode2origin( tc, max_level=21, dim=3, domain_size=[1,1,1] ):
    """
    Convert binary treecode to block origin x0=(x,y,z).

    Parameters
    ----------
    tc : Numerical tree code (binary), a single number describing the block's treecode
    max_level : The maximum level present in the grid (and not the maximum allowable level)
    dim : Data dimensionality, can be 2 or 3. The default is 2.
    domain_size : The default is [1,1,1].

    Returns
    -------
    origin: the 2D/3D vector that describes the origin of the block in space.
    
    Notes
    -----
    
    The defaults do not make much sense: in most cases, you'll have to pass at least
    the max_level parameter of the grid. You need to figure out how many active levels
    the tree has before calling this routine.

    """
    
    origin = np.zeros(dim)
    for i_l in np.arange(max_level)+1:
        spacing = domain_size / (2**i_l)

        digit = tc_get_digit_at_level(tc, i_l, max_level, dim)
        spacing_fac = np.array([(digit//2)%2, digit%2, (digit//4)%2])
        origin += spacing[:dim] * spacing_fac[:dim]
    return origin[::-1]

# return treecode from coords_origin
def origin2treecode( origin, max_level=21, dim=3, domain_size=[1,1,1] ):
    treecode=0
    origin_n = np.copy(origin[::-1])
    for i_l in np.arange(max_level)+1:
        spacing = domain_size / (2**i_l)

        digit = (origin_n[0]>=spacing[0])*2 + (origin_n[1]>=spacing[1])*1
        if dim==3: digit +=(origin_n[2]>=spacing[2])*4
        treecode = tc_set_digit_at_level(treecode, digit, level=i_l, max_level=max_level, dim=dim)

        truth_array = [(origin_n[0]>=spacing[0]), (origin_n[1]>=spacing[1]), (origin_n[2]>=spacing[2])]
        origin_n[:dim] -= spacing[:dim] * truth_array[:dim]
    
    if treecode > 2**(dim*(max_level+1)) or treecode < 0:
        print(f"Invalid treecode created: {treecode}")
    return treecode

# return coords_spacing from level
def level2spacing( level, dim=3, block_size=[21,21,21], domain_size=[1,1,1] ):
    return np.array(domain_size[:dim] / (np.array(block_size[:dim])-1))/(2**level)

# return coords_spacing from level
def spacing2level( spacing, block_size=[21,21,21], domain_size=[1,1,1] ):
    if np.any(spacing[:] == 0): return np.infty
    level = np.log2(domain_size[0]/((block_size[0]-1)*spacing[0]))
    if level - np.rint(level) > 0.1:
        print(f"Level deviates much from integer: {level}")
    return np.rint( level ).astype(int)

# copy from fortran, find neighbors
def adjacent_neighbor(treecode, direction, level=None, dim=3, max_level=32):
    """
    Obtain neighbour in 3D for given direction with numerical binary treecode.
    
    Parameters:
        treecode (int): Numerical treecode in.
        direction (int): Direction for neighbor search.
           Each digit in variable direction represents one of the dimensions. Digits can take following values:
              1:-x, 2:+x, 3:-y, 4:+y, 5:-z, 6:+z
        level (int, optional): Level at which to encode, can be negative to set from max_level.
        dim (int, optional): Dimension (2 or 3), defaults to 3.
        max_level (int, optional): Max level possible.
        
    Returns:
        int: Numerical treecode out (neighbor).
    """
    
    if level is None: level = max_level
    if level < 0: level = max_level + level + 1
    
    if direction == 1:
        dir_sign = -1
        dir_fac = 2
    elif direction == 2:
        dir_sign = 1
        dir_fac = 2
    elif direction == 3:
        dir_sign = -1
        dir_fac = 1
    elif direction == 4:
        dir_sign = 1
        dir_fac = 1
    elif direction == 5:
        dir_sign = -1
        dir_fac = 4
    elif direction == 6:
        dir_sign = 1
        dir_fac = 4
    else:
        raise ValueError("Unknown direction")
    
    treecode_neighbor = 0
    
    for i_l in np.arange(level, 0, -1):
        digit_last = tc_get_digit_at_level(treecode, i_l, max_level, dim)
        
        treecode_neighbor += ((digit_last + dir_sign * dir_fac) % (2 * dir_fac) + (digit_last // (2 * dir_fac)) * 2 * dir_fac) * 2**(dim*(max_level-i_l))
        
        dir_sign = ((digit_last // dir_fac + (2 - dir_sign) // 2) & 1) * dir_sign
        
        if dir_sign == 0:
            treecode_neighbor += treecode // 2**(dim*(max_level-i_l+1)) * 2**(dim*(max_level-i_l+1))
            break
    return treecode_neighbor

# this is some hackery, we take a block, walk in x-direction for highest level (smallest block) and check if there is a block
# if that is the case then our block has to be of that level for leaf-grids, we increase the level to check until we find the first block
def level_from_treecode(tc, tc_array, max_level=21, dim=3):
    for i_l in np.arange(max_level, 0, -1):
        # build treecode for neighbor in +x direction for this block
        tc_n = adjacent_neighbor(tc, direction=2, level=i_l, max_level=max_level, dim=dim)
        # check if this block exists, if so then we have found the right level and can return
        if tc_n in tc_array: return i_l

        
        
# for a treecode list, return max and min level found
def get_min_max_level( treecode ):

    min_level = 99
    max_level = -99
    N = treecode.shape[0]
    for i in range(N):
        tc = np.asarray( treecode[i,:].copy(), dtype=int)
        level = treecode_level(tc)

        min_level = min([min_level,level])
        max_level = max([max_level,level])

    return min_level, max_level

#
def plot_1d_cut( wabbit_obj: WabbitHDF5file, y ):    
    if wabbit_obj.dim != 2:
        raise ValueError("Sadly, we do this only for 2D fields right now")
        
    if y < 0.0 or y >= wabbit_obj.domain_size[1]:
        raise ValueError(f"Sadly, you request a y value out of the domain: {y} not in 0-{wabbit_obj.domain_size[1]}")
        
    y_found = []
    # first check if any blocks contain the y value at all 
    for i in range(wabbit_obj.total_number_blocks):
        y_vct = np.arange(wabbit_obj.block_size[0])*wabbit_obj.coords_spacing[i,0] + wabbit_obj.coords_origin[i,0]
        
        if np.min( np.abs(y_vct-y) ) < wabbit_obj.coords_spacing[i,1]/2.0:
            iy = np.argmin( np.abs(y_vct-y) )
            y_found.append( y_vct[iy] )
            
    print(y_found)
    y_new = y_found[0]
    print('snapped to y=%f' % (y_new))
    
    x_values, f_values = [],[]
    
    for i in range(wabbit_obj.total_number_blocks):
        
        x_vct = np.arange(wabbit_obj.block_size[1])*wabbit_obj.coords_spacing[i,1] + wabbit_obj.coords_origin[i,1]
        y_vct = np.arange(wabbit_obj.block_size[0])*wabbit_obj.coords_spacing[i,0] + wabbit_obj.coords_origin[i,0]
        
        if np.min( np.abs(y_vct-y_new) ) < wabbit_obj.coords_spacing[i,0]/100.0:
            iy = np.argmin( np.abs(y_vct-y) )
            x_values.append( x_vct )
            f_values.append( wabbit_obj.blocks[i,iy,:].copy() )
            
    x_values = np.hstack(x_values)
    f_values = np.hstack(f_values)
    
    x_values, f_values = zip(*sorted(zip(x_values, f_values)))
           
    return x_values, f_values
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot(x_values, f_values, '-')

#
def plot_wabbit_file( wabbit_obj:WabbitHDF5file, savepng=False, savepdf=False, cmap='rainbow', caxis=None,
                     caxis_symmetric=False, title=True, mark_blocks=True, block_linewidth=1.0,
                     gridonly=False, contour=False, ax=None, fig=None, ticks=True,
                     colorbar=True, dpi=300, block_edge_color='k',
                     block_edge_alpha=1.0, shading='auto',
                     colorbar_orientation="vertical",
                     gridonly_coloring='mpirank', flipud=False, 
                     fileContainsGhostNodes=False, 
                     filename_png=None,
                     filename_pdf=None):
    """
    Read and plot a 2D wabbit file. Not suitable for 3D data, use Paraview for that.
    
    Input:
    ------
    
        file:  string
            file to visualize
        gridonly: bool
            If true, we plot only the blocks and not the actual, point data on those blocks.
        gridonly_coloring: string
            if gridonly is true we can still color the blocks. One of 'lgt_id', 'refinement-status', 'mpirank', 'level', 'file-index'
    
    """

    import numpy as np
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import h5py
    
    if filename_pdf is None:
        filename_pdf = wabbit_obj.orig_file.replace('h5','pdf')
    if filename_png is None:
        filename_png = wabbit_obj.orig_file.replace('h5','png')

    
    cb = []
    # read procs table, if we want to draw the grid only
    if gridonly:
        procs = wabbit_obj.procs

        if gridonly_coloring in ['refinement-status', 'refinement_status']:
            ref_status = wabbit_obj.refinement_status

        if gridonly_coloring == 'lgt_id':
            lgt_ids = wabbit_obj.lgt_ids

    # read data
    time = wabbit_obj.time
    x0 = wabbit_obj.coords_origin
    dx = wabbit_obj.coords_spacing
    box = wabbit_obj.domain_size
    data = wabbit_obj.blocks
    treecode_num = wabbit_obj.block_treecode_num
    level = wabbit_obj.level

    # get number of blocks and blocksize
    N, Bs = wabbit_obj.total_number_blocks, wabbit_obj.block_size

    # we need these lists to modify the colorscale, as each block usually gets its own
    # and we would rather like to have a global one.
    h, c1, c2 = [], [], []


    if fig is None:
        fig = plt.gcf()
        fig.clf()

    if ax is None:
        ax = fig.gca()

    # clear axes
    ax.cla()

    # if only the grid is plotted, we use grayscale for the blocks, and for
    # proper scaling we need to know the max/min level in the grid
    jmin, jmax = wabbit_obj.get_min_max_level()



    if gridonly:
        #----------------------------------------------------------------------
        # Grid data only (CPU distribution, level, or grid only)
        #----------------------------------------------------------------------
        cm = plt.cm.get_cmap(cmap)

        # loop over blocks and plot them individually
        for i in range(N):
            # draw some other qtys (mpirank, lgt_id or refinement-status)
            if gridonly_coloring in ['mpirank', 'cpu']:
                color = cm( procs[i]/max(procs) )

            elif gridonly_coloring in ['refinement-status', 'refinement_status']:
                color = cm((ref_status[i]+1.0) / 2.0)

            elif gridonly_coloring == 'level':
                level_now = level[i]
                if (jmax-jmin>0):
                    c = 0.9 - 0.75*(level_now-jmin)/(jmax-jmin)
                    color = [c,c,c]
                else:
                    color ='w'
                
                
            elif gridonly_coloring == 'file-index':
                color = cm( float(i)/float(N) )

                tag = "%i" % (i)
                x = Bs[1]/2*dx[i,1]+x0[i,1]
                if not flipud:
                    y = Bs[0]/2*dx[i,0]+x0[i,0]
                else:
                    y = box[0] - Bs[0]/2*dx[i,0]+x0[i,0]
                plt.text( x, y, tag, fontsize=6, horizontalalignment='center', verticalalignment='center')
                
            elif gridonly_coloring == 'lgt_id':
                color = cm( lgt_ids[i]/max(lgt_ids) )

                tag = "%i" % (lgt_ids[i])
                x = Bs[1]/2*dx[i,1]+x0[i,1]
                if not flipud:
                    y = Bs[0]/2*dx[i,0]+x0[i,0]
                else:
                    y = box[0] - Bs[0]/2*dx[i,0]+x0[i,0]
                
                plt.text( x, y, tag, fontsize=6, horizontalalignment='center', verticalalignment='center')
                
            elif gridonly_coloring == 'treecode':
                    color = 'w'
                    tag = tc_to_str(treecode_num[i], level[i], wabbit_obj.max_level, wabbit_obj.dim)

                    print(tag)
                                        
                    x = Bs[1]/2*dx[i,1]+x0[i,1]
                    if not flipud:
                        y = Bs[0]/2*dx[i,0]+x0[i,0]
                    else:
                        y = box[0] - Bs[0]/2*dx[i,0]+x0[i,0]
                    plt.text( x, y, tag, fontsize=6, horizontalalignment='center', verticalalignment='center')
                
                
            elif gridonly_coloring == 'none':
                color = 'w'
            else:
                raise ValueError(bcolors.FAIL + "ERROR! The value for gridonly_coloring is unkown" + bcolors.ENDC)

            # draw colored rectangles for the blocks
            if not fileContainsGhostNodes:                
                ax.add_patch( patches.Rectangle( (x0[i,1],x0[i,0]), (Bs[1]-1)*dx[i,1], (Bs[0]-1)*dx[i,0],
                                            fill=True, edgecolor=block_edge_color, alpha=block_edge_alpha,
                                            facecolor=color))
            else:
                ax.add_patch( patches.Rectangle( (x0[i,1]+6*dx[i,1],x0[i,0]+6*dx[i,0]), (Bs[1]-1-6*2)*dx[i,1], (Bs[0]-1-6*2)*dx[i,0],
                                            fill=True, edgecolor=block_edge_color, alpha=block_edge_alpha,
                                            facecolor=color))
            cb = None
            hplot = None

    else:
        #----------------------------------------------------------------------
        # Plot real data.
        #----------------------------------------------------------------------
        # loop over blocks and plot them individually
        for i in range(N):

            if not flipud :
                [X, Y] = np.meshgrid( np.arange(Bs[0])*dx[i,0]+x0[i,0], np.arange(Bs[1])*dx[i,1]+x0[i,1])
            else:
                [X, Y] = np.meshgrid( box[0]-np.arange(Bs[0])*dx[i,0]+x0[i,0], np.arange(Bs[1])*dx[i,1]+x0[i,1])

            # copy block data
            block = data[i,:,:].copy().transpose()

            if contour:
                # --- contour plot ----
                hplot = ax.contour( Y, X, block, [0.1, 0.2, 0.5, 0.75] )

            else:
                # --- pseudocolor plot ----
                #hplot=plt.pcolormesh(X,X,X)
                hplot = ax.pcolormesh( Y, X, block, cmap=cmap, shading=shading )

                # use rasterization for the patch we just draw
                hplot.set_rasterized(True)

            # unfortunately, each patch of pcolor has its own colorbar, so we have to take care
            # that they all use the same.
            h.append(hplot)
            a = hplot.get_clim()
            c1.append(a[0])
            c2.append(a[1])

            if mark_blocks:
                # empty rectangle to mark the blocks border
                ax.add_patch( patches.Rectangle( (x0[i,1],x0[i,0]), (Bs[1]-1)*dx[i,1], (Bs[0]-1)*dx[i,0],
                                                fill=False, edgecolor=block_edge_color, alpha=block_edge_alpha,
                                                linewidth=block_linewidth))

        # unfortunately, each patch of pcolor has its own colorbar, so we have to take care
        # that they all use the same.
        if caxis is None:
            if not caxis_symmetric:
                # automatic colorbar, using min and max throughout all patches
                for hplots in h:
                    hplots.set_clim( (min(c1),max(c2))  )
            else:
                    # automatic colorbar, but symmetric, using the SMALLER of both absolute values
                    c= min( [abs(min(c1)), max(c2)] )
                    for hplots in h:
                        hplots.set_clim( (-c,c)  )
        else:
            # set fixed (user defined) colorbar for all patches
            for hplots in h:
                hplots.set_clim( (min(caxis),max(caxis))  )

        # add colorbar, if desired
        cb = None
        if colorbar:
            cb = plt.colorbar(h[0], ax=ax, orientation=colorbar_orientation)

    if title:
        # note Bs in the file and Bs in wabbit are different (uniqueGrid vs redundantGrid
        # definition)
        plt.title( "t=%f Nb=%i Bs=(%i,%i)" % (time,N,Bs[1]-1,Bs[0]-1) )


    if not ticks:
        ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

        ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        right=False,      # ticks along the bottom edge are off
        left=False,         # ticks along the top edge are off
        labelleft=False) # labels along the bottom edge are off

#    plt.xlim([0.0, box[0]])
#    plt.ylim([0.0, box[1]])

    ax.axis('tight')
    ax.set_aspect('equal')
    fig.canvas.draw()

    if not gridonly:
        if savepng:
            plt.savefig( filename_png, dpi=dpi, transparent=False, bbox_inches='tight' )

        if savepdf:
            plt.savefig( filename_pdf, bbox_inches='tight', dpi=dpi )
    else:
        if savepng:
            plt.savefig( filename_png.replace('.h5','-grid.png'), dpi=dpi, transparent=False, bbox_inches='tight' )

        if savepdf:
            plt.savefig( filename_pdf.replace('.h5','-grid.pdf'), bbox_inches='tight' )

    return ax,cb,hplot


#
def prediction1D( signal1, order=4 ):
    import numpy as np

    N = len(signal1)

    signal_interp = np.zeros( [2*N-1] ) - 7

#    function f_fine = prediction1D(f_coarse)
#    % this is the multiresolution predition operator.
#    % it pushes a signal from a coarser level to the next higher by
#    % interpolation

    signal_interp[0::2] = signal1

    a =   9.0/16.0
    b =  -1.0/16.0

    signal_interp[1]  = (5./16.)*signal1[0] + (15./16.)*signal1[1] -(5./16.)*signal1[2] +(1./16.)*signal1[3]
    signal_interp[-2] = (1./16.)*signal1[-4] -(5./16.)*signal1[-3] +(15./16.)*signal1[-2] +(5./16.)*signal1[-1]

    for k in range(1,N-2):
        signal_interp[2*k+1] = a*signal1[k]+a*signal1[k+1]+b*signal1[k-1]+b*signal1[k+2]

    return signal_interp



#
def command_on_each_hdf5_file(directory, command):
    """
    This routine performs a shell command on each *.h5 file in a given directory!

    Input:
        directory - directory with h5 files
        command - a shell command which specifies the location of the file with %s
                    Example command = "touch %s"

    Example:
    command_on_each_hdf5_file("/path/to/my/data", "/path/to/wabbit/wabbit-post --dense-to-sparse --eps=0.02 %s")
    """
    import re
    import os
    import glob

    if not os.path.exists(directory):
        bcolors.err("The given directory does not exist!")

    files = glob.glob(directory+'/*.h5')
    files.sort()
    for file in files:
        c = command % file
        os.system(c)

# for wabbit files we always save the time in the name as 6 digits before . and 6 digits after
# this function takes a float and does the same
def time2wabbitstr(time):
    return f"{time:013.6f}".replace(".", "")










def dense_matrix(  x0, dx, data, level, dim=2, verbose=True ):
    """ Convert a WABBIT grid to a full dense grid in a single matrix.

    We asssume here that interpolation has already been performed, i.e. all
    blocks are on the same (finest) level (= an equidistant grid in block decomposition)

    Returns the full matrix and the domain size. Note matrix is periodic and can
    directly be compared to FLUSI-style results ``(x=L * 0:nx-1/nx)``.
    
    Parameters:
    ======
    
    x0, dx : (Nb,3) 
        Arrays with origin and spacing of each block
        
    level : (Nb)
        Level of each block. Should be all the same.
        
    data : (Nb, Bsz, Bsy, Bsx) 
        Array with data decomposed in blocks. 
    
    
    Returns:
    ========
        field (nz,ny,nx), box
    
    
    A tale of two kingdoms:
    =======================
    Ordering:
        
        - The **WABBIT** kingdom stores data as [ix,iy,iz,icomponent,ib], for a scalar (and this is the usual case in python, since
          wabbit stores each component in a separate file), this reduces to [ix,iy,iz,ib]. Last index is block index.
    
        - The **PYTHON** kingdom reads those arrays using HDF5, and the resulting order is [ib,iz,iy,ix]. First index is block index.
    
    Block size:
    
        - The **WABBIT** kingdom defines the block size BS as the interior nodes of a block. As of 09/2025, this is an EVEN number.
          When storing HDF5 output, we noticed paraview has gaps between the blocks - this is tedious. Hence, WABBIT kingdom now
          stores BS+1 point, including the first ghost node. This is an ODD number.
    
        - The **PYTHON** kingdom reads in the file, and it defines the block size as BS_PYTHON = BS_WABBIT + 1, which is the size of the array. This is an ODD 
          number. Both numbers [BS_PYTHON=33 & BS_WABBIT=32] describe THE SAME GRID AND SAME BLOCK SIZE. Confused yet?
    
    
    """
    
    # number of blocks
    Nb = data.shape[0]
    # size of each block
    Bs = np.asarray(data.shape[1:])

    # check if all blocks are on the same level or not
    jmin, jmax = np.min(level), np.max(level)
    if jmin != jmax:
        raise ValueError(bcolors.FAIL + "ERROR! not an equidistant grid yet..." + bcolors.ENDC)

    
    if dim==2:
        # in both uniqueGrid and redundantGrid format, a redundant point is included (it is the first ghost 
        # node in the uniqueGrid format!)
        nx = [int( np.sqrt(Nb)*(Bs[d]-1) ) for d in range(np.size(Bs))]
    else:
        nx = [int( round( (Nb)**(1.0/3.0)*(Bs[d]-1) ) ) for d in range(np.size(Bs))]


    # all spacings should be the same - it does not matter which one we use.    
    # domain size. Note is is returned as (Lx,Ly,Lz). 
    box = nx[dim+1:None:-1] * dx[0, dim+1:None:-1] # note odd nomenclature in python: https://stackoverflow.com/questions/17610096/reverse-indexing-in-python 
    
    if verbose:
        print("Nblocks                      :", (Nb))
        print("Bs (XYZ)                     :", Bs[::-1])
        print("Spacing (XYZ)                :", dx[0,::-1])
        print("Domain  (XYZ)                :", box)
        print("Dense field resolution (XYZ) :", nx[::-1] )

    if dim==2:
        # allocate target field
        field = np.zeros(nx)

        for i in range(Nb):
            # get starting index of block
            ix0 = int( round(x0[i,0]/dx[i,0]) )
            iy0 = int( round(x0[i,1]/dx[i,1]) )

            # copy block content to data field. Note we skip the last points, which
            # are the redundant nodes (or the first ghost node).
            field[ ix0:ix0+Bs[0]-1, iy0:iy0+Bs[1]-1 ] = data[i, 0:-1 ,0:-1]

    else:
        # allocate target field
        field = np.zeros([nx[0],nx[1],nx[2]])

        for i in range(Nb):
            # get starting index of block
            ix0 = int( round(x0[i,0]/dx[i,0]) )
            iy0 = int( round(x0[i,1]/dx[i,1]) )
            iz0 = int( round(x0[i,2]/dx[i,2]) )

            # copy block content to data field. Note we skip the last points, which
            # are the redundant nodes (or the first ghost node).
            field[ ix0:ix0+Bs[0]-1, iy0:iy0+Bs[1]-1, iz0:iz0+Bs[2]-1 ] = data[i, 0:-1, 0:-1, 0:-1]
    
    return(field, box)



#
def wabbit_error_vs_flusi(fname_wabbit, fname_flusi, norm=2, dim=2, verbose=True):
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
        print('I think due to fft2usample, this routine works only in 2D')
        raise ValueError

    # read in flusi's reference solution
    time_ref, box_ref, origin_ref, data_ref = insect_tools.read_flusi_HDF5( fname_flusi, verbose=verbose)
    if verbose: print(data_ref.shape)
    ny = data_ref.shape[1]

    # wabbit field to be analyzed: note has to be full already
    wabbit_obj = WabbitHDF5file()
    wabbit_obj.read(fname_wabbit, verbose=verbose)
    x0 = wabbit_obj.coords_origin
    dx = wabbit_obj.coords_spacing
    data = wabbit_obj.blocks
    level = wabbit_obj.level
    Bs = data.shape[1]
    Jflusi = (np.log2(ny/(Bs-1)))
    if verbose: print("Flusi resolution: %i %i %i so desired level is Jmax=%f" % (data_ref.shape[0], data_ref.shape[2], data_ref.shape[2], Jflusi) )

    if dim==2:
        # squeeze 3D flusi field (where dim0 == 1) to true 2d data
        data_ref = data_ref[0,:,:].copy().transpose()
        box_ref = box_ref[1:2].copy()

    # convert wabbit to dense field
    data_dense, box_dense = dense_matrix( x0, dx, data, level, dim, verbose=verbose)
    
    if data_dense.shape[0] < data_ref.shape[0]:
        # both datasets have different size
        s = int( data_ref.shape[0] / data_dense.shape[0] )
        data_ref = data_ref[::s, ::s].copy()
        raise ValueError(bcolors.FAIL + "ERROR! Both fields are not a the same resolutionn" + bcolors.ENDC)

    if data_dense.shape[0] > data_ref.shape[0]:
        bcolors.warn("WARNING! The reference solution is not fine enough for the comparison! UPSAMPLING!")
        import fourier_tools
        data_ref = fourier_tools.fft2_resample( data_ref, data_dense.shape[1], verbose=True)

    err = np.ndarray.flatten(data_ref-data_dense)
    exc = np.ndarray.flatten(data_ref)

    err = np.linalg.norm(err, ord=norm) / np.linalg.norm(exc, ord=norm)
    if verbose: print( "error was e=%e" % (err) )

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
    # box = box[1:]
    
    data_flusi = np.squeeze(data_flusi).T
    Bs = field_shape_to_bs(data_flusi.shape,level)
    
    dense_to_wabbit_hdf5(data_flusi, fname_wabbit, Bs, box, time, dtype=dtype)


#
def dense_to_wabbit_hdf5(ddata, fname, Bs, box_size = None, time = 0, iteration = 0, dtype=np.float64):

    """
    Given a dense matrix and a block size Bs, this code cuts the dense data into blocks and stores them to a WABBIT compatible
    file. No manipulation of data is done; it's exactly the same data just organized differently, on an equidistant block-based grid.

    Input:
    ======
    ddata : (Nz, Ny, Nx)
        Dense 2D/3D array of the data you want to write to a file. This array is assumed to hold unique values, i.e. only points with x=[0,L) (excluding L), as
        would be the default in any periodic code. This subroutine periodizes the data itself, do not pass data with redundant first/last line.
     
    fname : string
        filename for output WABBIT h5 file
    
    Bs : [Bsx, Bsy, Bsz]
        number of grid points per block is a 2D/3D dimensional array with Bs[0] being the number of grid points in x 
        direction etc. The data size in each dimension has to be dividable by Bs.
                    
    Optional Input:
    =============
        - box_size... 2D/3D array of the size of your box [Lx, Ly, Lz]
        - time    ... time of the data
        - iteration ... iteration of the time snappshot
        
        
    
    A tale of two kingdoms:
    =======================
    Ordering:
        
        - The **WABBIT** kingdom stores data as [ix,iy,iz,icomponent,ib], for a scalar (and this is the usual case in python, since
          wabbit stores each component in a separate file), this reduces to [ix,iy,iz,ib]. Last index is block index.
    
        - The **PYTHON** kingdom reads those arrays using HDF5, and the resulting order is [ib,iz,iy,ix]. First index is block index.
    
    Block size:
    
        - The **WABBIT** kingdom defines the block size BS as the interior nodes of a block. As of 09/2025, this is an EVEN number.
          When storing HDF5 output, we noticed paraview has gaps between the blocks - this is tedious. Hence, WABBIT kingdom now
          stores BS+1 point, including the first ghost node. This is an ODD number.
    
        - The **PYTHON** kingdom reads in the file, and it defines the block size as BS_PYTHON = BS_WABBIT + 1, which is the size of the array. This is an ODD 
          number. Both numbers [BS_PYTHON=33 & BS_WABBIT=32] describe THE SAME GRID AND SAME BLOCK SIZE. Confused yet?

    """
    dim   = ddata.ndim
    Nsize = np.asarray(ddata.shape)
    level = 0
    Bs    = np.asarray(Bs)# make sure Bs is a numpy array
    # See tale of two kingdoms
    Bs    = Bs[::-1] # flip Bs such that Bs=[BsY, BsX] the order is the same as for Nsize=[Ny,Nx]
        
    
    #########################################################
    # do some initial checks on the input data
    # 1) check if the size of the domain is given
    if box_size is None:
        box = np.ones(dim)
    else:
        box = np.asarray(box_size)

    # If a single bs (a single integer) is given, we use that value isotropically
    # (in all directions):
    if (type(Bs) is int):
        Bs = [Bs]*dim
        
    # 2) check if number of lattice points is block decomposable
    # loop over all dimensions
    for d in range(dim):
        # check if Block is devidable by Bs
        if (np.remainder(Nsize[d], Bs[d]-1) == 0):
            if(is_power2(Nsize[d]//(Bs[d]-1))):
                level = int(max(level, np.log2(Nsize[d]/(Bs[d]-1))))
            else:
                bcolors.err("Number of Intervals must be a power of 2!")
        else:
            bcolors.err("datasize must be multiple of Bs!")
            
    # 3) check dimension of array:
    if dim < 2 or dim > 3:
        bcolors.err("dimensions are wrong")
    #########################################################

    # Periodize input data (by adding one more row/column to it which repeats the
    # first row/column)
    data = np.zeros(Nsize+1, dtype=dtype)
    if dim == 2:
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

  
    print('~~~~~~~~~~~~~~~~~~~~dense_to_wabbit_hdf5~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('  ')
    print('Input data shape:  (ZYX)  ', ddata.shape)
    print('Input data dim:           ', dim)
    print('Target block size: (ZYX)  ', Bs, "(PYTHON notation)")
    print('Resulting level:          ', level)
    

    # number of intervals in each dimension
    Nintervals = [ int(2**level) ]*dim  # note [val]*3 means [val, val , val]
    Lintervals = box[:dim]/np.asarray(Nintervals)
    # See A tale of two kingdoms
    Lintervals = Lintervals[::-1]

    x0, dx, treecode, block_data = [], [], [], []
    
    if dim == 3:
        for ibx in range(Nintervals[0]):
            for iby in range(Nintervals[1]):
                for ibz in range(Nintervals[2]):
                    x0.append([ibx, iby, ibz]*Lintervals)
                    dx.append(Lintervals/(Bs-1))

                    # extract this blocks data from the full (dense) matrix
                    lower = [ibx, iby, ibz]* (Bs - 1)
                    lower = np.asarray(lower, dtype=int)
                    upper = lower + Bs

                    block_data.append(data[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
                    
                    # Compute this blocks treecode (TCA - array treecode)
                    #
                    # IT IS COMPLETELY UNCLEAR WHY THE HELL THIS IS THE CORRECT ORDERING: YZX it makes no sense but it works
                    treecode.append(blockindex2treecode([iby, ibz, ibx], 3, level)) # YZX
                    
    else:
        for ibx in range(Nintervals[0]):
            for iby in range(Nintervals[1]):
                x0.append([ibx, iby]*Lintervals)
                dx.append(Lintervals/(Bs-1))
                
                lower = [ibx, iby]* (Bs - 1)
                lower = np.asarray(lower, dtype=int)
                upper = lower + Bs
                
                treecode.append(blockindex2treecode([ibx, iby], 2, level))
                block_data.append(data[lower[0]:upper[0], lower[1]:upper[1]])
                
                raise ValueError("THis code part is untested. See strange ordering in 3D case")

    x0, dx     = np.asarray(x0, dtype=dtype), np.asarray(dx, dtype=dtype)
    treecode   = np.asarray(treecode, dtype=int)
    block_data = np.asarray(block_data, dtype=dtype)

    # convert array treecode to binary treecode
    treecode_num = tca_2_tcb(treecode, dim=dim, max_level=level)
    # blocks are dense so level is the same everywhere
    level = tca_2_level(treecode)

    w_obj = WabbitHDF5file()
    w_obj.fill_vars(box, block_data, treecode_num, level, time, iteration, max_level=level[0])
    w_obj.coords_origin = x0
    w_obj.coords_spacing = dx
    w_obj.write(fname)
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
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

    # flip again before returning array
    return treecode[::-1]















# debugging tests
if __name__ == "__main__":
    # state1 = WabbitHDF5file()
    # state1.read("../WABBIT/TESTING/jul/vorabs_000002000000.h5")

    # state_2D = WabbitHDF5file()
    # state_2D.read("../WABBIT/TESTING/jul/test_2D/phi_000000250000.h5")
    
    # print(block_level_distribution(state1))
    # print(np.transpose(block_proc_level_distribution(state1)))

    # print(tc_to_str(np.array(tc_encoding([2,5,1], max_level=5, dim=3)), level=5, max_level=5, dim=3))

    # state_test = WabbitHDF5file()
    # state_test.read("../WABBIT/phi_000000250000.h5")
    # state_test.replace_values_with_function(lambda xyz: INICOND_convdiff_blob(xyz, blob_pos=[0.75, 0.75]))

    # state_test.write("../WABBIT/correct-phi_000002500000.h5")


    pass






















