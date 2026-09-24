#!/usr/bin/env python3
import h5py, os, sys, argparse, glob, time, numpy as np, json
try:
  from mpi4py import MPI
  mpi_size = MPI.COMM_WORLD.Get_size()
  mpi_parallel = mpi_size > 1
  mpi_rank = MPI.COMM_WORLD.Get_rank()
  print(f"Running code in parallel on {mpi_size} cores")
except:
  print("Running code in serial")
  mpi_parallel = False
  mpi_rank = 0
  mpi_size = 1
try:
  from vtkmodules.vtkCommonCore import (
    vtkDoubleArray,
    vtkBitArray,
  )
  from vtkmodules.vtkCommonDataModel import (
      vtkHyperTreeGrid,
      vtkHyperTreeGridNonOrientedCursor,
  )
  import vtk
  loaded_vtk = True
except:
  print("Could not load vtk modules so we cannot create htg files")
  loaded_vtk = False

sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools
import bcolors


# Progress bar function

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int) going until total-1
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = int(100 * ((iteration + 1) / total))
    filled_length = length * (iteration + 1) / total
    element_c = str(int(10*(filled_length - int(filled_length)))) if filled_length != int(filled_length) else ''
    bar = fill * int(filled_length) + element_c + '-' * (length - int(filled_length) - (element_c != ''))
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '\r')
    # Print New Line on Complete
    if iteration == total - 1: 
        print()


'''
Prune the grid, i.e. remove Blocks with constant values if the values are not deviating up to a given tolerance
this has to be done before merging

This function takes in lists for all blocks and returns new lists where only the desired blocks are added
'''
def prune_grid(wobj, block_id_o, coords_origin_o, coords_spacing_o, level_o, treecode_o, sub_tree_size_o, sub_tree_positions_o, tolerance):
  block_id, coords_origin, coords_spacing, treecode, level, sub_tree_size, sub_tree_positions = [], [], [], [], [], [], []
  for i_block in block_id_o:
    id_block = i_block[0]  # assume that we did not group blocks yet
    block_id_now = wobj.get_block_id(wobj.block_treecode_num[id_block], wobj.level[id_block])
    block_values_now = wobj.block_read(block_id_now)

    # check if values deviate less than relative tolerance from mean
    mean_block = np.mean(block_values_now)
    removed_blocks = 0
    if np.all(np.abs(block_values_now - mean_block) < tolerance):
    # if np.all(block_values_now <= 0):
      # if so, we do not add this block to the new list
      continue
    else:
      # if not, we add the block to the new list
      block_id.append(block_id_o[id_block])
      coords_origin.append(coords_origin_o[id_block])
      coords_spacing.append(coords_spacing_o[id_block])
      level.append(level_o[id_block])
      treecode.append(treecode_o[id_block])
      sub_tree_size.append(sub_tree_size_o[id_block])
      sub_tree_positions.append(sub_tree_positions_o[id_block])
    
  return block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_positions


"""
Takes a wabbit object and tries to merge all blocks, where all sister blocks are available.

This function takes in lists for all blocks and returns new lists where only the desired blocks are added

The function loops over all blocks, finds all sisters and checks if they have the same sub-tree structure (in case a previous merge happened)
In case all of them are fitting, they are merged to one block with double the blocksize in each direction
We do this for all blocks, if blocks can be merged twice, this function needs to be called again (and again, until the grid is converged)
"""
def merge_sisters(block_id_o, coords_origin_o, coords_spacing_o, level_o, treecode_o, sub_tree_size_o, sub_tree_positions_o, max_level, dim=3):
  # dictionary used for lookup of blocks
  tc_find = {(tc, lvl): idx for idx, (tc, lvl) in enumerate(zip(treecode_o, level_o))}

  block_id, coords_origin, coords_spacing, treecode, level, sub_tree_size, sub_tree_positions = [], [], [], [], [], [], []
  id_merged = []

  # loop over all blocks, find all sisters and merge if they have the same sub-tree structure
  for i_b in range(len(block_id_o)):
    # extract it's position and from it's blocksize it's merge level
    i_treecode, i_level = treecode_o[i_b], level_o[i_b]
    all_sisters = True
    id_sisters, position_sisters = np.array([]), np.zeros([0,3])
    level_set = np.log2(len(block_id_o[i_b]))//dim
    id_find_0 = []
    # now loop over all sisters and try to find them
    for i_sister in range(2**dim):
      tc_sister = wabbit_tools.tc_set_digit_at_level(i_treecode, i_sister, i_level-level_set, max_level=max_level, dim=dim)
      id_find = tc_find.get((tc_sister, i_level),-1)
      if i_sister == 0: id_find_0 = id_find  # first id needs to be saved as it will be the master block of the merged one
      if id_find == -1:  # block not found - we do not merge
        all_sisters = False
        break
      if np.any(sub_tree_size_o[id_find] != sub_tree_size_o[i_b]):  # blocks do not have the same subtree structure so are not on the same level - we do not merge
        all_sisters = False
        break
      id_sisters = np.append(id_sisters, block_id_o[id_find]).astype(int)  # append ids of the sister to the list of the merged block, as we read in the data later
      position_shift = (np.array(wabbit_tools.tc_decoding(i_sister,level=1, max_level=1,dim=3))-1)*(np.array(sub_tree_size_o[i_b])).astype(int)  # shift position according to position on highest level, so that we get the relative position in the merged block
      position_sisters = np.append(position_sisters, sub_tree_positions_o[id_find] + position_shift, axis=0)  # append relative positions of the sister to the list of the merged block
      if id_find_0 in id_merged: break
    # we have found all sisters and proceed with merging
    if all_sisters and id_find_0 not in id_merged:
      # search for meta-data from block with entry zero, these are appended as the data of the new block
      coords_origin.append(coords_origin_o[id_find_0])
      coords_spacing.append(coords_spacing_o[id_find_0])
      level.append(i_level)
      treecode.append(wabbit_tools.tc_set_digit_at_level(i_treecode, 0, i_level-level_set, max_level=max_level, dim=dim))
      sub_tree_new = np.array(sub_tree_size_o[id_find_0].copy())
      sub_tree_new[:dim] = 2*sub_tree_new[:dim]
      sub_tree_size.append(sub_tree_new)  # double the blocksize!
      sub_tree_positions.append(position_sisters)
      block_id.append(id_sisters)

      # append id of block 0 to an array of finished blocks, so that we do not merge some several times
      id_merged.append(id_find_0)
    elif not all_sisters:
      # we did not find all sisters or they do not share the same blocksize and just append the current
      coords_origin.append(coords_origin_o[i_b])
      coords_spacing.append(coords_spacing_o[i_b])
      level.append(i_level)
      treecode.append(i_treecode)
      sub_tree_size.append(sub_tree_size_o[i_b])
      sub_tree_positions.append(sub_tree_positions_o[i_b])
      block_id.append(block_id_o[i_b])
  return block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_positions


"""
Takes a wabbit object and tries to merge all blocks, where all sister blocks are available.

This function takes in lists for all blocks and returns new lists where only the desired blocks are added

The function loops over all blocks, finds the neighbor in one direction and checks if they have the same sub-tree structure (in case a previous merge happened)
In case the other dimensions of the blocks are fitting, the neighbour is appended in that direction and the blocksize is increased
We do this for one direction only, so for the others we have to call the function again
"""
def merge_directional(block_id_o, coords_origin_o, coords_spacing_o, level_o, treecode_o, sub_tree_o, sub_tree_positions_o, max_level, dim=3, direction=0):
  # dictionary used for lookup of blocks
  tc_find = {(tc, lvl): idx for idx, (tc, lvl) in enumerate(zip(treecode_o, level_o))}

  block_id, coords_origin, coords_spacing, treecode, level, sub_tree_size, sub_tree_positions = [], [], [], [], [], [], []
  id_merged = []

  # loop over all blocks, find the neighbor in one direction and merge if they have the same sub-tree structure
  for i_b in range(len(block_id_o)):
    # skip this block if it has already been merged
    if i_b in id_merged: continue

    # extract it's position
    i_treecode, i_level = treecode_o[i_b], level_o[i_b]
    correct_neighbor = True
    id_n, position_n = np.array([]), np.zeros([0,3])
    i_sub_tree_size = np.array(sub_tree_o[i_b].copy())  # copy the original block size
    # let's add this block itself to the final list
    id_n = np.append(id_n, block_id_o[i_b]).astype(int)  # append block indices
    position_n = np.append(position_n, sub_tree_positions_o[i_b], axis=0)  # append positions of the block

    # now lets loop in direction and always try to find new blocks until we cannot find no more
    position_shift = np.array([0,0,0])
    while correct_neighbor:
      position_shift[direction] = i_sub_tree_size[direction]  # shift in the given direction to where the neighbor should be

      idx_b = wabbit_tools.tc_decoding(i_treecode,level=i_level, max_level=max_level,dim=dim)
      idx_n = idx_b + position_shift[:dim]  # shift indices in the given direction
      # there is a special case where the neighboring block is outside the periodic domain, then we do not proceed
      if idx_n[direction] > 2**i_level:
        correct_neighbor = False
        break
      else:
        tc_n = wabbit_tools.tc_encoding(idx_n, level=i_level, max_level=max_level, dim=dim)  # encode neighbor to it's treecode
        id_find = tc_find.get((tc_n, i_level),-1)
        if id_find == -1 or id_find in id_merged:  # block not found or already treated - we do not merge
          correct_neighbor = False
          break
        elif not level_o[id_find] == i_level:  # blocks are not on the same level - we do not merge
          correct_neighbor = False
          break
        elif not np.all(np.delete(sub_tree_o[i_b], direction) == np.delete(sub_tree_o[id_find], direction)):  # blocks do not have the same subtree structure in other directions - we do not merge
          correct_neighbor = False
          break
        else:
          id_n = np.append(id_n, block_id_o[id_find]).astype(int)  # append block ids
          position_n = np.append(position_n, sub_tree_positions_o[id_find] + position_shift, axis=0)  # append positions of the neighbor
          i_sub_tree_size[direction] += sub_tree_o[id_find][direction]  # increase the block size in the given direction
          id_merged.append(id_find)  # append id of the merged block to the list of merged blocks
    
    # we have found all neighbors and proceed with merging
    coords_origin.append(coords_origin_o[i_b])
    coords_spacing.append(coords_spacing_o[i_b])
    level.append(i_level)
    treecode.append(i_treecode)
    sub_tree_size.append(i_sub_tree_size)
    sub_tree_positions.append(position_n)
    block_id.append(id_n)
    
  
  return block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_positions


"""
Shared helpers to embed a block's own data into the vtkhdf image's (X, Y, Z) space.

A wabbit block always stores its data along its own "local" axes in the fixed order
(local-x, local-y, local-z) - or, for 2D data, (local-x, local-y, thickness), where
"thickness" is a dummy, size-1 (size-2 for PointData) stand-in for the direction that is
not part of the 2D plane the data lives in.

`axis_map` says which global vtkhdf axis (0=X, 1=Y, 2=Z) each local axis is embedded into.
For 3D data this is always the identity [0, 1, 2]. For 2D data it depends on plane_2D, e.g.
for a slice living in the XZ-plane, local-y is embedded into global Z and the dummy
thickness axis into global Y - see where axis_map is built in hdf2vtkhdf() for the full table.
This must stay in sync with how Origin/Spacing/WholeExtent are constructed for each plane.

vtkhdf/HDF5 stores per-block arrays with numpy shape (Z, Y, X) - reversed relative to how
Origin/Spacing/WholeExtent are given (X, Y, Z). These two helpers apply axis_map and then
that reversal, once, so the permutation logic is not duplicated at every write site.
"""
def local_axes_to_numpy_shape(local_sizes, axis_map):
  global_sizes = [None, None, None]
  for local_axis, global_slot in enumerate(axis_map):
    global_sizes[global_slot] = local_sizes[local_axis]
  return tuple(global_sizes[::-1])


def local_axes_to_numpy_slices(local_slices, axis_map):
  global_slices = [None, None, None]
  for local_axis, global_slot in enumerate(axis_map):
    global_slices[global_slot] = local_slices[local_axis]
  return tuple(global_slices[::-1])


def local_array_to_numpy(local_order_array, axis_map):
  """
  local_order_array must have its first 3 axes in local (x, y, thickness/z) order (same
  convention as local_axes_to_numpy_shape/_slices above); any further trailing axes (e.g. a
  vector-component axis) are left untouched. Transposes those first 3 axes the same way
  local_axes_to_numpy_shape/_slices permute sizes/slices, so that the RESULT actually has the
  shape those functions promised.

  This is needed because plain numpy broadcasting is not enough here: it only ever inserts
  missing axes on the LEFT. That accidentally "worked" before axis_map existed, because the
  thickness axis of a 2D block always sat at numpy axis 0 (the XY-plane case). For XZ/YZ planes
  the thickness axis can end up in the middle or at the end, where implicit broadcasting cannot
  place it - so we must explicitly transpose the source data into the right axis order instead
  of relying on broadcasting to paper over a shape mismatch.
  """
  inv_axis_map = [None, None, None]
  for local_axis, global_slot in enumerate(axis_map): inv_axis_map[global_slot] = local_axis
  transpose_order = [inv_axis_map[2 - numpy_axis] for numpy_axis in range(3)]
  transpose_order += list(range(3, local_order_array.ndim))  # keep any trailing axes (e.g. vector components) as-is
  return np.transpose(local_order_array, transpose_order)


def block_data_to_local_order(block_data, dim):
  """
  Raw wabbit block data (as returned by block_read) is stored axis-reversed - [z,y,x] for 3D
  data, [y,x] for 2D - the same quirk noted for coords_origin/coords_spacing. This undoes that
  reversal so the array's axes are in local (x, y, [z]) order. For 2D data it then appends the
  dummy, size-1 thickness axis, so the result always has exactly 3 axes in local
  (x, y, thickness/z) order and is ready to be passed into local_array_to_numpy().
  """
  local_order = np.transpose(block_data, axes=list(range(dim))[::-1])
  if dim == 2: local_order = local_order[..., np.newaxis]
  return local_order


def hdf2vtkhdf(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", scalars=False, split_levels=False, merge=True, prune_tolerance=None, grid2field=None, data_type="CellData", exclude_prefixes=[], include_prefixes=[], origin_translate=[0,0,0], scale=1, plane_2D="XY", low_memory=False):
  """
  Create a multi block dataset from the available data
    w_obj        - Required  : Object representeing the wabbit data or List of objects
                               list - objs are on same grid at same time and represent different variables and will be combined
    save_file    - Optional  : save string
    verbose      - Optional  : verbose flag
    save_mode    - Optional  : how to encode data - "ascii", "binary" or "appended"
    scalars      - Optional  : should vectors be identified or all included as scalars?
    split_levels - Optional  : For full-tree grids blocks are overlapping. This option entangles the blocks level-wise by giving them an offset
    merge        - Optional  : This option tries to merge sisters consecutively to minimize the amount of uniform blocks to load for paraview
    low_memory   - Optional  : How to fetch the actual block DATA for this timestep (the metadata in w_obj is always fully
                               loaded already - it is cheap). w_obj itself never holds the heavy block arrays; instead, a
                               local "block_cache" (one dict per variable, see below) is built right before it is needed and
                               discarded again once this function returns - so the long-lived w_obj objects (which callers
                               may keep around for many timesteps, e.g. in a {time: [w_obj, ...]} map) never accumulate
                               block data across timesteps.
                                 False (default, "mode B"): read each variable's ENTIRE block array once, in one bulk HDF5
                                   call (WabbitHDF5file.block_read_bulk(block_ids=None)). Fastest, needs one timestep's
                                   data (all variables) to fit in RAM - fine unless the grid is very large/high-res or many
                                   MPI ranks share a node's RAM.
                                 True ("mode C"): first figure out exactly which blocks THIS RANK needs (its slice of the
                                   merged grid below), then read only that subset per variable, again in one bulk call per
                                   variable (block_read_bulk(block_ids=[...])). Keeps memory down to ~1/mpi_size of the
                                   data, at the cost of one extra bookkeeping pass before reading.
                               Either way we avoid the slow path of calling block_read() once per sub-block (which opens
                               and closes the HDF5 file on every single call - by far the dominant cost for large grids).
  """
  ### Check if input contains only wabbit hdf5 files
  correct_input = False
  # if w_obj is only one obj, simply transcribe this one
  if isinstance(w_obj, wabbit_tools.WabbitHDF5file):
    w_obj_list = [w_obj]
    correct_input = True
  # if w_obj is list: create list of variables
  if isinstance(w_obj, list) and all(isinstance(elem, wabbit_tools.WabbitHDF5file) for elem in w_obj):
    w_obj_list = w_obj
    correct_input = True
  if not correct_input:
    print(bcolors.FAIL + "ERROR: Wrong input of wabbit state - input single object or list of objects" + bcolors.ENDC)
    return

  ### check if files have same mesh, attr and time
  w_main = w_obj_list[0]
  w_main.sort_list(do_resorting=True)
  for i_wobj in w_obj_list[1:]:
    i_wobj.sort_list(do_resorting=True)
    same_mesh = w_main.compareGrid(i_wobj)
    same_attrs = w_main.compareAttr(i_wobj)
    same_time = w_main.compareTime(i_wobj)

    if not all([same_mesh, same_attrs, same_time]):
      if verbose: print(bcolors.FAIL + f"ERROR: data are not similar and cannot be combined" + bcolors.ENDC)
      return
  
  ### create variable list
  # the option scalars forces the code to ignore the trailing x,y,z icons
  # and treat all fields as scalars
  # vector / scalar handling: if it ends on {x,y,z} the prefix indicates a vector
  # otherwise, we deal with a scalar field.
  s_names, s_ind, v_names, v_ind, p_names = [], [], [], [], []
  for i_n, i_wobj in enumerate(w_obj_list):
    f_now = i_wobj.var_from_filename()
    if f_now[-1] in ["x", "y", "z"] and not scalars:
      f_name_now = f_now[:-1]
      if f_name_now in exclude_prefixes: continue  # skip excluded prefixes
      if f_name_now not in include_prefixes and len(include_prefixes) > 0: continue  # skip not included prefixes
      if not f_name_now in v_names:
        v_names.append(f_name_now)
        v_ind.append([])  # empty handly vor all the indices
      v_ind[v_names.index(f_name_now)].append(i_n)  # append this to the index list
      p_names.append(f_now)
    else:
      if f_now in exclude_prefixes: continue  # skip excluded prefixes
      if f_now not in include_prefixes and len(include_prefixes) > 0: continue  # skip not included prefixes
      s_names.append(f_now)
      s_ind.append(i_n)
  # check if vectors are full elsewise add them as scalars
  for pre in v_names:
    if (pre+'x' in p_names and pre+'y' in p_names and pre+'z' in p_names and w_main.dim==3):
        if verbose and mpi_rank==0: print( f'   {pre} is a 3D vector (x,y,z)')
    elif (pre+'x' in p_names and pre+'y' in p_names and w_main.dim==2):
        if verbose and mpi_rank==0: print( f'   {pre} is a 2D vector (x,y)')
    else:
        if mpi_rank==0: print( f"   WARRNING: {pre} is not a vector (its x-, y- or z- component is missing..)")
        v_ind.remove(v_ind[v_names.index(pre)])
        v_names.remove( pre )
        # if pre+'x' in p_names: scalars.append(pre+'x')
        # if pre+'y' in p_names: scalars.append(pre+'y')
        # if pre+'z' in p_names: scalars.append(pre+'z')

  if grid2field == None: grid2field = []
#  print(f"    Adding {len(s_names)} scalar field{'s' if len(s_names) != 1 else ''} {'\"' + ', '.join(s_names) + '\"' if len(s_names) > 0 else ''}, {len(v_names)} vector field{'s' if len(v_names) != 1 else ''} {'\"' + ', '.join(v_names) + '\" ' if len(v_names) > 0 else ''}and {len(grid2field)} grid field{'s' if len(grid2field) != 1 else ''} {'\"' + ', '.join(grid2field) + '\" ' if len(grid2field) > 0 else ''}to vtkhdf file")

  ### prepare filename
  file_ending = '.vtkhdf'
  if save_file is None: save_file = w_main.orig_file.replace(".h5", file_ending)
  if not save_file.endswith(file_ending): save_file += file_ending

  # host deletes old file if it exists - needed for parallelization
  if mpi_rank == 0:
    if os.path.isfile(save_file): os.remove(save_file)  # a bit brute-force, maybe ask for deletion?
  if mpi_parallel: MPI.COMM_WORLD.Barrier()  # ensure that file is properly deleted before all processes continue#

  # now all processes open the file and create most important meta structures
  if not mpi_parallel: f =  h5py.File(save_file, 'w')
  else: f = h5py.File(save_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)

  vtkhdf_group = f.create_group('VTKHDF', track_order=True)
  vtkhdf_group.attrs.create('Type', 'PartitionedDataSetCollection'.encode('ascii'), dtype=h5py.string_dtype('ascii', len('PartitionedDataSetCollection')))
  vtkhdf_group.attrs.create('Version', np.array([2, 3], dtype='i8'))
  assembly_group = vtkhdf_group.create_group('Assembly', track_order=True)

  ### merging - tries to merge blocks, so that paraview needs to load less blocks
  # this is a high-level optimization, but neatly reduces the time to load for paraview
  # we do not merge inside the wabbit_obj, but rather represent it with lists, that we modify:
  #    coords_origin     - origin of the block (as before)
  #    coords_spacing    - spacing of the block (as before)
  #    level             - level of the block (as before)
  #    treecode          - treecode of the block (as before)
  #    sub_tree_size     - block_size of the block, as we merge blocks this can be larger than original block_size
  #    sub_tree_position - relative position of the sub-blocks in the merged block
  #    block_id          - list of lists of block ids that are contained in the merged block, so that we can read in the data
  
  # prepare all arrays, as we merge by looping over them and reducing them
  start_time = time.time()
  coords_origin, coords_spacing, level, treecode = w_main.coords_origin, w_main.coords_spacing, w_main.level, w_main.block_treecode_num
  sub_tree_size = [[1,1,1]]*w_main.total_number_blocks  # this one is important, as it contains the size of the merged block subtree
  sub_tree_position = np.zeros([w_main.total_number_blocks, 1, 3])  # it contains the position of the individual block ids in the subtree
  total_blocks = w_main.total_number_blocks
  block_id = [[i_b] for i_b in np.arange(w_main.total_number_blocks)]  # this is the block_id, for sub_tree this contains all different block_ids of the subtrees
  bs_o = np.array(w_main.block_size.copy())  # original blocksize
  # WABBIT blocks are stored as points (block_size points per direction), but vtkhdf ImageData
  # WholeExtent/CellData count CELLS between points, hence the -1. This "bs_o" (block size in
  # cells) is what all the extent/shape math below is built from. Note bs_o is always indexed in
  # NATURAL (x, y, z) order (0=x, 1=y, 2=z) - unlike coords_origin/coords_spacing/block data below,
  # which wabbit stores axis-reversed. For 2D data, dim=2 so only indices 0,1 are decremented;
  # index 2 stays at its untouched value of 1 and later acts as the dummy "thickness" axis.
  bs_o[:w_main.dim] -= 1

  # sort everything after treecode - potentially usefull for neighbour merging
  combined_list = list(zip(treecode, level, range(total_blocks)))
  id_sorted = [idx for _, _, idx in sorted(combined_list, key=lambda x: (x[0], x[1]))]
  coords_origin, coords_spacing = [coords_origin[i] for i in id_sorted], [coords_spacing[i] for i in id_sorted]
  level, treecode = [level[i] for i in id_sorted], [treecode[i] for i in id_sorted]
  sub_tree_size, block_id = [sub_tree_size[i] for i in id_sorted], [block_id[i] for i in id_sorted]
  # print to user
  if args.verbose and mpi_rank == 0:
    minutes, seconds = divmod(time.time() - start_time, 60)
    if minutes > 0: print(f"    Init blocks :          {total_blocks:7d} blocks, took {int(minutes)}m {seconds:04.1f}s")
    else: print(f"    Init blocks :          {total_blocks:7d} blocks, took {seconds:.2g}s")

  # pruning in case prune_tolerance is not None
  if prune_tolerance is not None:
    block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position = prune_grid(w_main, block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position, prune_tolerance)
    total_blocks = len(block_id)
    # print to user
    if args.verbose and mpi_rank == 0:
      minutes, seconds = divmod(time.time() - start_time, 60)
      if minutes > 0: print(f"    Prune blocks :         {total_blocks:7d} blocks, took {int(minutes)}m {seconds:04.1f}s")
      else: print(f"    Prune blocks :         {total_blocks:7d} blocks, took {seconds:.2g}s")

  # this is the actual sister merging loop, we loop until no new blocks are merged
  jmin, jmax = w_main.get_min_max_level()
  if merge: merge_blocks_it = jmax
  else: merge_blocks_it = 0
  for i_merge in range(merge_blocks_it):
    start_time = time.time()
    total_blocks_old = total_blocks
    # call merge function which does all the job
    block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position = merge_sisters(block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position, w_main.max_level, w_main.dim)
    total_blocks = len(block_id)
    # print to user
    if args.verbose and mpi_rank == 0:
      minutes, seconds = divmod(time.time() - start_time, 60)
      if minutes > 0: print(f"    Merged subtrees it {i_merge+1:2d}: {total_blocks:7d} blocks, took {int(minutes)}m {seconds:04.1f}s")
      else: print(f"    Merged subtrees it {i_merge+1:2d}: {total_blocks:7d} blocks, took {seconds:.2g}s")
    if total_blocks_old == total_blocks: break

  # now we are merging blocks in one direction, this is useful for highly adapted grids
  if merge:
    dir_names = ["x", "y", "z"]
    for i_dir in range(w_main.dim):
      start_time = time.time()
      total_blocks_old = total_blocks
      # call merge function which does all the job
      block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position = merge_directional(block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position, w_main.max_level, dim=w_main.dim, direction=i_dir)
      total_blocks = len(block_id)
      # print to user
      if args.verbose and mpi_rank == 0:
        minutes, seconds = divmod(time.time() - start_time, 60)
        if minutes > 0: print(f"    Merged in {dir_names[i_dir]}-dir:       {total_blocks:7d} blocks, took {int(minutes)}m {seconds:04.1f}s")
        else: print(f"    Merged in {dir_names[i_dir]}-dir:       {total_blocks:7d} blocks, took {seconds:.2g}s")

  ### figure out how a block's own local axes map onto the vtkhdf global (X, Y, Z) axes.
  # For 3D data this is trivially the identity - local (x, y, z) already ARE global (X, Y, Z).
  # For 2D data, only two of the three local axes carry real data (x, y); the third is the dummy
  # "thickness" axis (size 1, or 2 for PointData) standing in for the direction perpendicular to
  # the 2D plane. plane_2D picks which global axis each local axis lands on - this MUST stay in
  # sync with how Origin/Spacing/WholeExtent are built per-block below, and is used further down
  # to place each block's raw data into the vtkhdf array via local_axes_to_numpy_shape/_slices.
  if w_main.dim == 2:
    if plane_2D.upper() == "XY":   axis_map = [0, 1, 2]  # local x->X, local y->Y, thickness->Z
    elif plane_2D.upper() == "XZ": axis_map = [0, 2, 1]  # local x->X, local y->Z, thickness->Y
    elif plane_2D.upper() == "YZ": axis_map = [1, 2, 0]  # local x->Y, local y->Z, thickness->X
  else:
    axis_map = [0, 1, 2]  # 3D: local (x, y, z) already match global (X, Y, Z)

  ### collective loop creating the metadata - all processes need to do this
  start_time = time.time()
  data_group = [[]]*total_blocks
  for i_block in range(total_blocks):
    # for overfull CVS grids we have the option to split them into levels to make the overlay visible
    split_levels_add = (split_levels * (level[i_block]-1) * np.max(w_main.domain_size)) * 1.1

    # Create this block itself
    block_group = vtkhdf_group.create_group(f'Block{i_block}', track_order=True)

    # Add attributes to block
    if w_main.dim == 2:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      if plane_2D.upper() == "XY":
        origin = (np.append(coords_origin[i_block][::-1], 0) + np.array([0,split_levels_add,0]) + np.array(origin_translate)) * scale
        spacing = np.append(coords_spacing[i_block][::-1], 0)*scale
        extent = np.array([0, bs_o[0]*sub_tree_size[i_block][0], 0, bs_o[1]*sub_tree_size[i_block][1], 0, 1])
      elif plane_2D.upper() == "XZ":
        origin = (np.array([coords_origin[i_block][1], 0, coords_origin[i_block][0]]) + np.array([0,split_levels_add,0]) + np.array(origin_translate)) * scale
        spacing = np.array([coords_spacing[i_block][1], 0, coords_spacing[i_block][0]])*scale
        extent = np.array([0, bs_o[0]*sub_tree_size[i_block][0], 0, 1, 0, bs_o[1]*sub_tree_size[i_block][1]])
      elif plane_2D.upper() == "YZ":
        origin = (np.array([0, coords_origin[i_block][1], coords_origin[i_block][0]]) + np.array([0,split_levels_add,0]) + np.array(origin_translate)) * scale
        spacing = np.array([0, coords_spacing[i_block][1], coords_spacing[i_block][0]])*scale
        extent = np.array([0, 1, 0, bs_o[0]*sub_tree_size[i_block][0], 0, bs_o[1]*sub_tree_size[i_block][1]])
      block_group.attrs.create('Origin', origin, dtype='f8')
      block_group.attrs.create('Spacing', spacing, dtype='f8')
      block_group.attrs.create('WholeExtent', extent, dtype='i8')      
    else:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      block_group.attrs.create('Origin', (coords_origin[i_block][::-1] + np.array([0,0,split_levels_add]) + np.array(origin_translate)) * scale, dtype='f8')
      block_group.attrs.create('Spacing', coords_spacing[i_block][::-1] * scale, dtype='f8')
      block_group.attrs.create('WholeExtent', np.array([0, bs_o[0]*sub_tree_size[i_block][0], 0, bs_o[1]*sub_tree_size[i_block][1], 0, bs_o[2]*sub_tree_size[i_block][2]], dtype='i8'))
    block_group.attrs.create('Type', 'ImageData'.encode('ascii'), dtype=h5py.string_dtype('ascii', len('ImageData')))
    block_group.attrs.create('Version', np.array([2, 3], dtype='i8'))
    block_group.attrs.create('Index', i_block, dtype='i8')
    assembly_group[f'Block{i_block}'] = h5py.SoftLink(f'/VTKHDF/Block{i_block}')

    # Add block data
    if data_type == "CellData": data_group[i_block] = block_group.create_group('CellData', track_order=True)
    elif data_type == "PointData": data_group[i_block] = block_group.create_group('PointData', track_order=True)

    # Create empty dataset for scalars
    # bs_now is the block's size in LOCAL (x, y, thickness/z) order - same order as bs_o and
    # sub_tree_size. We only get the actual (Z, Y, X) numpy shape to allocate after routing it
    # through axis_map, since for 2D data the "thickness" axis may land on any of X/Y/Z depending
    # on plane_2D (see where axis_map is built above).
    bs_now = np.array(sub_tree_size[i_block]) * bs_o
    if data_type == "PointData": bs_now[:w_main.dim] += 1
    if w_main.dim == 2: bs_now[2] = 1 + (data_type == "PointData")
    bs_now_numpy_shape = local_axes_to_numpy_shape(bs_now, axis_map)
    for i_s, i_n in zip(s_names, s_ind): data_group[i_block].create_dataset(i_s, shape=bs_now_numpy_shape, dtype=np.float64)

    # Create empty datasets for vectors
    for i_v, i_n in zip(v_names, v_ind): data_group[i_block].create_dataset(i_v, shape=np.append(bs_now_numpy_shape, w_main.dim), dtype=np.float64)

    # Create empty datasets for grid2Field
    if grid2field is not None:
      for i_f in grid2field:
        # scalar fields
        if i_f in ["level", "treecode", "refinement_status", "procs", "lgt_ID"]: data_group[i_block].create_dataset(i_f, shape=bs_now_numpy_shape, dtype=np.float64)
        # vector fields
        if i_f in ["coords_origin", "coords_spacing"]: data_group[i_block].create_dataset(i_f, shape=np.append(bs_now_numpy_shape, w_main.dim), dtype=np.float64)
  if args.verbose and mpi_rank == 0:
    minutes, seconds = divmod(time.time() - start_time, 60)
    if minutes > 0: print(f"    Created metadata:                      took {int(minutes)}m {seconds:04.1f}s")
    else: print(f"    Created metadata:                      took {seconds:.2g}s")

  ### fetch the actual block DATA this rank will need for the loop below, in one bulk read per
  # variable file rather than one file-open per sub-block (see block_read_bulk() in wabbit_tools.py
  # for why that matters - it is by far the dominant cost for large grids). The result is a plain,
  # local list of dicts - "block_cache[i_n][i_b_now]" gives the same array block_read(i_b_now) on
  # w_obj_list[i_n] would have returned, just fetched all at once. This cache lives only for the
  # duration of this function call and is never attached to the w_obj_list objects themselves, so
  # a caller that keeps w_obj_list around for many timesteps (e.g. hdf2vtkhdf.py's time_process
  # dict) never accumulates block data in memory across timesteps.
  start_time = time.time()
  i_block_start, i_block_end = int(mpi_rank/mpi_size*total_blocks), int((mpi_rank+1)/mpi_size*total_blocks)
  # every variable index (both plain scalars and each component of a vector) that block data is
  # actually read from below - this is what we need to preload per w_obj_list entry
  all_var_ind = s_ind + [i_ndim for group in v_ind for i_ndim in group]
  if not low_memory:
    # mode B (default): read every needed variable file fully, once, in a single bulk HDF5 call
    block_cache = [w_obj_list[i_n].block_read_bulk() if i_n in all_var_ind else {} for i_n in range(len(w_obj_list))]
  else:
    # mode C (--low-memory): first walk through this rank's slice of the merged grid to collect
    # exactly which (per-variable) logical block ids it will touch - the same (treecode,level) ->
    # block id translation as the actual attach loop below, just to gather ids, not to read data -
    # then read only that subset per variable, again in a single bulk call each.
    needed_ids = [set() for _ in w_obj_list]
    for i_block in range(i_block_start, i_block_end):
      for i_merged_id in block_id[i_block]:
        tc_now, lvl_now = w_main.block_treecode_num[i_merged_id], w_main.level[i_merged_id]
        for i_n in all_var_ind:
          needed_ids[i_n].add(w_obj_list[i_n].get_block_id(tc_now, lvl_now))
    block_cache = [w_obj_list[i_n].block_read_bulk(needed_ids[i_n]) if i_n in all_var_ind else {} for i_n in range(len(w_obj_list))]
  if args.verbose and mpi_rank == 0:
    minutes, seconds = divmod(time.time() - start_time, 60)
    if minutes > 0: print(f"    Preloaded block data:                  took {int(minutes)}m {seconds:04.1f}s")
    else: print(f"    Preloaded block data:                  took {seconds:.2g}s")

  ### independent loop attaching the actual data - this is parallelized
  start_time = time.time()
  for i_block in range(i_block_start, i_block_end):
    rem_time = (total_blocks - i_block) * (time.time() - start_time) / (i_block + 1e-4*(i_block == 0))
    # Format remaining time in HH:MM:SS format
    hours, rem = divmod(rem_time, 3600)
    minutes, seconds = divmod(rem, 60)
    if verbose and mpi_rank==0 and i_block < int(total_blocks/mpi_size):
        print_progress_bar(i_block, int(total_blocks/mpi_size), prefix=f'    Processing data:', suffix=f'ETA: {int(hours)}h {int(minutes):02d}m { seconds:02.1f}s', length=20)

    # get celldatagroup
    if data_type == "CellData": vtkhdf_group[f'Block{i_block}']['CellData']
    elif data_type == "PointData": vtkhdf_group[f'Block{i_block}']['PointData']

    # Attach data for scalars - currently copying but maybe there is a more clever way
    id_now = block_id[i_block]
    bs_now = np.array(sub_tree_size[i_block]) * bs_o
    if data_type == "PointData": bs_now[:w_main.dim] += 1
    if w_main.dim == 2: bs_now[2] = 1 + (data_type == "PointData")
    bs_now_numpy_shape = local_axes_to_numpy_shape(bs_now, axis_map)
    for i_s, i_n in zip(s_names, s_ind):
      # block is composed of subtree
      data_append = np.zeros(bs_now_numpy_shape)
      for i_merged in range(len(id_now)):
        # translate id from main to this object as the block ids could be shuffled
        i_b_now = w_obj_list[i_n].get_block_id(w_main.block_treecode_num[id_now[i_merged]], w_main.level[id_now[i_merged]])
        j_block = block_cache[i_n][i_b_now]  # preloaded above (mode B: full array, mode C: this rank's subset)
        # get block position of this sub-octree, in the SAME local (x, y, thickness/z) order as bs_o
        b_id = sub_tree_position[i_block][i_merged].astype(int)
        if w_main.dim == 2: b_id = np.append(b_id, 0)
        # j_block itself is always stored [z,y,x] (3D) / [y,x] (2D) by wabbit, independent of
        # plane_2D - we build the write-window per local axis first, then let axis_map decide
        # which numpy axis of data_append (which IS plane-dependent) each one lands on.
        local_slices = [slice(bs_o[d]*b_id[d], bs_o[d]+(data_type=="PointData")+bs_o[d]*b_id[d]) for d in range(3)]
        j_block_trimmed = j_block[tuple([slice(None,-1 if data_type == "CellData" and not np.all(j_block.shape == bs_o[:w_main.dim]) else None)]*w_main.dim)]
        # Explicitly reorder j_block into the same axis order as the destination window - do NOT
        # rely on numpy broadcasting here, it only auto-inserts missing axes on the left, which
        # silently "worked" for the XY-plane (thickness axis at numpy axis 0) but is wrong once
        # the thickness axis moves to the middle/end for XZ/YZ.
        j_block_local = block_data_to_local_order(j_block_trimmed, w_main.dim)
        data_append[local_axes_to_numpy_slices(local_slices, axis_map)] = local_array_to_numpy(j_block_local, axis_map)
      data_group[i_block][i_s][:] = data_append
    # Attach data for vectors - currently copying but maybe there is a more clever way
    for i_v, i_n in zip(v_names, v_ind):
      # block is composed of subtree
      data_append = np.zeros(np.append(bs_now_numpy_shape, len(i_n)))
      for i_depth, i_ndim in enumerate(i_n):
        for i_merged in range(len(id_now)):
          # translate id from main to this object as the block ids could be shuffled
          i_b_now = w_obj_list[i_ndim].get_block_id(w_main.block_treecode_num[id_now[i_merged]], w_main.level[id_now[i_merged]])
          j_block = block_cache[i_ndim][i_b_now]  # preloaded above (mode B: full array, mode C: this rank's subset)
          # get block position of this sub-octree, in the SAME local (x, y, thickness/z) order as bs_o
          b_id = sub_tree_position[i_block][i_merged].astype(int)
          if w_main.dim == 2: b_id = np.append(b_id, 0)
          # same reasoning as the scalar case above; the trailing vector-component axis
          # (i_depth) is untouched by axis_map since it is not one of the 3 spatial axes
          local_slices = [slice(bs_o[d]*b_id[d], bs_o[d]+(data_type=="PointData")+bs_o[d]*b_id[d]) for d in range(3)]
          j_block_trimmed = j_block[tuple([slice(None,-1 if data_type == "CellData" and not np.all(j_block.shape == bs_o[:w_main.dim]) else None)]*w_main.dim)]
          j_block_local = block_data_to_local_order(j_block_trimmed, w_main.dim)
          data_append[local_axes_to_numpy_slices(local_slices, axis_map) + (i_depth,)] = local_array_to_numpy(j_block_local, axis_map)
      data_group[i_block][i_v][:] = data_append

    # Attach data for grid2Field
    if grid2field is not None:
      for i_f in grid2field:
        for i_merged in range(len(id_now)):
          # translate id from main to this object as the block ids could be shuffled
          i_b_now = w_main.get_block_id(w_main.block_treecode_num[id_now[i_merged]], w_main.level[id_now[i_merged]])
          # get block position of this sub-octree, in the SAME local (x, y, thickness/z) order as bs_o
          # (unlike the scalar/vector loops above, b_id here is already length-3 for 2D data too,
          # since sub_tree_position always stores 3 components - no padding needed)
          b_id = sub_tree_position[i_block][i_merged].astype(int)
          local_slices = [slice(bs_o[d]*b_id[d], bs_o[d]+(data_type=="PointData")+bs_o[d]*b_id[d]) for d in range(3)]
          # Unlike bs_o/j_block, these constant-fill arrays are built directly from bs_o[:dim]
          # (no axis-reversal involved), so they are already in local (x, y, [z]) order. For 2D
          # data we still need to append the size-1 (size-2 for PointData) dummy thickness axis
          # ourselves, since bs_o[:dim] only has 2 entries in that case.
          local_shape = list(bs_o[:w_main.dim] + (data_type=="PointData"))
          if w_main.dim == 2: local_shape.append(1 + (data_type=="PointData"))

          # scalar variables
          if i_f in ["level", "treecode", "refinement_status", "procs", "lgt_ID"]:
            if i_f == "level": grid_value = w_main.level[i_b_now]
            elif i_f == "treecode": grid_value = w_main.block_treecode_num[i_b_now]
            elif i_f == "refinement_status": grid_value = w_main.refinement_status[i_b_now]
            elif i_f == "procs": grid_value = w_main.procs[i_b_now]
            elif i_f == "lgt_ID": grid_value = w_main.lgt_ids[i_b_now]
            # grid_value is a single constant filled across the whole block, so the CONTENT of
            # the np.full array does not depend on axis order - but its SHAPE still has to be
            # permuted into the destination's axis order via local_array_to_numpy, same as any
            # other block data, otherwise the assignment below fails or (worse) silently misplaces
            # the size-1 thickness axis for XZ/YZ planes.
            grid_value_local = np.full(local_shape, grid_value, dtype=np.float64)
            data_group[i_block][i_f][local_axes_to_numpy_slices(local_slices, axis_map)] = local_array_to_numpy(grid_value_local, axis_map)

          # vector fields
          if i_f in ["coords_origin", "coords_spacing"]:
            if i_f == "coords_origin": grid_values = w_main.coords_origin[i_b_now]
            elif i_f == "coords_spacing": grid_values = w_main.coords_spacing[i_b_now]
            # same reasoning as the scalar case: each component is a constant fill, but the
            # trailing vector-component axis must be excluded from the axis_map permutation
            data_append = np.empty(local_shape + [w_main.dim], dtype=np.float64)
            for d in range(w_main.dim): data_append[..., d] = grid_values[d]
            data_group[i_block][i_f][local_axes_to_numpy_slices(local_slices, axis_map) + (slice(None),)] = local_array_to_numpy(data_append, axis_map)

  # close file
  f.close()

  if args.verbose and mpi_rank == 0:
    minutes, seconds = divmod(time.time() - start_time, 60)
    if minutes > 0: print(f"    Added data:                            took {int(minutes)}m {seconds:04.1f}s")
    else: print(f"    Added data:                            took {seconds:.2g}s")

  if mpi_parallel: MPI.Finalize()
            
def vtkhdf_time_bundle(in_folder, out_name, timestamps=[], verbose=True):
  if in_folder not in out_name: vtkhdf_files = sorted(glob.glob(f"{out_name}_*.vtkhdf"))
  else: vtkhdf_files = sorted(glob.glob(f"{out_name}_*.vtkhdf"))
  # extract times
  vtkhdf_timesteps = timestamps
  if len(vtkhdf_timesteps) == 0 or len(vtkhdf_timesteps) != len(vtkhdf_files):
    for i, filename in enumerate(vtkhdf_files): vtkhdf_timesteps.append(filename.split("_")[-1].split(".")[0])
  # Create a list of file entries with time indices
  vtkhdf_entries = [{"name": os.path.split(fname)[1], "time": float(vtkhdf_timesteps[i])} for i, fname in enumerate(vtkhdf_files)]
  # Create the JSON structure
  vtkhdf_data = {
      "file-series-version": "1.0",
      "files": vtkhdf_entries
  }
  # Write the JSON file
  if in_folder not in out_name: series_filename = os.path.join(in_folder, f"{out_name}.vtkhdf.series")
  else: series_filename = f"{out_name}.vtkhdf.series"
  with open(series_filename, "w") as json_file:
      json.dump(vtkhdf_data, json_file, indent=4)
  if verbose: print(f"Bundled data for different times: {series_filename}")

def collect_unused_leaves(cursor, added_indices, unused_indices):
  """
  Recursively walk a vtkHyperTreeGrid from the cursor's current position and collect the
  GlobalNodeIndex of every leaf that is not in added_indices. These are leaves that only exist
  because SubdivideLeaf() creates all siblings of a node at once, even though only one sibling
  may lie on the path to an actual wabbit block - as well as whole subtrees that a
  pruned/non-uniform domain never touches at all.
  """
  if cursor.IsLeaf():
    idx = cursor.GetGlobalNodeIndex()
    if idx not in added_indices: unused_indices.append(idx)
  else:
    for i_child in range(cursor.GetNumberOfChildren()):
      cursor.ToChild(i_child)
      collect_unused_leaves(cursor, added_indices, unused_indices)
      cursor.ToParent()


def hdf2htg(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", split_levels=False, exclude_prefixes=[], include_prefixes=[], origin_translate=[0,0,0], scale=1, plane_2D="XY"):
  """
  Create a HTG containing all block information
  Creating a HTG for actual block data is not possible and very expensive as each point in a hypertreegrid cannot be further divided

    w_obj            - Required  : Object representeing the wabbit data or List of objects
    save_file        - Optional  : save string
    verbose          - Optional  : verbose flag
    save_mode        - Optional  : how to encode data - "ascii", "binary" or "appended"
    origin_translate - Optional  : Ability to move the origin in 3D space, same as in hdf2vtkhdf()
    scale            - Optional  : Ability to rescale the data in 3D space, same as in hdf2vtkhdf()
    plane_2D         - Optional  : For 2D data, which plane it is embedded in - "XY", "XZ" or "YZ",
                                    same convention (and same axis_map) as in hdf2vtkhdf()
  """
  correct_input = False
  if isinstance(w_obj, wabbit_tools.WabbitHDF5file):
    w_obj_list = [w_obj]
    correct_input = True
  if isinstance(w_obj, list) and all(isinstance(elem, wabbit_tools.WabbitHDF5file) for elem in w_obj):
    w_obj_list = w_obj
    correct_input = True
  if not correct_input:
    print(bcolors.FAIL + "ERROR: Wrong input of wabbit state - input single object or list of objects" + bcolors.ENDC)
    return
  
  ### create object that will hold all timesteps, then loop over each timestep and create the grid
  ### However, currently multiple timesteps are not really supported so its better to call it one by one
  multi_block_dataset = vtk.vtkMultiBlockDataSet()
  i_count = 0
  for i_wobj in w_obj_list:
    # skip if this prefix is excluded or not included
    if i_wobj.var_from_filename() in exclude_prefixes: continue  # skip excluded prefixes
    if len(include_prefixes) > 0 and i_wobj.var_from_filename() not in include_prefixes: continue  # skip not included prefixes

    dim = i_wobj.dim
    l_min, l_max = w_obj.get_min_max_level()
    depth = 1 if not split_levels else l_max - l_min+1  # how many different grids are there?

    ### initialize hypertreegrid and all arrays
    htg = [None for _ in range(depth)]
    for i_d in range(depth):
      htg[i_d] = vtkHyperTreeGrid()
      htg[i_d].Initialize()

    # scalar arrays
    names_s = ['level', 'treecode', 'refinement_status', 'procs', 'lgt_ID']
    wabbit_s = [i_wobj.level, i_wobj.block_treecode_num, i_wobj.refinement_status, i_wobj.procs, i_wobj.lgt_ids]
    s_data = [[None for _ in range(depth)] for _ in names_s]
    for i_d in range(depth):
      for i_a, i_array in enumerate(names_s):
          s_data[i_a][i_d] = vtkDoubleArray()
          s_data[i_a][i_d].SetName(i_array)
          s_data[i_a][i_d].SetNumberOfValues(0)
          htg[i_d].GetCellData().AddArray(s_data[i_a][i_d])

    # vector arrays
    names_v = ['coords_spacing', 'coords_origin']
    wabbit_v = [i_wobj.coords_spacing, i_wobj.coords_origin]
    v_data = [[None for _ in range(depth)] for _ in names_v]
    for i_d in range(depth):
      for i_a, i_array in enumerate(names_v):
          v_data[i_a][i_d] = vtkDoubleArray()
          v_data[i_a][i_d].SetName(i_array)
          v_data[i_a][i_d].SetNumberOfValues(0)
          v_data[i_a][i_d].SetNumberOfComponents(dim)
          htg[i_d].GetCellData().AddArray(v_data[i_a][i_d])

    # Figure out how this block's own local axes map onto the global (X, Y, Z) axes - the SAME
    # convention and axis_map as hdf2vtkhdf(): for 2D data, local axis 2 is the dummy "thickness"
    # axis (no data varies along it, so the HTG is collapsed to a single point on whichever
    # global axis it lands on for the chosen plane_2D). This must stay in sync with hdf2vtkhdf().
    if dim == 2:
      if plane_2D.upper() == "XY":   axis_map = [0, 1, 2]  # local x->X, local y->Y, thickness->Z
      elif plane_2D.upper() == "XZ": axis_map = [0, 2, 1]  # local x->X, local y->Z, thickness->Y
      elif plane_2D.upper() == "YZ": axis_map = [1, 2, 0]  # local x->Y, local y->Z, thickness->X
    else:
      axis_map = [0, 1, 2]  # 3D: local (x, y, z) already match global (X, Y, Z)
    inv_axis_map = [None, None, None]
    for local_axis, global_slot in enumerate(axis_map): inv_axis_map[global_slot] = local_axis

    # vtkHyperTreeGrid numbers a node's children by combining a per-axis bit with a per-axis
    # weight, in global (X, Y, Z) order - but a degenerate (1-point, non-subdividing) axis
    # contributes NO bit at all, so child indices only stay contiguous (0..3 for 2D) if the
    # weights of the two real axes are 1 and 2, in global-axis order, with the degenerate axis
    # simply skipped. Naively always using weights (1, 2, 4) for (X, Y, Z) - as if Z were always
    # the degenerate one - only works for the XY-plane; for XZ/YZ it produces indices like 4 or 5
    # for a node that only has 4 children, which corrupts the tree (observed as a segfault in
    # vtk's writer).
    degenerate_axis = axis_map[2] if dim == 2 else None
    child_bit_weight = [0, 0, 0]
    w = 1
    for global_axis in range(3):
      if global_axis == degenerate_axis: continue
      child_bit_weight[global_axis] = w
      w *= 2

    for i_d in range(depth):
      # a HyperTreeGrid axis with only 1 point does not subdivide - we use this to collapse the
      # dummy thickness axis of 2D data. (Previously this always thinned out global axis Z,
      # i.e. it silently assumed the XY-plane; now it thins out whichever global axis the
      # thickness maps to for the chosen plane_2D.)
      dimensions = [2, 2, 2]
      if dim == 2: dimensions[axis_map[2]] = 1
      htg[i_d].SetDimensions(dimensions)
      htg[i_d].SetBranchFactor(2)

    ### Define grid coordinates
    for i_d in range(depth):
      offset = 1.1*np.max(i_wobj.domain_size) * (i_d + (l_min-1)*split_levels)
      for i_dim in range(3):
        # which local axis (x=0, y=1, thickness/z=2) is embedded into this global axis?
        local_axis = inv_axis_map[i_dim]
        val_range = vtkDoubleArray()
        if local_axis == 2 and dim == 2:
          # dummy thickness axis: collapse to a single point, placed purely via origin_translate
          # (+ the split-levels offset, which - like in hdf2vtkhdf() - is always applied along
          # global Y, regardless of which local axis Y happens to carry for this plane)
          val_range.SetNumberOfValues(1)
          val_range.SetValue(0, (origin_translate[i_dim] + (i_dim==1)*offset) * scale)
        else:
          val_range.SetNumberOfValues(2)
          val_range.SetValue(0, (0 + origin_translate[i_dim] + (i_dim==1)*offset) * scale)
          val_range.SetValue(1, (i_wobj.domain_size[local_axis] + origin_translate[i_dim] + (i_dim==1)*offset) * scale)
        if i_dim == 0: htg[i_d].SetXCoordinates(val_range)
        elif i_dim == 1: htg[i_d].SetYCoordinates(val_range)
        elif i_dim == 2: htg[i_d].SetZCoordinates(val_range)

    ### 
    #   crawl along each cell and insert data
    #   vtkHyperTreeGrid functions with cursors actually walking the trees
    #   so that is what we do here, always walk up and down the tree for each block
    ###

    unknown_value = -10

    # lets create the cursor and root cell
    cursor = [None for _ in range(depth)]
    block_added = [{} for _ in range(depth)]
    for i_d in range(depth):
      cursor[i_d] = vtkHyperTreeGridNonOrientedCursor()
      offsetIndex = 0
      htg[i_d].InitializeNonOrientedCursor(cursor[i_d], 0, True)
      cursor[i_d].SetGlobalIndexStart(offsetIndex)
      # insert zero data for root
      for i_a in range(len(s_data)):
        s_data[i_a][i_d].InsertTuple1(cursor[i_d].GetGlobalNodeIndex(), unknown_value)
      for i_a in range(len(v_data)):
        for i_dim in range(dim):
          v_data[i_a][i_d].InsertComponent(cursor[i_d].GetGlobalNodeIndex(), i_dim, unknown_value)

    # loop over all blocks, crawl and insert points
    start_time = time.time()
    for i_block in range(i_wobj.total_number_blocks):
      level, treecode = i_wobj.level[i_block], i_wobj.block_treecode_num[i_block]
      d_p = 0 if not split_levels else level-l_min  # depth of this point

      rem_time = (i_wobj.total_number_blocks - i_block) * (time.time() - start_time) / (i_block + 1e-4*(i_block == 0))
      # Format remaining time in HH:MM:SS format
      hours, rem = divmod(rem_time, 3600)
      minutes, seconds = divmod(rem, 60)
      if verbose and mpi_rank==0:
        print_progress_bar(i_block, i_wobj.total_number_blocks, prefix=f'    Processing {save_file}:', suffix=f'ETA: {int(hours):02d}h {int(minutes):02d}m { seconds:02.1f}s', length=20)

      # go down the tree
      for i_level in np.arange(level)+1:
        i_digit_raw = wabbit_tools.tc_get_digit_at_level(treecode, i_level, max_level=i_wobj.max_level, dim=i_wobj.dim)
        # WABBIT's own treecode digit convention: the 1s-bit flips in the local-y direction, the
        # 2s-bit in local-x, and (3D only) the 4s-bit in local-z. For 2D data the thickness axis
        # has no treecode bit at all and never subdivides, so its bit is always 0.
        local_bit = [(i_digit_raw // 2) % 2, i_digit_raw % 2, (i_digit_raw // 4) % 2]
        # Route each local bit onto whichever global axis it is embedded into for this plane
        # (same axis_map as above / as hdf2vtkhdf()), then recombine with child_bit_weight -
        # NOT a fixed (1, 2, 4), since the degenerate axis must contribute no weight at all
        # (see child_bit_weight comment above for why).
        global_bit = [0, 0, 0]
        for local_axis, global_slot in enumerate(axis_map): global_bit[global_slot] = local_bit[local_axis]
        i_digit = sum(global_bit[a] * child_bit_weight[a] for a in range(3))

        for i_d in range(depth):
          if i_level > l_min+i_d and split_levels: continue

          if cursor[i_d].IsLeaf(): cursor[i_d].SubdivideLeaf()
          cursor[i_d].ToChild(i_digit)
          c_index = cursor[i_d].GetGlobalNodeIndex()

          # insert zero for non-leafs as we only have leafs in our code currently
          # only insert values the first time this branch is walked and new blocks are encountered
          if not block_added[i_d].get(c_index, False):
            for i_a in range(len(s_data)):
              s_data[i_a][i_d].InsertTuple1(c_index, unknown_value)
            for i_a in range(len(v_data)):
              for i_dim in range(dim):
                v_data[i_a][i_d].InsertComponent(c_index, i_dim, unknown_value)

      # insert points on block level
      for i_a in range(len(s_data)):
        s_data[i_a][d_p].InsertTuple1(cursor[d_p].GetGlobalNodeIndex(), wabbit_s[i_a][i_block])
      for i_a in range(len(v_data)):
        for i_dim in range(dim):
          v_data[i_a][d_p].InsertComponent(cursor[d_p].GetGlobalNodeIndex(), i_dim, wabbit_v[i_a][i_block, i_dim])
      # insert index as treated
      block_added[d_p][cursor[d_p].GetGlobalNodeIndex()] = True

      # In theory we could create a 16x16x16 block or 32x32x32 block and treat them as full childrens in the HyperTreeGrid
      # However, this is painfully slow and creates unnecessary large files

      # # insert block as 16x16x16 grid
      # # first - interpolate the block
      # depth = 4
      # zoom_factors = np.array([2**depth]*dim) / wabbit_obj.blocks.shape[1:]
      # interpolated_block = scipy.ndimage.zoom(wabbit_obj.blocks[i_block, :], zoom_factors, order=1)  # order=1 for linear interpolation

      # for ix in range(2**depth):
      #   for iy in range(2**depth):
      #     for iz in range(2**depth * (dim==3) + (dim==2)):
      #       # build treecode
      #       treecode = wabbit_tools2.tc_encoding([ix, iy, iz], max_level=depth, dim=dim)
      #       tc_s = wabbit_tools2.tc_to_str(treecode, depth, depth, dim)
      #       for i_depth in range(depth):
      #         if cursor.IsLeaf(): cursor.SubdivideLeaf()

      #         # insert zero for intermediates
      #         idx = cursor.GetGlobalNodeIndex()
      #         scalarArray.InsertTuple1(idx, 0)
      #         blockArray.InsertTuple1(cursor.GetGlobalNodeIndex(), 0)

      #         # extract digit from treecode and go in direction
      #         dir_now = wabbit_tools2.tc_get_digit_at_level(treecode, i_depth, max_level=depth, dim=dim)
      #         cursor.ToChild(dir_now)
            
      #       # insert data
      #       idx = cursor.GetGlobalNodeIndex()
      #       scalarArray.InsertTuple1(idx, 0)
      #       if dim == 2:
      #         blockArray.InsertTuple1(cursor.GetGlobalNodeIndex(), interpolated_block[ix, iy])
      #       else:
      #         blockArray.InsertTuple1(cursor.GetGlobalNodeIndex(), interpolated_block[iz, ix, iy])

      #       # go back up
      #       for i_depth in range(depth): cursor.ToParent()

      # go up the tree
      for i_d in range(depth): cursor[i_d].ToRoot()

    # hide every leaf that never received real block data (unused siblings created by
    # SubdivideLeaf(), and whole subtrees a pruned/non-uniform domain never visits)
    for i_d in range(depth):
      unused_indices = []
      collect_unused_leaves(cursor[i_d], block_added[i_d], unused_indices)
      if len(unused_indices) > 0:
        n_cells = htg[i_d].GetNumberOfCells()
        mask = vtkBitArray()
        mask.SetNumberOfValues(n_cells)
        for i in range(n_cells): mask.SetValue(i, 0)
        for idx in unused_indices: mask.SetValue(idx, 1)
        htg[i_d].SetMask(mask)

    # Add the vtkHyperTreeGrid to the multi-block dataset
    for i_d in range(depth):
      multi_block_dataset.SetBlock(i_count, htg[i_d])
      multi_block_dataset.GetMetaData(i_count).Set(vtk.vtkCompositeDataSet.NAME(), f"Time={np.round(i_wobj.time, 12)}, Depth={i_d}")
      i_count += 1


  # Setup the writer
  if len(w_obj_list) == 1 and depth==1:
    writer = vtk.vtkXMLHyperTreeGridWriter()
    writer.SetInputData(htg[0])
    file_ending = '.htg'
  else:
    writer = vtk.vtkXMLMultiBlockDataWriter()
    writer.SetInputData(multi_block_dataset)
    file_ending = '.vtm'
  if save_file is None: save_file = w_obj_list[0].orig_file.replace(".h5", file_ending)
  if not save_file.endswith(file_ending): save_file += file_ending
  writer.SetFileName(save_file)
  if save_mode.lower() == "ascii": writer.SetDataModeToAscii()
  elif save_mode.lower() == "binary": writer.SetDataModeToBinary()
  elif save_mode.lower() == "appended": writer.SetDataModeToAppended()
  else: print(bcolors.FAIL + f"ERROR: save mode unknown - {save_mode}" + bcolors.ENDC)
  writer.Write()


def htg_time_bundle(in_folder, out_name, timestamps=[], verbose=True):
  if in_folder not in out_name: htg_files = sorted(glob.glob(os.path.join(in_folder, f"{out_name}_*.htg")))
  else: htg_files = sorted(glob.glob(f"{out_name}_*.htg"))
  # Create PVD file content
  grid_content = '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n'
  # write grid files
  for i, filename in enumerate(htg_files):
      if len(timestamps) == 0 or len(timestamps) != len(htg_files):
        time_stamp = filename.split("_")[-1].split(".")[0]
      else:
        time_stamp = timestamps[i]
      grid_content += f'    <DataSet timestep="{time_stamp}" file="{os.path.split(filename)[1]}"/>\n'
  grid_content += '  </Collection>\n</VTKFile>'
  # Write to PVD files
  if in_folder not in out_name: series_filename = os.path.join(in_folder, f"{out_name}-grid.pvd")
  else: series_filename = f"{out_name}-grid.pvd"
  with open(series_filename, "w") as f: f.write(grid_content)
      
  if verbose: print(f"Bundled grids for different times in file {out_name}-grid.pvd'")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  group_htg = parser.add_mutually_exclusive_group()
  group_htg.add_argument("--htg", help="""Write Hypertreegrid file to investigate the block metadatas like level, refinement status or procs.
  If input is a directory, each h5 file create one htg file""", action="store_true")
  group_htg.add_argument("--htg1", help="""Write Hypertreegrid file to investigate the block metadatas like level, refinement status or procs.
  If input is a directory only one htg per time-step will be created from the first h5 file""", action="store_true")
  parser.add_argument("--vtkhdf", help="Write block data as vtkhdf file. Each time-step results in one vtkhdf file", action="store_true")

  parser.add_argument("-o", "--outfile", help="vtkhdf file to write to, default is all_[Time].vtkhdf", default="all")
  parser.add_argument("-i", "--infile", help="file or directory of h5 files, if not ./", default="./")

  parser.add_argument("--cvs-split-levels", help="For overfull CVS grids, divide them by levels", action="store_true")
  parser.add_argument("-m", "--merge-grid", help="Use the merge algorithm to merge full sister blocks", action="store_true")

  parser.add_argument("--low-memory", help="""Read block data for this timestep in small batches (only the blocks a given
  MPI rank actually needs), instead of loading each variable's entire block array into RAM at once. Slower to prepare (an
  extra bookkeeping pass to figure out which blocks are needed) but keeps the per-rank memory footprint down - use this if
  a single timestep does not fit in memory (very large/high-resolution grids, or many local MPI ranks on one node).""", action="store_true")

  parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")

  parser.add_argument("-t", "--time-bundle", help="Bundle all htg files for different times to one file. Works only for folders as input and for --vtkhdf or --htg1.", action="store_true")
  parser.add_argument("-p", "--point-data", help="Save as pointdata, elsewise celldata is saved", action="store_true")

  parser.add_argument("--prune", help="Prune the grid, i.e. remove blocks with constant values with respect to the first file per timestep", action="store_true")
  parser.add_argument("--prune-tolerance", help="Allowed maximum deviation from block mean value (defaults to 1e-3)", default=1e-3, type=float)

  parser.add_argument("--grid2field", help="List of grid variables that will be additionally saved as field variables. Attention: This can be memory intensive.", nargs='+', default=None, type=str)

  parser.add_argument("--origin", help="Ability to move the origin in 3D space", default=[0, 0, 0], nargs=3, type=float)
  parser.add_argument("--scale", help="Ability to rescale the data in 3D space", default=1, type=float)
  parser.add_argument("--2D-plane", help="If the data is 2D, in what plane lies the grid? Options are XY, XZ or YZ", default="XY", type=str)

  # parser.add_argument("-n", "--time-by-fname", help="""How shall we know at what time the file is? Sometimes, you'll end up with several
  # files at the same time, which have different file names. Then you'll want to
  # read the time from the filename, since paraview crashes if two files are at the
  # same instant. Setting -n will force hdf2xmf.py to read from filename, eg mask_00010.h5
  # will be at time 10, even if h5 attributes tell it is at t=0.1""", action="store_true")
  # parser.add_argument("-1", "--one-file-per-timestep", help="""Sometimes, it is useful to generate one XMF file per
  # time step (and not one global file), for example to compare two time steps. The -1 option generates these individual
  # files. If -o outfile.xmf is set, then the files are named outfile_0000.xmf, outfile_0001.xmf etc.""", action="store_true")
  
  parser.add_argument("-q", "--scalars", help="""Overwrite vector recongnition. Normally, a file ux_8384.h5 is interpreted as vector,
  so we also look for uy_8384.h5 and [in 3D mode] for uz_8384.h5. -q overwrites this behavior and individually processes all prefixes as scalars.
  This option is useful if for some reason you have a file that ends with {x,y,z} is not a vector or if you downloaded just one component, e.g. ux_00100.h5
  """, action="store_true")
  group1 = parser.add_mutually_exclusive_group()
  group1.add_argument("--include-prefixes", help="Include just these prefixes, if the files exist (space separated)", nargs='+')
  group1.add_argument("--exclude-prefixes", help="Exclude these prefixes (space separated)", nargs='+')
  group2 = parser.add_mutually_exclusive_group()
  group2.add_argument("--include-timestamps", help="Include just use these timestamps, if the files exist (space separated)", nargs='+')
  group2.add_argument("--exclude-timestamps", help="Exclude these timestamps (space separated)", nargs='+')
  # group3 = parser.add_mutually_exclusive_group()
  # group3.add_argument("--skip-incomplete-timestamps", help="If some files are missing, skip the time step", action="store_true")
  # group3.add_argument("--skip-incomplete-prefixes", help="If some files are missing, skip the prefix", action="store_true")
  args = parser.parse_args()

  if args.verbose and mpi_rank == 0:
    print( bcolors.OKGREEN + "*"*50 + bcolors.ENDC )
    if not mpi_parallel:
      print( bcolors.OKGREEN + "**    " + f'hdf2vtkhdf.py in serial mode'.ljust(42) + "**" + bcolors.ENDC )
    else:
      print( bcolors.OKGREEN + "**    " + f'hdf2vtkhdf.py in parallel mode, np={mpi_size}'.ljust(42) + "**" + bcolors.ENDC )

    print( bcolors.OKGREEN + "*"*50 + bcolors.ENDC )
  
  # check if we want to convert anything at all
  if not any([args.htg, args.htg1, args.vtkhdf]):
    print(bcolors.FAIL + "ERROR: Please select any of --htg, --htg1 or --vtkhdf to convert the files" + bcolors.ENDC)
    exit(0)
  if (args.htg or args.htg1) and not loaded_vtk:
    print(bcolors.FAIL + "ERROR: Please install vtk to use --htg or --htg1" + bcolors.ENDC)
    exit(0)
  
  # on a large dataset of files, it may be useful to ignore some time steps
  # if you're not interested in them. The --exclude-timestamps option lets you do that
  if args.exclude_timestamps is None: args.exclude_timestamps = []
  else:
    args.exclude_timestamps = np.array([float(t) for t in args.exclude_timestamps])
    print(f"We will exclude the following timestamps: {args.exclude_timestamps}")

  # on a large dataset of files, it may be useful to use just some time steps
  # and ignore all other.
  if args.include_timestamps is None: args.include_timestamps = []
  else:
    args.include_timestamps = np.array([float(t) for t in args.include_timestamps])
    print("We will include only the following timestamps: ", args.include_timestamps)
  
  if args.exclude_prefixes is None: args.exclude_prefixes = []
  if args.include_prefixes is None: args.include_prefixes = []

  
  # set directory in case infile is dir and outfile is default
  if args.outfile == "all" and os.path.isdir(args.infile):
    args.outfile = os.path.join(args.infile, args.outfile)

  # check for pruning
  if not args.prune: args.prune_tolerance = None

  # check if the inputted grid2field variables are valid
  if args.grid2field is not None:
    valid_variables = [
      "level", "treecode", "refinement_status", "procs", "lgt_ID", "coords_spacing", "coords_origin"
    ]
    for i_grid_variable in args.grid2field:
      if i_grid_variable not in valid_variables:
        print(bcolors.FAIL + f"ERROR: Grid2field variable {i_grid_variable} is not valid. Valid variables are: {valid_variables}" + bcolors.ENDC)
        exit(0)

  # for one file we simply read in this file and process it
  time_process = {}
  if os.path.isfile(args.infile) and args.infile.endswith(".h5"):
    state_1 = wabbit_tools.WabbitHDF5file()
    state_1.read(args.infile, read_var="meta", verbose=args.verbose and mpi_rank == 0)
    time_1 = np.round(state_1.time, 12)  # round to 12 digits to avoid floating points diffrences
    time_process[time_1] = [state_1]
    filelist = [1]  # for verbose
    # set name to var name in case if infile is file and outfile is default, results will be same but with different ending (.htg or .vtm)
    if args.outfile == "all" and os.path.isfile(args.infile):
      args.outfile = os.path.join(os.path.split(args.infile)[0], state_1.var_from_filename())
  elif os.path.isdir(args.infile):
      # get the list of files
    filelist = sorted( glob.glob(os.path.join(args.infile,"*.h5")) )
    
    # remove all files from the list that are not on the include list, if an include list is given
    if args.include_prefixes:
        filelist = [ f for f in filelist if os.path.basename(f).split("_", 1)[0] in args.include_prefixes ]
    
    # remove all files from the list that are on the exclude list, if an exclude list is given
    if args.exclude_prefixes:
        filelist = [ f for f in filelist if not os.path.basename(f).split("_", 1)[0] in args.exclude_prefixes ]
      
    
    for i_file in filelist:
      state_1 = wabbit_tools.WabbitHDF5file()
      state_1.read(i_file, read_var='meta', verbose=args.verbose and mpi_rank == 0)
      time_1 = np.round(state_1.time, 12)  # round to 12 digits to avoid floating points diffrences
      
      if not time_1 in time_process:
        time_process[time_1] = []
      time_process[time_1].append(state_1)
  
  #-------------------------------------------------------------------------------
  # remove all time instants that we do not want
  #-------------------------------------------------------------------------------
  remove_t = []
  for t in time_process.keys():
    if t in args.exclude_timestamps: remove_t.append(t)
    if t not in args.include_timestamps and len(args.include_timestamps) > 0: remove_t.append(t)
  for i_remove in remove_t:
    if i_remove in time_process: del time_process[i_remove]
  
  if len(time_process) == 0:
    print(bcolors.FAIL + f"ERROR: I did not find any .h5 files on path {args.infile}" + bcolors.ENDC)
  if args.verbose and mpi_rank == 0:
    print(f"Found {len(filelist)} .h5 file(s) on {len(time_process)} time instant(s)")

  for i_n, i_time in enumerate(time_process):
    start_time = time.time()
    if args.verbose and mpi_rank == 0: print(f"Time {i_time}, {i_n+1}/{len(time_process)}")

    # create hypertreegrid
    if args.htg1: hdf2htg(time_process[i_time][0], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, split_levels=args.cvs_split_levels, origin_translate=args.origin, scale=args.scale, plane_2D=args.__dict__["2D_plane"])
    elif args.htg:
      for i_wobj in time_process[i_time]:
        save_file = f"{args.outfile}-{i_wobj.var_from_filename(verbose=False)}_{wabbit_tools.time2wabbitstr(i_time)}"
        hdf2htg(i_wobj, save_file=save_file, verbose=args.verbose, split_levels=args.cvs_split_levels, exclude_prefixes=args.exclude_prefixes, include_prefixes=args.include_prefixes, origin_translate=args.origin, scale=args.scale, plane_2D=args.__dict__["2D_plane"])

    # create vtkhdf
    if args.vtkhdf:
      hdf2vtkhdf(time_process[i_time], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, scalars=args.scalars, split_levels=args.cvs_split_levels, merge=args.merge_grid, prune_tolerance=args.prune_tolerance, data_type="CellData" if not args.point_data else "PointData", exclude_prefixes=args.exclude_prefixes, include_prefixes=args.include_prefixes, grid2field=args.grid2field, origin_translate=args.origin, scale=args.scale, plane_2D=args.__dict__["2D_plane"], low_memory=args.low_memory)

    # output timing
    if args.verbose and mpi_rank == 0:
      minutes, seconds = divmod(time.time() - start_time, 60)
      if minutes > 0: print(f"    Converted file:                        took {int(minutes)}m {seconds:04.1f}s")
      else: print(f"    Converted file:                        took {seconds:.2g}s")

  # vtkhdf or htg files are created one file for each time-step, but we can luckily bundle them all up so let's do this!
  if args.time_bundle:
    if args.vtkhdf: vtkhdf_time_bundle(args.infile, args.outfile, timestamps=sorted(time_process.keys()), verbose=args.verbose)
    if args.htg1: htg_time_bundle(args.infile, args.outfile, timestamps=sorted(time_process.keys()), verbose=args.verbose)
