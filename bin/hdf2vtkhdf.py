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


def hdf2vtkhdf(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", scalars=False, split_levels=False, merge=True, prune_tolerance=None, grid2field=None, data_type="CellData", exclude_prefixes=[], include_prefixes=[]):
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
  print(f"    Adding {len(s_names)} scalar field{'s' if len(s_names) != 1 else ''} {'\"' + ', '.join(s_names) + '\"' if len(s_names) > 0 else ''}, {len(v_names)} vector field{'s' if len(v_names) != 1 else ''} {'\"' + ', '.join(v_names) + '\" ' if len(v_names) > 0 else ''}and {len(grid2field)} grid field{'s' if len(grid2field) != 1 else ''} {'\"' + ', '.join(grid2field) + '\" ' if len(grid2field) > 0 else ''}to vtkhdf file")

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
  assembly_group = vtkhdf_group.create_group('Assembly')

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
    else: print(f"    Init blocks :          {total_blocks:7d} blocks, took {seconds:.1g}s")

  # pruning in case prune_tolerance is not None
  if prune_tolerance is not None:
    block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position = prune_grid(w_main, block_id, coords_origin, coords_spacing, level, treecode, sub_tree_size, sub_tree_position, prune_tolerance)
    total_blocks = len(block_id)
    # print to user
    if args.verbose and mpi_rank == 0:
      minutes, seconds = divmod(time.time() - start_time, 60)
      if minutes > 0: print(f"    Prune blocks :         {total_blocks:7d} blocks, took {int(minutes)}m {seconds:04.1f}s")
      else: print(f"    Prune blocks :         {total_blocks:7d} blocks, took {seconds:.1g}s")

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
      else: print(f"    Merged subtrees it {i_merge+1:2d}: {total_blocks:7d} blocks, took {seconds:.1g}s")
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
        else: print(f"    Merged in {dir_names[i_dir]}-dir:       {total_blocks:7d} blocks, took {seconds:.1g}s")

  ### collective loop creating the metadata - all processes need to do this
  start_time = time.time()
  data_group = [[]]*total_blocks
  for i_block in range(total_blocks):
    # for overfull CVS grids we have the option to split them into levels to make the overlay visible
    split_levels_add = (split_levels * (level[i_block]-1) * np.max(w_main.domain_size)) * 1.1

    # Create this block itself
    block_group = vtkhdf_group.create_group(f'Block{i_block}')

    # Add attributes to block
    if w_main.dim == 2:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      block_group.attrs.create('Origin', np.append(coords_origin[i_block][::-1], 0) + np.array([0,split_levels_add,0]), dtype='f8')
      block_group.attrs.create('Spacing', np.append(coords_spacing[i_block][::-1], 0), dtype='f8')
      block_group.attrs.create('WholeExtent', np.array([0, bs_o[0]*sub_tree_size[i_block][0], 0, bs_o[1]*sub_tree_size[i_block][1], 0, 1], dtype='i8'))      
    else:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      block_group.attrs.create('Origin', coords_origin[i_block][::-1] + np.array([0,0,split_levels_add]), dtype='f8')
      block_group.attrs.create('Spacing', coords_spacing[i_block][::-1], dtype='f8')
      block_group.attrs.create('WholeExtent', np.array([0, bs_o[0]*sub_tree_size[i_block][0], 0, bs_o[1]*sub_tree_size[i_block][1], 0, bs_o[2]*sub_tree_size[i_block][2]], dtype='i8'))
    block_group.attrs.create('Type', 'ImageData'.encode('ascii'), dtype=h5py.string_dtype('ascii', len('ImageData')))
    block_group.attrs.create('Version', np.array([2, 3], dtype='i8'))
    block_group.attrs.create('Index', i_block, dtype='i8')
    assembly_group[f'Block{i_block}'] = h5py.SoftLink(f'/VTKHDF/Block{i_block}')

    # Add block data
    if data_type == "CellData": data_group[i_block] = block_group.create_group('CellData')
    elif data_type == "PointData": data_group[i_block] = block_group.create_group('PointData')

    # Create empty dataset for scalars
    bs_now = np.array(sub_tree_size[i_block]) * bs_o
    if data_type == "PointData": bs_now[:w_main.dim] += 1
    if w_main.dim == 2: bs_now[2] == 1
    for i_s, i_n in zip(s_names, s_ind): data_group[i_block].create_dataset(i_s, shape=bs_now[::-1], dtype=np.float64)
    
    # Create empty datasets for vectors
    for i_v, i_n in zip(v_names, v_ind): data_group[i_block].create_dataset(i_v, shape=np.append(bs_now[::-1], w_main.dim), dtype=np.float64)

    # Create empty datasets for grid2Field
    if grid2field is not None:
      for i_f in grid2field:
        # scalar fields
        if i_f in ["level", "treecode", "refinement_status", "procs", "lgt_ID"]: data_group[i_block].create_dataset(i_f, shape=bs_now[::-1], dtype=np.float64)
        # vector fields
        if i_f in ["coords_origin", "coords_spacing"]: data_group[i_block].create_dataset(i_f, shape=np.append(bs_now[::-1], w_main.dim), dtype=np.float64)
  if args.verbose and mpi_rank == 0:
    minutes, seconds = divmod(time.time() - start_time, 60)
    if minutes > 0: print(f"    Created metadata:                      took {int(minutes)}m {seconds:04.1f}s")
    else: print(f"    Created metadata:                      took {seconds:.1g}s")

  ### independent loop attaching the actual data - this is parallelized
  start_time = time.time()
  for i_block in range(int(mpi_rank/mpi_size*total_blocks), int((mpi_rank+1)/mpi_size*total_blocks)):
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
    if w_main.dim == 2: bs_now[2] == 1
    for i_s, i_n in zip(s_names, s_ind):  
      # block is composed of subtree
      data_append = np.zeros(bs_now[::-1])
      for i_merged in range(len(id_now)):
        # translate id from main to this object as the block ids could be shuffled
        i_b_now = w_obj_list[i_n].get_block_id(w_main.block_treecode_num[id_now[i_merged]], w_main.level[id_now[i_merged]])
        j_block = w_obj_list[i_n].block_read(i_b_now)
        # get block position of this sub-octree
        b_id = sub_tree_position[i_block][i_merged].astype(int)
        if w_main.dim == 2: b_id = np.append(b_id, 0)
        # the block structure is in order [z,y,x] and the treecode ordering is [y,x,z]
        data_append[bs_o[2]*b_id[2]:bs_o[2]+(data_type=="PointData")+bs_o[2]*b_id[2], \
                bs_o[1]*b_id[1]:bs_o[1]+(data_type=="PointData")+bs_o[1]*b_id[1], \
                bs_o[0]*b_id[0]:bs_o[0]+(data_type=="PointData")+bs_o[0]*b_id[0]] = j_block[tuple([slice(None,-1 if data_type == "CellData" and not np.all(j_block.shape == bs_o[:w_main.dim]) else None)]*w_main.dim)]
      data_group[i_block][i_s][:] = data_append
    # Attach data for vectors - currently copying but maybe there is a more clever way
    for i_v, i_n in zip(v_names, v_ind):
      # block is composed of subtree
      data_append = np.zeros(np.append(bs_now[::-1], len(i_n)))
      for i_depth, i_ndim in enumerate(i_n):
        for i_merged in range(len(id_now)):
          # translate id from main to this object as the block ids could be shuffled
          i_b_now = w_obj_list[i_ndim].get_block_id(w_main.block_treecode_num[id_now[i_merged]], w_main.level[id_now[i_merged]])
          j_block = w_obj_list[i_ndim].block_read(i_b_now)
          # get block position of this sub-octree
          b_id = sub_tree_position[i_block][i_merged].astype(int)
          if w_main.dim == 2: b_id = np.append(b_id, 0)
          # the block structure is in order [z,y,x] and the treecode ordering is [y,x,z]
          data_append[bs_o[2]*b_id[2]:bs_o[2]+(data_type=="PointData")+bs_o[2]*b_id[2], \
                  bs_o[1]*b_id[1]:bs_o[1]+(data_type=="PointData")+bs_o[1]*b_id[1], \
                  bs_o[0]*b_id[0]:bs_o[0]+(data_type=="PointData")+bs_o[0]*b_id[0],i_depth] = j_block[tuple([slice(None,-1 if data_type == "CellData" and not np.all(j_block.shape == bs_o[:w_main.dim]) else None)]*w_main.dim)]
      data_group[i_block][i_v][:] = data_append

    # Attach data for grid2Field
    if grid2field is not None:
      for i_f in grid2field:
        for i_merged in range(len(id_now)):
          # translate id from main to this object as the block ids could be shuffled
          i_b_now = w_main.get_block_id(w_main.block_treecode_num[id_now[i_merged]], w_main.level[id_now[i_merged]])
          # get block position of this sub-octree
          b_id = sub_tree_position[i_block][i_merged].astype(int)

          # scalar variables
          if i_f in ["level", "treecode", "refinement_status", "procs", "lgt_ID"]:
            if i_f == "level": grid_value = w_main.level[i_b_now]
            elif i_f == "treecode": grid_value = w_main.block_treecode_num[i_b_now]
            elif i_f == "refinement_status": grid_value = w_main.refinement_status[i_b_now]
            elif i_f == "procs": grid_value = w_main.procs[i_b_now]
            elif i_f == "lgt_ID": grid_value = w_main.lgt_ids[i_b_now]
            data_group[i_block][i_f][bs_o[2]*b_id[2]:bs_o[2]+(data_type=="PointData")+bs_o[2]*b_id[2], \
                  bs_o[1]*b_id[1]:bs_o[1]+(data_type=="PointData")+bs_o[1]*b_id[1], \
                  bs_o[0]*b_id[0]:bs_o[0]+(data_type=="PointData")+bs_o[0]*b_id[0]] = np.full(bs_o[:w_main.dim]+(data_type=="PointData"), grid_value, dtype=np.float64)

          # vector fields
          if i_f in ["coords_origin", "coords_spacing"]:
            if i_f == "coords_origin": grid_values = w_main.coords_origin[i_b_now]
            elif i_f == "coords_spacing": grid_values = w_main.coords_spacing[i_b_now]
            data_append = np.empty(list(bs_o[:w_main.dim]+(data_type=="PointData")) + [w_main.dim], dtype=np.float64)
            for d in range(w_main.dim): data_append[..., d] = grid_values[d]
            data_group[i_block][i_f][bs_o[2]*b_id[2]:bs_o[2]+(data_type=="PointData")+bs_o[2]*b_id[2], \
                  bs_o[1]*b_id[1]:bs_o[1]+(data_type=="PointData")+bs_o[1]*b_id[1], \
                  bs_o[0]*b_id[0]:bs_o[0]+(data_type=="PointData")+bs_o[0]*b_id[0], :] = data_append

  # close file
  f.close()

  if args.verbose and mpi_rank == 0:
    minutes, seconds = divmod(time.time() - start_time, 60)
    if minutes > 0: print(f"    Added data:                            took {int(minutes)}m {seconds:04.1f}s")
    else: print(f"    Added data:                            took {seconds:.1g}s")

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

def hdf2htg(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", split_levels=False, exclude_prefixes=[], include_prefixes=[]):
  """
  Create a HTG containing all block information
  Creating a HTG for actual block data is not possible and very expensive as each point in a hypertreegrid cannot be further divided

    w_obj       - Required  : Object representeing the wabbit data or List of objects
    save_file   - Optional  : save string
    verbose     - Optional  : verbose flag
    save_mode   - Optional  : how to encode data - "ascii", "binary" or "appended"
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

    for i_d in range(depth):
      htg[i_d].SetDimensions([2, 2, dim-1])
      htg[i_d].SetBranchFactor(2)

    ### Define grid coordinates
    for i_d in range(depth):
      offset = 1.1*np.max(i_wobj.domain_size) * (i_d + (l_min-1)*split_levels)
      for i_dim in range(3):
        val_range = vtkDoubleArray()
        val_range.SetNumberOfValues(2 - (i_dim == dim+1))
        val_range.SetValue(0, 0 + (i_dim==1)*offset)
        # if not 2D and we look at Z we set the second direction
        if i_dim != dim: val_range.SetValue(1, i_wobj.domain_size[i_dim] + (i_dim==1)*offset)
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
        i_digit = wabbit_tools.tc_get_digit_at_level(treecode, i_level, max_level=i_wobj.max_level, dim=i_wobj.dim)
        # Y and X are swapped as the TC for 1 changes in Y-direction and for 2 in X-direction
        Y = i_digit%2
        X = (i_digit//2)%2
        Z = (i_digit//4)%2
        i_digit = X + 2*Y + 4*Z

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

  parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")

  parser.add_argument("-t", "--time-bundle", help="Bundle all htg files for different times to one file. Works only for folders as input and for --vtkhdf or --htg1.", action="store_true")
  parser.add_argument("-p", "--point-data", help="Save as pointdata, elsewise celldata is saved", action="store_true")

  parser.add_argument("--prune", help="Prune the grid, i.e. remove blocks with constant values with respect to the first file per timestep", action="store_true")
  parser.add_argument("--prune-tolerance", help="Allowed maximum deviation from block mean value (defaults to 1e-3)", default=1e-3, type=float)

  parser.add_argument("--grid2field", help="List of grid variables that will be additionally saved as field variables. Attention: This can be memory intensive.", nargs='+', default=None, type=str)

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
    if args.htg1: hdf2htg(time_process[i_time][0], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, split_levels=args.cvs_split_levels)
    elif args.htg:
      for i_wobj in time_process[i_time]:
        save_file = f"{args.outfile}-{i_wobj.var_from_filename(verbose=False)}_{wabbit_tools.time2wabbitstr(i_time)}"
        hdf2htg(i_wobj, save_file=save_file, verbose=args.verbose, split_levels=args.cvs_split_levels, exclude_prefixes=args.exclude_prefixes, include_prefixes=args.include_prefixes)

    # create vtkhdf
    if args.vtkhdf:
      hdf2vtkhdf(time_process[i_time], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, scalars=args.scalars, split_levels=args.cvs_split_levels, merge=args.merge_grid, prune_tolerance=args.prune_tolerance, data_type="CellData" if not args.point_data else "PointData", exclude_prefixes=args.exclude_prefixes, include_prefixes=args.include_prefixes, grid2field=args.grid2field)

    # output timing
    if args.verbose and mpi_rank == 0:
      minutes, seconds = divmod(time.time() - start_time, 60)
      if minutes > 0: print(f"    Converted file:                        took {int(minutes)}m {seconds:04.1f}s")
      else: print(f"    Converted file:                        took {seconds:.1g}s")

  # vtkhdf or htg files are created one file for each time-step, but we can luckily bundle them all up so let's do this!
  if args.time_bundle:
    if args.vtkhdf: vtkhdf_time_bundle(args.infile, args.outfile, timestamps=sorted(time_process.keys()), verbose=args.verbose)
    if args.htg1: htg_time_bundle(args.infile, args.outfile, timestamps=sorted(time_process.keys()), verbose=args.verbose)