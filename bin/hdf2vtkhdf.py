#!/usr/bin/env python3
import h5py, os, sys, argparse, glob, time, numpy as np
try:
  try: from mpi4py import MPI
  except: print("Could not load mpi4py")
  mpi_size = MPI.COMM_WORLD.Get_size()
  mpi_parallel = mpi_size > 1
  mpi_rank = MPI.COMM_WORLD.Get_rank()
  print(f"Running code in parallel on {mpi_size} cores")
except:
  print("Running code in serial")
  mpi_parallel = False
  mpi_rank = 0
  mpi_size = 1

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


def merge_sisters(block_id_o, coords_origin_o, coords_spacing_o, level_o, treecode_o, sub_tree_o, max_level, dim=3):
  """
  Takes a wabbit object and tries to merge all blocks, where all sister blocks are available.
  Returns blocks, coords_origin, coords_spacing, level, sub_tree as arrays with one entry per block
  """
  # dictionary used for lookup of blocks
  tc_find = {(tc, lvl): idx for idx, (tc, lvl) in enumerate(zip(treecode_o, level_o))}

  block_id, coords_origin, coords_spacing, treecode, level, sub_tree = [], [], [], [], [], []
  id_merged = []

  # loop over all blocks and try to merge
  for i_b in range(len(block_id_o)):
    # extract it's position and from it's blocksize it's merge level
    i_treecode, i_level = treecode_o[i_b], level_o[i_b]
    all_sisters = True
    id_sisters = []
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
      if np.any(sub_tree_o[id_find] != sub_tree_o[i_b]):  # blocks do not have the same subtree structure - we do not merge
        all_sisters = False
        break
      id_sisters = np.append(id_sisters, block_id_o[id_find]).astype(int)  # append treecodes
      if id_find_0 in id_merged: break
    # we have found all sisters and proceed with merging
    if all_sisters and id_find_0 not in id_merged:
      # search for meta-data from block with entry zero
      coords_origin.append(coords_origin_o[id_find_0])
      coords_spacing.append(coords_spacing_o[id_find_0])
      level.append(i_level)
      treecode.append(wabbit_tools.tc_set_digit_at_level(i_treecode, 0, i_level-level_set, max_level=max_level, dim=dim))
      sub_tree_new = np.array(sub_tree_o[id_find_0].copy())
      sub_tree_new[:dim] = 2*sub_tree_new[:dim]
      sub_tree.append(sub_tree_new)  # double the blocksize!
      block_id.append(id_sisters)

      # append id of block 0 to an array of finished blocks, so that we do not merge some several times
      id_merged.append(id_find_0)
    elif not all_sisters:
      # we did not find all sisters or they do not share the same blocksize and just append the old block
      coords_origin.append(coords_origin_o[i_b])
      coords_spacing.append(coords_spacing_o[i_b])
      level.append(i_level)
      treecode.append(i_treecode)
      sub_tree.append(sub_tree_o[i_b])
      block_id.append(block_id_o[i_b])
  return block_id, coords_origin, coords_spacing, level, treecode, sub_tree



def hdf2vtkhdf(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", scalars=False, split_levels=False, merge=True):
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
  for i_wobj in w_obj_list[1:]:
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
      if not f_now[:-1] in v_names:
        v_names.append(f_now[:-1])
        v_ind.append([])  # empty handly vor all the indices
      v_ind[v_names.index(f_now[:-1])].append(i_n)  # append this to the index list
      p_names.append(f_now)
    else:
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

  ### merging - tries to merge blocks where all sisters exist, so that paraview needs to load less blocks
  # this is a high-level optimization, but neatly reduces the time to load for paraview
  
  # prepare all arrays, as we merge by looping over them and reducing them
  start_time = time.time()
  coords_origin, coords_spacing, level, treecode = w_main.coords_origin, w_main.coords_spacing, w_main.level, w_main.block_treecode_num
  sub_tree = [[1,1,1]]*w_main.total_number_blocks  # this one is important, as it contains the size of the merged blocks as a subtree (for now uniform, but could stretch only in one direction)
  total_blocks = w_main.total_number_blocks
  block_id = [[i_b] for i_b in np.arange(w_main.total_number_blocks)]
  bs_o = np.array(w_main.block_size.copy())  # original blocksize
  bs_o[:w_main.dim] -= 1

  # sort everything after treecode - potentially usefull for neighbour merging
  combined_list = list(zip(treecode, level, range(total_blocks)))
  id_sorted = [idx for _, _, idx in sorted(combined_list, key=lambda x: (x[0], x[1]))]
  coords_origin, coords_spacing = [coords_origin[i] for i in id_sorted], [coords_spacing[i] for i in id_sorted]
  level, treecode = [level[i] for i in id_sorted], [treecode[i] for i in id_sorted]
  sub_tree, block_id = [sub_tree[i] for i in id_sorted], [block_id[i] for i in id_sorted]
  if args.verbose and mpi_rank == 0: print(f"    Init blocks :    {time.time() - start_time:.3f} seconds, {total_blocks} blocks")

  # this is the actual merging loop, we loop until no new blocks are merged
  jmin, jmax = w_main.get_min_max_level()
  if merge: merge_blocks_it = jmax
  else: merge_blocks_it = 0
  for i_merge in range(merge_blocks_it):
    start_time = time.time()
    total_blocks_old = total_blocks
    # call merge function which does all the job
    block_id, coords_origin, coords_spacing, level, treecode, sub_tree = merge_sisters(block_id, coords_origin, coords_spacing, level, treecode, sub_tree, w_main.max_level, w_main.dim)
    total_blocks = len(block_id)
    if args.verbose and mpi_rank == 0: print(f"Merged blocks {i_merge+1:2d}:    {time.time() - start_time:.3f} seconds, {total_blocks} blocks")
    if total_blocks_old == total_blocks: break


  ### collective loop creating the metadata - all processes need to do this
  start_time = time.time()
  for i_block in range(total_blocks):
    # for overfull CVS grids we have the option to split them into levels to make the overlay visible
    split_levels_add = (split_levels * (level[i_block]-1) * np.max(w_main.domain_size))

    # Create this block itself
    block_group = vtkhdf_group.create_group(f'Block{i_block}')

    # Add attributes to block
    if w_main.dim == 2:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      block_group.attrs.create('Origin', np.append(coords_origin[i_block][::-1], 0) + np.array([0,0,split_levels_add]), dtype='f8')
      block_group.attrs.create('Spacing', np.append(coords_spacing[i_block][::-1], 0), dtype='f8')
      block_group.attrs.create('WholeExtent', np.array([0, bs_o[0]*sub_tree[i_block][0], 0, bs_o[1]*sub_tree[i_block][1], 0, 1], dtype='i8'))      
    else:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      block_group.attrs.create('Origin', coords_origin[i_block][::-1] + np.array([0,0,split_levels_add]), dtype='f8')
      block_group.attrs.create('Spacing', coords_spacing[i_block][::-1], dtype='f8')
      block_group.attrs.create('WholeExtent', np.array([0, bs_o[0]*sub_tree[i_block][0], 0, bs_o[1]*sub_tree[i_block][1], 0, bs_o[2]*sub_tree[i_block][2]], dtype='i8'))
    block_group.attrs.create('Type', 'ImageData'.encode('ascii'), dtype=h5py.string_dtype('ascii', len('ImageData')))
    block_group.attrs.create('Version', np.array([2, 3], dtype='i8'))
    block_group.attrs.create('Index', i_block, dtype='i8')
    assembly_group[f'Block{i_block}'] = h5py.SoftLink(f'/VTKHDF/Block{i_block}')

    # Add block data
    cell_data_group = block_group.create_group('CellData')

    # Create empty dataset for scalars
    bs_now = np.array(sub_tree[i_block]) * bs_o
    if w_main.dim == 2: bs_now[2] == 1
    for i_s, i_n in zip(s_names, s_ind):
      if w_main.dim == 2: cell_data_group.create_dataset(i_s, shape=bs_now[::-1], dtype=np.float64)
      else: cell_data_group.create_dataset(i_s, shape=bs_now[::-1], dtype=np.float64)
    # Create empty datasets for vectors
    for i_v, i_n in zip(v_names, v_ind):
      if w_main.dim == 2: cell_data_group.create_dataset(i_v, shape=np.append(bs_now[::-1], w_main.dim), dtype=np.float64)
      else: cell_data_group.create_dataset(i_v, shape=np.append(bs_now[::-1], w_main.dim), dtype=np.float64)
  if args.verbose and mpi_rank == 0: print(f"   Created metadata: {time.time() - start_time:.3f} seconds")

  ### independent loop attaching the actual data - this is parallelized
  start_time = time.time()
  for i_block in range(int(mpi_rank/mpi_size*total_blocks), int((mpi_rank+1)/mpi_size*total_blocks)):
    rem_time = (total_blocks - i_block) * (time.time() - start_time) / (i_block + 1e-4*(i_block == 0))
    # Format remaining time in HH:MM:SS format
    hours, rem = divmod(rem_time, 3600)
    minutes, seconds = divmod(rem, 60)
    if verbose and mpi_rank==0 and i_block < int(total_blocks/mpi_size):
        print_progress_bar(i_block, int(total_blocks/mpi_size), prefix=f'   Processing data:', suffix=f'ETA: {int(hours):02d}h {int(minutes):02d}m { seconds:02.1f}s')

    # get celldatagroup
    cell_data_group = vtkhdf_group[f'Block{i_block}']['CellData']

    # Attach data for scalars - currently copying but maybe there is a more clever way
    id_now = block_id[i_block]
    bs_now = np.array(sub_tree[i_block]) * bs_o
    if w_main.dim == 2: bs_now[2] == 1
    for i_s, i_n in zip(s_names, s_ind):  
      # block is composed of subtree
      data_append = np.zeros(bs_now[::-1])
      for i_sister in range(len(id_now)):
        # translate id from main to this object as the block ids could be shuffled
        i_b_now = w_obj_list[i_n].get_block_id(w_main.block_treecode_num[id_now[i_sister]], w_main.level[id_now[i_sister]])
        j_block = w_obj_list[i_n].block_read(i_b_now)
        # get block position of this sub-octree
        b_id = np.array(wabbit_tools.tc_decoding(i_sister,level=int(np.log2(len(id_now))//w_main.dim), max_level=int(np.log2(len(id_now))//w_main.dim),dim=w_main.dim))-1
        if w_main.dim == 2: b_id = np.append(b_id, 0)
        # the block structure is in order [z,y,x] and the treecode ordering is [y,x,z]
        data_append[bs_o[2]*b_id[2]:bs_o[2]+bs_o[2]*b_id[2], \
                bs_o[1]*b_id[1]:bs_o[1]+bs_o[1]*b_id[1], \
                bs_o[0]*b_id[0]:bs_o[0]+bs_o[0]*b_id[0]] = j_block[tuple([slice(None,-1)]*w_main.dim)]
      cell_data_group[i_s][:] = data_append
    # Attach data for vectors - currently copying but maybe there is a more clever way
    for i_v, i_n in zip(v_names, v_ind):
      # block is composed of subtree
      data_append = np.zeros(np.append(bs_now[::-1], len(i_n)))
      for i_depth, i_ndim in enumerate(i_n):
        for i_sister in range(len(id_now)):
          # translate id from main to this object as the block ids could be shuffled
          i_b_now = w_obj_list[i_ndim].get_block_id(w_main.block_treecode_num[id_now[i_sister]], w_main.level[id_now[i_sister]])
          j_block = w_obj_list[i_ndim].block_read(i_b_now)
          # get block position of this sub-octree
          b_id = np.array(wabbit_tools.tc_decoding(i_sister,level=int(np.log2(len(id_now))//w_main.dim), max_level=int(np.log2(len(id_now))//w_main.dim),dim=w_main.dim))-1
          if w_main.dim == 2: b_id = np.append(b_id, 0)
          # the block structure is in order [z,y,x] and the treecode ordering is [y,x,z]
          data_append[bs_o[2]*b_id[2]:bs_o[2]+bs_o[2]*b_id[2], \
                  bs_o[1]*b_id[1]:bs_o[1]+bs_o[1]*b_id[1], \
                  bs_o[0]*b_id[0]:bs_o[0]+bs_o[0]*b_id[0],i_depth] = j_block[tuple([slice(None,-1)]*w_main.dim)]
      cell_data_group[i_v][:] = data_append

  # close file
  f.close()

  if args.verbose and mpi_rank == 0: print(f"   Added data:       {time.time() - start_time:.3f} seconds")

  if mpi_parallel: MPI.Finalize()
            


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("-o", "--outfile", help="vtkhdf file to write to, default is all_[Time].vtkhdf", default="all")
  parser.add_argument("-i", "--infile", help="file or directory of h5 files, if not ./", default="./")

  parser.add_argument("--cvs-split-levels", help="For overfull CVS grids, divide them by levels", action="store_true")
  parser.add_argument("-m", "--merge-grid", help="Use the merge algorithm to merge full sister blocks", action="store_true")

  parser.add_argument("-v", "--verbose", help="Enable verbose output", action="store_true")

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
  # group1 = parser.add_mutually_exclusive_group()
  # group1.add_argument("-e", "--exclude-prefixes", help="Exclude these prefixes (space separated)", nargs='+')
  # group1.add_argument("-i", "--include-prefixes", help="Include just these prefixes, if the files exist (space separated)", nargs='+')
  # group2 = parser.add_mutually_exclusive_group()
  # group2.add_argument("-t", "--include-timestamps", help="Include just use these timestamps, if the files exist (space separated)", nargs='+')
  # group2.add_argument("-x", "--exclude-timestamps", help="Exclude these timestamps (space separated)", nargs='+')
  # group3 = parser.add_mutually_exclusive_group()
  # group3.add_argument("-p", "--skip-incomplete-timestamps", help="If some files are missing, skip the time step", action="store_true")
  # group3.add_argument("-l", "--skip-incomplete-prefixes", help="If some files are missing, skip the prefix", action="store_true")
  args = parser.parse_args()

  if args.verbose and mpi_rank == 0:
    print( bcolors.OKGREEN + "*"*50 + bcolors.ENDC )
    if not mpi_parallel:
      print( bcolors.OKGREEN + "**    " + f'hdf2vtkhdf.py in serial mode'.ljust(42) + "**" + bcolors.ENDC )
    else:
      print( bcolors.OKGREEN + "**    " + f'hdf2vtkhdf.py in parallel mode, np={mpi_size}'.ljust(42) + "**" + bcolors.ENDC )

    print( bcolors.OKGREEN + "*"*50 + bcolors.ENDC )
  
  # set directory in case infile is dir and outfile is default
  if args.outfile == "all" and os.path.isdir(args.infile):
    args.outfile = os.path.join(args.infile, args.outfile)

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
    filelist = sorted( glob.glob(os.path.join(args.infile,"*.h5")) )
    for i_file in filelist:
      state_1 = wabbit_tools.WabbitHDF5file()
      state_1.read(i_file, read_var='meta', verbose=args.verbose and mpi_rank == 0)
      time_1 = np.round(state_1.time, 12)  # round to 12 digits to avoid floating points diffrences
      if not time_1 in time_process:
        time_process[time_1] = []
      time_process[time_1].append(state_1)
  
  if len(time_process) == 0:
    print(bcolors.FAIL + f"ERROR: I did not find any .h5 files on path {args.infile}" + bcolors.ENDC)
  if args.verbose and mpi_rank == 0:
    print(f"Found {len(filelist)} .h5 file(s) on {len(time_process)} time instant(s)")

  for i_n, i_time in enumerate(time_process):
    start_time = time.time()
    if args.verbose and mpi_rank == 0: print(f"Time {i_time}, {i_n+1}/{len(time_process)}")

    # create vtkhdf
    hdf2vtkhdf(time_process[i_time], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, scalars=args.scalars, split_levels=args.cvs_split_levels, merge=args.merge_grid)

    # output timing
    if args.verbose and mpi_rank == 0: print(f"   Converted file:   {time.time() - start_time:.3f} seconds")