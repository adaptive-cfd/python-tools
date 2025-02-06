#!/usr/bin/env python3
import h5py, os, sys, argparse, glob, time, numpy as np
try:
  try: from mpi4py import MPI
  except: print("Could not load mpi4py - do you have it installed? h5py parallel needs it!")
  mpi_size = MPI.COMM_WORLD.Get_size()
  mpi_parallel = mpi_size > 1
  mpi_rank = MPI.COMM_WORLD.Get_rank()
except:
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


def hdf2vtkhdf(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", scalars=False, split_levels=False):
  """
  Create a multi block dataset from the available data
  This creates many many sub-files but will be changed to a hdf-based implementation soon

    w_obj       - Required  : Object representeing the wabbit data or List of objects
                              list - objs are on same grid at same time and represent different variables and will be combined
    save_file   - Optional  : save string
    verbose     - Optional  : verbose flag
    save_mode   - Optional  : how to encode data - "ascii", "binary" or "appended"
    scalars     - Optional  : should vectors be identified or all included as scalars?
  """
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

  # check if files have same mesh, attr and time
  w_main = w_obj_list[0]
  for i_wobj in w_obj_list[1:]:
    same_mesh = w_main.compareGrid(i_wobj)
    same_attrs = w_main.compareAttr(i_wobj)
    same_time = w_main.compareTime(i_wobj)

    if not all([same_mesh, same_attrs, same_time]):
      if verbose: print(bcolors.FAIL + f"ERROR: data are not similar and cannot be combined" + bcolors.ENDC)
      return
  
  # create variable list
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

  # prepare filename
  file_ending = '.vtkhdf'
  if save_file is None: save_file = w_main.orig_file.replace(".h5", file_ending)
  if not save_file.endswith(file_ending): save_file += file_ending

  # host deletes old file if it exists
  if mpi_rank == 0:
    if os.path.isfile(save_file): os.remove(save_file)  # a bit brute-force, maybe ask for deletion?
  if mpi_parallel: MPI.COMM_WORLD.Barrier()  # ensure that file is properly deleted before all processes continue#

  # now all processes open the file
  if not mpi_parallel: f =  h5py.File(save_file, 'w')
  else: f = h5py.File(save_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)

  vtkhdf_group = f.create_group('VTKHDF', track_order=True)
  vtkhdf_group.attrs.create('Type', 'PartitionedDataSetCollection'.encode('ascii'), dtype=h5py.string_dtype('ascii', len('PartitionedDataSetCollection')))
  vtkhdf_group.attrs.create('Version', np.array([2, 3], dtype='i8'))
  assembly_group = vtkhdf_group.create_group('Assembly')

  # collective loop creating the metadata - all processes need to do this
  start_time = time.time()
  for i_block in range(w_main.total_number_blocks):
    # for overfull CVS grids we have the option to split them into levels to make the overlay visible
    split_levels_add = (split_levels * (w_main.level[i_block]-1) * np.max(w_main.domain_size))

    # Create this block itself
    block_group = vtkhdf_group.create_group(f'Block{i_block}')

    # Add attributes to block
    if w_main.dim == 2:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      block_group.attrs.create('Origin', np.append(w_main.coords_origin[i_block][::-1], 0) + np.array([0,0,split_levels_add]), dtype='f8')
      block_group.attrs.create('Spacing', np.append(w_main.coords_spacing[i_block][::-1], 0), dtype='f8')
      block_group.attrs.create('WholeExtent', np.array([0, w_main.block_size[0]-1, 0, w_main.block_size[1]-1, 0, 1], dtype='i8'))      
    else:
      block_group.attrs.create('Direction', np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype='f8'))
      block_group.attrs.create('Origin', w_main.coords_origin[i_block][::-1] + np.array([0,0,split_levels_add]), dtype='f8')
      block_group.attrs.create('Spacing', w_main.coords_spacing[i_block][::-1], dtype='f8')
      block_group.attrs.create('WholeExtent', np.array([0, w_main.block_size[0]-1, 0, w_main.block_size[1]-1, 0, w_main.block_size[2]-1], dtype='i8'))
    block_group.attrs.create('Type', 'ImageData'.encode('ascii'), dtype=h5py.string_dtype('ascii', len('ImageData')))
    block_group.attrs.create('Version', np.array([2, 3], dtype='i8'))
    block_group.attrs.create('Index', i_block, dtype='i8')
    assembly_group[f'Block{i_block}'] = h5py.SoftLink(f'/VTKHDF/Block{i_block}')

    # Add block data
    cell_data_group = block_group.create_group('CellData')

    # Create dataset for scalars
    for i_s, i_n in zip(s_names, s_ind):
      if w_main.dim == 2: cell_data_group.create_dataset(i_s, shape=[1,w_main.block_size[0]-1, w_main.block_size[1]-1], dtype=np.float64)
      else: cell_data_group.create_dataset(i_s, shape=w_main.block_size-1, dtype=np.float64)
    # Create datasets for vectors
    for i_v, i_n in zip(v_names, v_ind):
      if w_main.dim == 2: cell_data_group.create_dataset(i_v, shape=[1,w_main.block_size[0]-1, w_main.block_size[1]-1, w_main.dim], dtype=np.float64)
      else: cell_data_group.create_dataset(i_v, shape=np.append(w_main.block_size-1, w_main.dim), dtype=np.float64)
  if args.verbose and mpi_rank == 0: print(f"   Created metadata: {time.time() - start_time:.3f} seconds")

  # independent loop attaching the data - this is parallelized
  start_time = time.time()
  for i_block in range(int(mpi_rank/mpi_size*w_main.total_number_blocks), int((mpi_rank+1)/mpi_size*w_main.total_number_blocks)):
    rem_time = (w_main.total_number_blocks - i_block) * (time.time() - start_time) / (i_block + 1e-4*(i_block == 0))
    # Format remaining time in HH:MM:SS format
    hours, rem = divmod(rem_time, 3600)
    minutes, seconds = divmod(rem, 60)
    if verbose and mpi_rank==0 and i_block < int(w_main.total_number_blocks/mpi_size):
        print_progress_bar(i_block, int(w_main.total_number_blocks/mpi_size), prefix=f'   Processing data:', suffix=f'ETA: {int(hours):02d}h {int(minutes):02d}m { seconds:02.1f}s')

    # get celldatagroup
    cell_data_group = vtkhdf_group[f'Block{i_block}']['CellData']

    # Attach data for scalars - currently copying but maybe there is a more clever way
    for i_s, i_n in zip(s_names, s_ind):
      if w_main.dim == 2:
        data_append = w_obj_list[i_n].block_read(i_block)[:-1, :-1].reshape([1,w_main.block_size[0]-1, w_main.block_size[1]-1])
      else:
        data_append = w_obj_list[i_n].block_read(i_block)[:-1, :-1, :-1]
      cell_data_group[i_s][:] = data_append
    # Attach data for vectors - currently copying but maybe there is a more clever way
    for i_v, i_n in zip(v_names, v_ind):
      if w_main.dim == 2:
        data_append = np.zeros([1, w_main.block_size[0]-1, w_main.block_size[1]-1, 0])
        for i_ndim in i_n: data_append = np.append(data_append, w_obj_list[i_ndim].block_read(i_block)[np.newaxis, :-1, :-1, np.newaxis], axis=3)
      else:
        data_append = np.zeros([w_main.block_size[0]-1, w_main.block_size[1]-1, w_main.block_size[2]-1, 0])
        for i_ndim in i_n: data_append = np.append(data_append, w_obj_list[i_ndim].block_read(i_block)[:-1, :-1, :-1, np.newaxis], axis=3)
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
      state_1.read(i_file, verbose=args.verbose and mpi_rank == 0)
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
    hdf2vtkhdf(time_process[i_time], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, scalars=args.scalars, split_levels=args.cvs_split_levels)

    # output timing
    if args.verbose and mpi_rank == 0: print(f"   Converted file:   {time.time() - start_time:.3f} seconds")