#!/usr/bin/env python3
from vtkmodules.vtkCommonCore import (
    vtkDoubleArray,
)
from vtkmodules.vtkCommonDataModel import (
    vtkHyperTreeGrid,
    vtkHyperTreeGridNonOrientedCursor,
)
import vtk
import numpy as np
import time
import argparse
import glob, os, sys

sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
import wabbit_tools
import bcolors

# Progress bar function
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
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



def hdf2htg(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", split_levels=False):
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
      if verbose:
        print_progress_bar(i_block, i_wobj.total_number_blocks, prefix=f'Processing htg:', suffix=f'ETA: {int(hours):02d}h {int(minutes):02d}m { seconds:02.1f}s')

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
  htg_files = sorted(glob.glob(os.path.join(in_folder, f"{out_name}_*.htg")))
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
  with open(os.path.join(in_folder, f"{out_name}-grid.pvd"), "w") as f: f.write(grid_content)
      
  if verbose: print(f"Bundled grids for different times in file {out_name}-grid.pvd'")


def hdf2vtm(w_obj: wabbit_tools.WabbitHDF5file, save_file=None, verbose=True, save_mode="appended", scalars=False, split_levels=False):
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
        if verbose: print( pre+' is a 3D vector (x,y,z)')
    elif (pre+'x' in p_names and pre+'y' in p_names and w_main.dim==2):
        if verbose: print( pre+' is a 2D vector (x,y)')
    else:
        print( f"WARRNING: {pre} is not a vector (its x-, y- or z- component is missing..)")
        v_ind.remove(v_ind[v_names.index(pre)])
        v_names.remove( pre )
        # if pre+'x' in p_names: scalars.append(pre+'x')
        # if pre+'y' in p_names: scalars.append(pre+'y')
        # if pre+'z' in p_names: scalars.append(pre+'z')

  # Create the DataSet
  multi_block = vtk.vtkMultiBlockDataSet()
  # amr_block = vtk.vtkNonOverlappingAMR()
  # amr_block = vtk.vtkOverlappingAMR()

  # Define the number of blocks
  multi_block.SetNumberOfBlocks(w_main.total_number_blocks)
  # amr_block.Initialize(w_main.max_level, wabbit_tools.block_level_distribution(w_main))
  # amr_block_id = np.zeros(w_main.max_level).astype(int)
  
  start_time = time.time()
  for i_b in range(w_main.total_number_blocks):
      b_tc, b_lvl = w_main.block_treecode_num[i_b], w_main.level[i_b]

      rem_time = (w_main.total_number_blocks - i_b) * (time.time() - start_time) / (i_b + 1e-4*(i_b == 0))
      # Format remaining time in HH:MM:SS format
      hours, rem = divmod(rem_time, 3600)
      minutes, seconds = divmod(rem, 60)
      if verbose:
          print_progress_bar(i_b, w_main.total_number_blocks, prefix=f'Processing vtm:', suffix=f'ETA: {int(hours):02d}h {int(minutes):02d}m { seconds:02.1f}s')

      ###
      #   Create block
      #   vtk wants the dimensions and spacing for the edge-based noation so we have to adapt block_size and spacing
      ###
      # block = vtk.vtkUniformGrid()
      block = vtk.vtkImageData()
      # for overfull CVS grids we have the option to split them into levels to make the overlay visible
      split_levels_add = (split_levels * (w_main.level[i_b]-1) * np.max(w_main.domain_size)) * 1.1
      # attention: the spacings have to be inverted, I don't completely know why but it is necessary
      if w_main.dim == 2:
        block.SetDimensions(w_main.block_size[1], w_main.block_size[0], 1)
        block.SetOrigin(w_main.coords_origin[i_b, 1], w_main.coords_origin[i_b, 0], 0 + split_levels_add)
        spacing_now = w_main.coords_spacing[i_b, :w_main.dim] * w_main.block_size[:w_main.dim] / (w_main.block_size[:w_main.dim])
        block.SetSpacing(spacing_now[1], spacing_now[0], 0)
      else:
        block.SetDimensions(w_main.block_size[2], w_main.block_size[1], w_main.block_size[0])
        block.SetOrigin(w_main.coords_origin[i_b, 2], w_main.coords_origin[i_b, 1], w_main.coords_origin[i_b, 0] + split_levels_add)
        spacing_now = w_main.coords_spacing[i_b, :w_main.dim] * w_main.block_size[:w_main.dim] / (w_main.block_size[:w_main.dim])
        block.SetSpacing(spacing_now[2], spacing_now[1], spacing_now[0])

      # Attach data for scalars - currently copying but maybe there is a more clever way
      for i_s, i_n in zip(s_names, s_ind):
        # files could have different block ordering so lets correct this
        i_b_now = w_obj_list[i_n].get_block_id(b_tc, b_lvl)

        data_now = vtk.vtkDoubleArray()
        data_now.SetName(i_s)
        data_now.SetNumberOfValues((w_main.block_size[0]-1) * (w_main.block_size[1] -1) * (w_main.block_size[2] - (w_main.dim == 3)))

        # Copy the data into the vtkDoubleArray
        if w_main.dim == 2:
          block_data_flat = w_obj_list[i_n].blocks[i_b_now, :-1, :-1].flatten()  # Flatten the array if it's not already 1D
        else:
          block_data_flat = w_obj_list[i_n].blocks[i_b_now, :-1, :-1, :-1].flatten()  # Flatten the array if it's not already 1D    
        for i in range(block_data_flat.size):
          data_now.SetValue(i, block_data_flat[i])

        block.GetCellData().AddArray(data_now)
      
      # Attach data for vectors - currently copying but maybe there is a more clever way
      for i_v, i_n in zip(v_names, v_ind):
        data_now = vtk.vtkDoubleArray()
        data_now.SetName(i_v)
        data_now.SetNumberOfValues((w_main.block_size[0]-1) * (w_main.block_size[1] -1) * (w_main.block_size[2] - (w_main.dim == 3))*w_main.dim)
        data_now.SetNumberOfComponents(w_main.dim)
        for i_ind, i_ndim in enumerate(i_n):
          # files could have different block ordering so lets correct this
          i_b_now = w_obj_list[i_ndim].get_block_id(b_tc, b_lvl)
          # Copy the data into the vtkDoubleArray
          if w_main.dim == 2:
            block_data_flat = w_obj_list[i_ndim].blocks[i_b_now, :-1, :-1].flatten()  # Flatten the array if it's not already 1D
          else:
            block_data_flat = w_obj_list[i_ndim].blocks[i_b_now, :-1, :-1, :-1].flatten()  # Flatten the array if it's not already 1D  
          for i in range(block_data_flat.size):
            data_now.SetComponent(i, i_ind, block_data_flat[i])

        block.GetCellData().AddArray(data_now)

      geometryFilter = vtk.vtkGeometryFilter()
      geometryFilter.SetInputData(block)
      geometryFilter.Update()

      unstructuredBlock = vtk.vtkUnstructuredGrid()
      unstructuredBlock.DeepCopy(geometryFilter.GetOutput())

      # Assign blocks to the multi-block dataset
      # multi_block.SetBlock(i_b, block)
      multi_block.SetBlock(i_b, unstructuredBlock)
      # box1 = vtk.vtkAMRBox()
      # amr_block.SetAMRBox(w_main.level[i_b]-1, amr_block_id[w_main.level[i_b]-1], box1)
      # amr_block.SetDataSet(w_main.level[i_b]-1, amr_block_id[w_main.level[i_b]-1], block)
      # amr_block_id[w_main.level[i_b]-1] += 1

  # Setup the writer
  writer = vtk.vtkXMLMultiBlockDataWriter()
  writer.SetInputData(multi_block)
  file_ending = '.vtm'

  # writer = vtk.vtkXMLUniformGridAMRWriter()
  # writer.SetInputData(amr_block)
  # file_ending = '.vtu'

  # writer = vtk.vtkHDFWriter()
  # writer.SetInputData(multi_block)
  # file_ending = '.vtkhdf'

  if save_file is None: save_file = w_main.orig_file.replace(".h5", file_ending)
  if not save_file.endswith(file_ending): save_file += file_ending
  writer.SetFileName(save_file)
  if save_mode.lower() == "ascii": writer.SetDataModeToAscii
  elif save_mode.lower() == "binary": writer.SetDataModeToBinary()
  elif save_mode.lower() == "appended": writer.SetDataModeToAppended()
  else: print(bcolors.FAIL + f"ERROR: save mode unknown - {save_mode}" + bcolors.ENDC)
  writer.Write()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  group_htg = parser.add_mutually_exclusive_group()
  group_htg.add_argument("--htg", help="""Write Hypertreegrid file to investigate the block metadatas like level, refinement status or procs.
  If input is a directory, each h5 file create one htg file""", action="store_true")
  group_htg.add_argument("--htg1", help="""Write Hypertreegrid file to investigate the block metadatas like level, refinement status or procs.
  If input is a directory only one htg per time-step will be created from the first h5 file""", action="store_true")
  parser.add_argument("--vtm", help="Write block data as vtm file. Each time-step results in one vtm file", action="store_true")

  parser.add_argument("-o", "--outfile", help="vtk file to write to, default is all_[Time].vtm / *.htg", default="all")
  parser.add_argument("-i", "--infile", help="file or directory of h5 files, if not ./", default="./")

  parser.add_argument("-t", "--time-bundle", help="Bundle all htg files for different times to one file. Works only for folders as input.", action="store_true")

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

  if args.verbose:
    print( bcolors.OKGREEN + "*"*50 + bcolors.ENDC )
    print( bcolors.OKGREEN + "**    " + "hdf2vtk.py".ljust(42) + "**" + bcolors.ENDC )
    print( bcolors.OKGREEN + "*"*50 + bcolors.ENDC )

  # check if we want to convert anything at all
  if not any([args.htg, args.htg1, args.vtm]):
    print(bcolors.FAIL + "ERROR: Please select any of --htg, --htg1 or --vtm to convert the files" + bcolors.ENDC)
    exit(0)
  
  # set directory in case infile is dir and outfile is default
  if args.outfile == "all" and os.path.isdir(args.infile):
    args.outfile = os.path.join(args.infile, args.outfile)

  # for one file we simply read in this file and process it
  time_process = {}
  if os.path.isfile(args.infile) and args.infile.endswith(".h5"):
    state_1 = wabbit_tools.WabbitHDF5file()
    state_1.read(args.infile, verbose=args.verbose)
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
      if args.vtm: state_1.read(i_file, verbose=args.verbose)
      else: state_1.read(i_file, read_var="meta", verbose=args.verbose)
      time_1 = np.round(state_1.time, 12)  # round to 12 digits to avoid floating points diffrences
      if not time_1 in time_process:
        time_process[time_1] = []
      time_process[time_1].append(state_1)
  
  if len(time_process) == 0:
    print(bcolors.FAIL + f"ERROR: I did not find any .h5 files on path {args.infile}" + bcolors.ENDC)
  if args.verbose:
    print(f"Found {len(filelist)} .h5 file(s) on {len(time_process)} time instant(s)")

  for i_n, i_time in enumerate(time_process):
    if args.verbose: print(f"Time {i_time}, {i_n+1}/{len(time_process)}")

    # create hypertreegrid
    if args.htg1: hdf2htg(time_process[i_time][0], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, split_levels=args.cvs_split_levels)
    elif args.htg:
      for i_wobj in time_process[i_time]:
        save_file = f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}_{i_wobj.var_from_filename(verbose=False)}"
        hdf2htg(i_wobj, save_file=save_file, verbose=args.verbose, split_levels=args.cvs_split_levels)
    
    # create vtm with blockdata
    if args.vtm:
      hdf2vtm(time_process[i_time], save_file=f"{args.outfile}_{wabbit_tools.time2wabbitstr(i_time)}", verbose=args.verbose, scalars=args.scalars, split_levels=args.cvs_split_levels)

  # vtkhdf is created one file for each time-step, but we can luckily bundle them all up so let's do this!
  if args.time_bundle: htg_time_bundle(args.infile, args.outfile, timestamps=sorted(time_process.keys()), verbose=args.verbose)

  # # debug stuff
  # # state1 = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/vorabs_000002000000.h5")
  # # hdf2htg(state1)
  # # hdf2vtm(state1)

  # state_phi   = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/test_2D/phi_000000250000.h5")
  # state_testx = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/test_2D/testx_000000250000.h5")
  # state_testy = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/test_2D/testy_000000250000.h5")
  # w_obj_list = [state_phi, state_testx, state_testy]
  # hdf2vtm(w_obj_list, save_file="../WABBIT/TESTING/jul/data_2D_test")

  # state_phi   = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/test_3D/phi_000000051549.h5")
  # state_testx = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/test_3D/testx_000000051549.h5")
  # state_testy = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/test_3D/testy_000000051549.h5")
  # state_testz = wabbit_tools2.WabbitState("../WABBIT/TESTING/jul/test_3D/testz_000000051549.h5")
  # w_obj_list = [state_phi, state_testx, state_testy, state_testz]
  # hdf2vtm(w_obj_list, save_file="../WABBIT/TESTING/jul/data_3D_test")

  # ### test with ACM data
  # acm_root = "/home/julius/Julius_Files/Career/PhD/WABBIT/WABBIT/TESTING/acm/acm_cyl_adaptive_CDF44/"
  # acm_time = "000000100000"
  # acm_var = ["ux", "uy", "p"]
  # w_obj_list = []
  # for i_var in acm_var:
  #   i_path = acm_root + i_var + "_" + acm_time + ".h5"
  #   w_obj_list.append(wabbit_tools2.WabbitState(i_path))
  # hdf2vtm(w_obj_list, save_file="../WABBIT/TESTING/jul/data_acm_test")

