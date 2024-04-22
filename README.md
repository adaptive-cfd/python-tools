# Python-tools
A collection of python scripts for WABBIT, FLUSI and insects

## Main tools
<details>
<summary>The top level folder contains various scripts with tools around WABBIT & co. Those functions are meant to be used and included within other scripts.</summary>

### wabbit_tools
Various functions and tools to work with preparation and postprocessing of wabbit simulations. Most notable are:
- WabbitState class for reading in a WABBIT `.h5` file to postprocess it later
- Functions to modify init-files
- Functions to deal with treecode in numerical or array representation
- Functions to compare wabbit to flusi results

### insect_tools
Functions dealing with kinematics and force computation of results from insect simulations with WABBIT.
</details>

## Command line utilities
<details>
<summary>
These tools within the bin-folder are meant to be called from the command line. These are also used within the testing suite of WABBIT itself.
</summary>

### WABBIT command line utilities
Various scripts, often wrapping utilities from wabbit_tools, to be used from the command line

### File conversion utilities
Convert results from WABBIT to other known file-formats
- `hdf2mat`: Make data available to Matlab
- `hdf2vtk`: Use vtk-xml format to represent meta-data and fields, readable in ParaView
- `hdf2xmf`: Writes data in xdmf format, readable in ParaView

</details>
