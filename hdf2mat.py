#!/usr/bin/env python3
# I really hate python already:
from __future__ import print_function
import glob, os
import h5py
import argparse
import numpy as np
from scipy.io import savemat
from wabbit_tools import read_wabbit_hdf5
from wabbit_tools import dense_matrix
import shutil



class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))


def get_dset_name( fname ):
    from os.path import basename
    dset_name = basename(fname)
    dset_name = dset_name[0:dset_name.find('_')]

    return dset_name


def get_timestamp( fname ):
    import re
    from os.path import basename

    fname = basename(fname)
    # extract everything between "_" and "." so mask_00000.h5 gives 00000
    # note escape character "\"
    m = re.search('\_(.+?)\.', fname )

    if m:
        timestamp = m.group(1)
    else:
        print("An error occured: we couldn't extract the timestamp")

    return timestamp


def print_list( l ):
    for p in l:
        print(bcolors.HEADER + p + " " + bcolors.ENDC, end='')
    # print just one newline
    print(' ')


def warn( msg ):
    print( bcolors.FAIL + "WARNING! " + bcolors.ENDC + msg)


def uniquelist( l ):
    # if the list has only one unique value, return it, if not, error
    if len(l) == 0:
        return None
    l = sorted(list(set(l)))
    if len(l) == 1:
        return(l[0])
    elif len(l) == 0:
        warn('uniquelist: List is completely empty!')
        return None
    else:
        warn('uniquelist: List ist not unique...something went wrong.')
        print('these are the values we found in the list:')
        print(l)
        return l[0]

def write_mat_file_wabbit(args, outfile, times, timestamps, prefixes, scalars, vectors, directory, mpicommand = "mpirun -np 2 ", level = ""):
    print('-------------------------')
    print('- WABBIT module matlab  -')
    print('-------------------------')

    # use any file to get blocksize and dimensionality
    file = directory + prefixes[0] + '_' + timestamps[0] + '.h5'
    # open h5 file
    f = h5py.File(file)
    # get the dataset handle
    dset_id = f.get('blocks')

    res = dset_id.shape
    dim = len(res)-1
    densedir = directory+"/densedir%d/"%len(timestamps)
    if not os.path.exists(densedir):
        os.makedirs(densedir)
    # loop over time steps
    all_data = []
    data_list = { "names" : prefixes, "data" : [], "time" : times, "domain_size" : np.zeros(dim)}
    for prefix in prefixes:
        data_tmp = []
        for i in range(len(timestamps)):
             #use any of our files at the same timestamp to determine number of blocks
            file = directory + prefix + '_' + timestamps[i] + '.h5'
            file_dense = densedir + prefix + '_' + timestamps[i] + '.h5'
            command = mpicommand + " " + \
                 "wabbit-post --sparse-to-dense "+file + " "+file_dense+ " "  \
                  + level + " 4" #+ order    
            command = command + " >> " + densedir+"wabbit-post.log"
            print("\n",command,"\n")
            ierr = os.system(command)
            if (ierr > 0):
                warn( "Need wabbit for sparse-to-dense! Please export wabbit in PATH. 
                      export PATH=$PATH:/path/to/wabbit/" )
                return 0
            time, x0, dx, box, data, treecode = read_wabbit_hdf5( file_dense )
            data_dense, box_dense = dense_matrix( x0, dx, data, treecode, dim )
            data_tmp.append(data_dense)
            if os.path.isfile(file_dense):
                os.remove(file_dense)
            else:    ## Show an error ##
                print("Error: %s file not found" % file_dense)     
        data_list["domain_size"] = box_dense
        data_list["dx"] = dx
        data_list["data"].append(np.asarray(data_tmp))
  
    # save everything in one file 
    savemat(directory+"/"+outfile, data_list)
    # remove temporal directory
    try:
        shutil.rmtree(densedir)
    except OSError as e:
            print("Error: %s : %s" % (densedir, e.strerror))
    




def main():
    print( bcolors.OKGREEN + "**********************************************" + bcolors.ENDC )
    print( bcolors.OKGREEN + "**   hdf2xml.py                             **" + bcolors.ENDC )
    print( bcolors.OKGREEN + "**********************************************" + bcolors.ENDC )

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--time-by-fname", help="""How shall we know at what time the file is? Sometimes, you'll end up with several
    files at the same time, which have different file names. Then you'll want to
    read the time from the filename, since paraview crashes if two files are at the
    same instant. Setting -n will force hdf2xmf.py to read from filename, eg mask_00010.h5
    will be at time 10, even if h5 attributes tell it is at t=0.1""", action="store_true")
    parser.add_argument("-1", "--one-file-per-timestep", help="""Sometimes, it is useful to generate one XMF file per
    time step (and not one global file), for example to compare two time steps. The -1 option generates these individual
    files. If -o outfile.xmf is set, then the files are named outfile_0000.xmf, outfile_0001.xmf etc.""", action="store_true")
    parser.add_argument("-o", "--outfile", help="Mat file to write to, default is ALL.mat")
    parser.add_argument("-0", "--ignore-origin", help="force origin to 0,0,0", action="store_true")
    parser.add_argument("-u", "--unit-spacing", help="use unit spacing dx=dy=dz=1 regardless of what is specified in h5 files", action="store_true")
    parser.add_argument("-d", "--directory", help="directory of h5 files, if not ./")
    parser.add_argument("-q", "--scalars", help="""Overwrite vector recongnition. Normally, a file ux_8384.h5 is interpreted as vector,
    so we also look for uy_8384.h5 and [in 3D mode] for uz_8384.h5. -q overwrites this behavior and individually processes all prefixes as scalars.
    This option is useful if for some reason
    you have a file that ends with {x,y,z} is not a vector or if you downloaded just one component, e.g. ux_00100.h5
    """, action="store_true")
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("-e", "--exclude-prefixes", help="Exclude these prefixes (space separated)", nargs='+')
    group1.add_argument("-i", "--include-prefixes", help="Include just these prefixes, if the files exist (space separated)", nargs='+')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("-t", "--include-timestamps", help="Include just use these timestamps, if the files exist (space separated)", nargs='+')
    group2.add_argument("-x", "--exclude-timestamps", help="Exclude these timestamps (space separated)", nargs='+')
    group3 = parser.add_mutually_exclusive_group()
    group3.add_argument("-p", "--skip-incomplete-timestamps", help="If some files are missing, skip the time step", action="store_true")
    group3.add_argument("-l", "--skip-incomplete-prefixes", help="If some files are missing, skip the prefix", action="store_true")
    group3.add_argument("-r", "--refinement-level", help="refinement level of the equidistant mat-field")
    args = parser.parse_args()

    if args.directory is None:
        directory = './'
    else:
        directory = args.directory

    if directory[-1] != "/":
        directory = directory + '/'
    print("looking for files in dir: " + bcolors.HEADER + directory + bcolors.ENDC)

    # parse the filename to write to, as in previous versions, the default value
    # is ALL.xmf
    if args.outfile == None:
        args.outfile="ALL.mat"
    print("Output will be written to: "+bcolors.HEADER+args.outfile+bcolors.ENDC)

    # How shall we know at what time the file is? Sometimes, you'll end up with several
    # files at the same time, which have different file names. Then you'll want to
    # read the time from the filename, since paraview crashes if two files are at the
    # same instant.
    if args.time_by_fname:
        print("Time will be read from: "+bcolors.HEADER + "filename" + bcolors.ENDC)
    else:
        print("Time will be read from: "+bcolors.HEADER + "dataset" + bcolors.ENDC)

    # force unit spacing, dx=dy=dz=1 (useful for micro-ct data, occasionally)
    if args.unit_spacing:
        print("We will force unit spacing! dx=dy=dz=1 regardless of what is specified in h5 files")

    if args.ignore_origin:
        print("We will force origin = 0.0, 0.0, 0.0 regardless of what is specified in h5 files")
    if args.refinement_level == None:
        refinement_level = ""
        print("We will use the maximal refinement level of the data for refiment of coarser blocks")
    else:
        refinement_level = args.refinement_level

    # will vector recognition be turned off? This option is useful if for some reason
    # you have a file that ends with x is not a vector or if you downloaded just one
    # component
    if args.scalars:
        print(bcolors.HEADER + "Vector recongnition is turned OFF! All files treated as scalars." + bcolors.ENDC)

    # it happens that you want to ignore some files with a given prefix, for example
    # if you're missing some files or want to save on memory. the --exclude option
    # lets you specify a number of space-separated prefixes for the script to ignore.
    if args.exclude_prefixes is None:
        args.exclude_prefixes = []
    print("We will exclude the following prefixes: ", end='')
    print_list(args.exclude_prefixes)

    # ...
    if args.include_prefixes is None:
        args.include_prefixes = []
    print("We will include only the following prefixes: ", end='')
    print_list(args.include_prefixes)

    # on a large dataset of files, it may be useful to ignore some time steps
    # if you're not interested in them. The --exclude-timestamps option lets you do that
    if args.exclude_timestamps is None:
        args.exclude_timestamps = []
    print("We will exclude the following timestamps: ", end='')
    print_list(args.exclude_timestamps)

    # on a large dataset of files, it may be useful to use just some time steps
    # and ignore all other.
    if args.include_timestamps is None:
        args.include_timestamps = []
    print("We will include only the following timestamps: ", end='')
    print_list(args.include_timestamps)


    # will we generate one or many XMF files?
    if args.one_file_per_timestep:
        print("XMF: One file per timestep will be written")
    else:
        print("XMF: One file with all timesteps will be generated")

    #-------------------------------------------------------------------------------
    # get the list of all h5 files in the current directory.
    #-------------------------------------------------------------------------------
    print('-------------------------------------------------------------------')
    print("Looking for files...")
    # get the list of h5 files and sort them
    filelist = sorted( glob.glob(directory + "*.h5") )
    if not filelist:
        warn('No *.h5 files found')
        return
    print("We found " + bcolors.HEADER + "%i" % len(filelist) + bcolors.ENDC + " *.h5-files in directory")

    # initialize the list of prefixes
    prefixes = []
    vectors = []
    scalars = []
    filelist_used = []
    # initialize list of times
    times = []
    # mode can be either wabbit or flusi
    mode = "WABBIT"

    # loop over all h5 files, add their prefix and timestamp to a list
    for file in filelist:
        # read file
        f = h5py.File(file, 'r')
        # list all hdf5 datasets in the file - usually, we expect
        # to find only one.
        datasets = list(f.keys())

        if mode is None:
            # this message is also issued for FLUSI runtime backup files...
            warn('File: '+file+' seems to be neither WABBIT nor FLUSI data. Skip.')

        else:
            # prefix name
            prefix = get_dset_name(file)            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # judging from the dataset, do we use this file?
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # a priori, we'll not use this this
            used = False
            # Variant I: given list of exclude prefixes:
            if args.exclude_prefixes:
                # the --exclude-prefixe option helps us ignore some prefixes, for example if they
                # have incomplete data. so, if the current prefix is on that list, ignore it:
                if prefix not in args.exclude_prefixes:
                    # we used this file:
                    used = True

            # Variant II: given list of include prefixes:
            if args.include_prefixes:
                # the --include-prefixes option helps us focus on some prefixes, for example if they
                # have incomplete data. so, if the current prefix is on that list, ignore it:
                if prefix in args.include_prefixes:
                    # we used this file:
                    used = True

            # variant III: neither of both:
            if not args.exclude_prefixes and not args.include_prefixes:
                # we used this file:
                used = True

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # judging from the timestamp, do we use this file?
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if used:
                # now the condition for the dataset was met, we again suppose not
                # to use the file.
                used = False
                # get filename timestamp
                timestamp = get_timestamp( file )
                # variant I: given list of timestamps to exlude
                if args.exclude_timestamps:
                    if timestamp not in args.exclude_timestamps:
                        # we used this file:
                        used = True

                # variant II: given the list of timestamps
                if args.include_timestamps:
                    if timestamp in args.include_timestamps:
                        # we used this file:
                        used = True

                # variant III neither of both
                if not args.exclude_timestamps and not args.include_timestamps:
                    # we used this file:
                    used = True

            if used:
                # add to list of actually used files
                filelist_used.append(file)
                # store the dsetname in the list of prefixes. here, the entries are non-unique
                # we'll remove duplicates later.
                prefixes.append(prefix)

                # the option --scalars forces the code to ignore the trailing x,y,z icons
                # and treat all fields as scalars
                # vector / scalar handling: if it ends on {x,y,z} the prefix indicates a vector
                # otherwise, we deal with a scalar field.
                if prefix[len(prefix)-1:len(prefix)] in ['x','y','z'] and not args.scalars:
                    # add prefix name without trailing x,y,z to list of vectors
                    vectors.append( prefix[0:len(prefix)-1] )
                else:
                    # it's a scalar!
                    scalars.append(prefix)

    # remove duplicates
    prefixes = sorted( list(set(prefixes)) )
    vectors = sorted( list(set(vectors)) )
    scalars = sorted( list(set(scalars)) )

    #-------------------------------------------------------------------------------
    # check if vectors are complete, if not, add them to scalars (ux_00.h5 uy_00.h5 uz_00.h5)
    #-------------------------------------------------------------------------------
    for pre in vectors:
        if (pre+'x' in prefixes and pre+'y' in prefixes and pre+'z' in prefixes):
            print( pre+' is a 3D vector (x,y,z)')
        elif (pre+'x' in prefixes and pre+'y' in prefixes):
            print( pre+' is a 2D vector (x,y)')
        else:
            warn( pre+' is not a vector (its x-component is missing..)')
            vectors.remove( pre )
            if pre+'x' in prefixes:
                scalars.append(pre+'x')
            if pre+'y' in prefixes:
                scalars.append(pre+'y')
            if pre+'z' in prefixes:
                scalars.append(pre+'z')


    #-------------------------------------------------------------------------------
    # retrieve unique prefixes
    #-------------------------------------------------------------------------------
    print("We found the following prefixes: ", end='')
    print_list( prefixes )
    print("We found the following vectors: ", end='')
    print_list( vectors )
    print("We found the following scalars: ", end='')
    print_list( scalars )

    #-------------------------------------------------------------------------------
    # loop over all used files and extract timestamps
    #-------------------------------------------------------------------------------
    timestamps=[]
    for file in filelist_used:
        timestamps.append( get_timestamp( file ) )

    # retrieve unique timestamps
    timestamps = sorted( list(set(timestamps)) )
    print("We found the following timestamps: ", end='')
    print_list( timestamps )

    #-------------------------------------------------------------------------------
    # check if all files from the matrix exist
    #-------------------------------------------------------------------------------
    timestamps_to_remove, prefixes_to_remove =[], []

    for t in timestamps:
        for p in prefixes:
            # construct filename
            fname = directory + p + "_" + t + ".h5"
            if not os.path.isfile(fname):
                warn("File "+fname+ " NOT found!")

                # if desired, remove the timestamp from the list:
                if args.skip_incomplete_timestamps and t not in timestamps_to_remove:
                    warn("removing timestamp "+t+ " completely!")
                    timestamps_to_remove.append(t)

                # if desired, remove the prefix from the list:
                if args.skip_incomplete_prefixes:
                    warn("removing prefix "+p+ " completely!")
                    prefixes_to_remove.append(p)
                    if not args.scalars:
                        raise ValueError("Please use --skip_incomplete_prefixes (-l) only with --scalars (-q) ")

    for t in timestamps_to_remove:
        timestamps.remove(t)

    for p in prefixes_to_remove:
        prefixes.remove(p)

    print("We found the following timestamps: ", end='')
    print_list( timestamps )

    # we have now the timestamps as an ordered list, and the times array as an ordered list
    # however, if we exclude / include some files, the lists do not match, and we select files with the
    # wrong timestamp in the xmf file.
    times=[]
    for timestamp in timestamps:
        time = None

        # if desired, we read the actual data time from the filename and not from
        # the file. it sounds silly - but it proved to be very useful, if you have two files
        # at the same time in dataset but different file name. happens not often, but happens.
        if args.time_by_fname:
            # convert the string to a float, simply.
            time = float( timestamp )
        else:
            for p in prefixes:
                fname = directory + p + "_" + timestamp + ".h5"
                # read time from file
                f = h5py.File(fname, 'r')
                # dataset name depends on program
                dset_name = 'blocks'
                    #
                # get the dataset handle
                dset_id = f.get(dset_name)
                # from the dset handle, read the attributes
                tmp = dset_id.attrs.get('time')
                if time is None:
                    time = tmp
                else:
                    if abs(tmp-time) > 1.0e-5:
                        warn('It appears not all prefixes (with the same timestamp) are at the same time. consider using -n option. %s is at %f which is not %f' % (fname,tmp,time))

        # add time to the list of times.
        times.append( time )


    # check if the times are strictly increasing
    # PARAVIEW crashes with no clear error message if they dont
    if not strictly_increasing(times):
        print('-----------debug output----------')
        for t, tt in zip(timestamps, times):
            print("Timestamp %s time=%f" % (t,tt) )
        warn('List of times t is NOT monotonically increasing, this might cause PARAVIEW reader errors. Consider using the -n option')


    # warn if we re about to write an empty file
    if not prefixes or not timestamps:
        warn('No prefixes or timestamps..an empty file is created')

    print("The XMF file(s) refers to " + bcolors.HEADER + "%i" % (len(timestamps)*len(prefixes)) + bcolors.ENDC + " of these *.h5-files")

    if args.one_file_per_timestep:
        # extract base filename and extension
        fname, fext = os.path.splitext( args.outfile )
        # write one file per timestep
        for i in range(0, len(timestamps)):
            # construct filename
            outfile = fname + "_" + timestamps[i] + ".mat"
            print("writing " + outfile + "....")
        
            write_mat_file_wabbit( args, outfile, [times[i]], [timestamps[i]], prefixes, scalars, vectors, directory, level= refinement_level)
            
    else:
        # one file for the dataset
        # write the acual xmf file with the information extracted above
        print("writing " + args.outfile + "....")
        write_mat_file_wabbit( args, args.outfile, times, timestamps, prefixes, scalars, vectors, directory, level=refinement_level)
        
    print("Done. Enjoy!")

# i hate python:
# LIKE, THAT IS EASY!
if __name__ == "__main__":
    main()
