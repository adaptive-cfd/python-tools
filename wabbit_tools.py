#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:41:48 2017

@author: engels
"""
import numpy as np

class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        
        
def find_WABBIT_main_inifile(run_directory='./'):
    """
    find_WABBIT_main_inifile: In a folder, there are usually several INI files,
    one of which describes the simulation (this is the main one) and others (wing shape, etc).
    This routine figures out the main INI file in a folder.

    Parameters
    ----------
    run_directory : string
        Path of the simulation

    Raises
    ------
    ValueError
        If none is found, error is raised.

    Returns
    -------
    inifile : TYPE
        If found, this is the main INI file.

    """
    import glob
    
    found_main_inifile = False
    for inifile in glob.glob( run_directory+"/*.ini" ):
        section1 = exists_ini_section(inifile, 'Blocks')
        section2 = exists_ini_section(inifile, 'Insects')
        
        # if we find both sections, we likely found the INI file
        if section1 and section2:
            found_main_inifile = True
            print('Found simulations main INI file: '+inifile)
            return inifile
        
    if not found_main_inifile:
        raise ValueError("Did not find simulations main INI file - unable to proceed")
        
        

def warn( msg ):
    print( bcolors.WARNING + "WARNING! " + bcolors.ENDC + msg)

def err( msg ):
    print( bcolors.FAIL + "CRITICAL! " + bcolors.ENDC + msg)

def info( msg ):
    print( bcolors.OKBLUE + "Information:  " + bcolors.ENDC + msg)

#
def check_parameters_for_stupid_errors( file ):
    """
    For a given WABBIT parameter file, check for the most common stupid errors
    the user can commit: Jmax<Jmain, negative time steps, etc.
    Ths function should be used on supercomputers at every job submission; I added
    it to my launch wrappers.
    
    Input:
    ------
    
        file:  string
            The file to be checked
    
    Output:
    -------
        Warnings are printed on screen directly.
    
    """
    import os
    
    # print('~~~~~~~~~~~~~~~~~~~~~ini-file~~~~~~~~~~~')
    # # read jobfile
    # with open(file) as f:
    #     # loop over all lines
    #     for line in f:
    #         line = line.lstrip()
    #         line = line.rstrip()
    #         if len(line)>0:
    #             if ';' in line:
    #                 line = line[0:line.index(";")]
    #             if len(line)>0:
    #                 if '[' in line and ']' in line:
    #                     print(bcolors.OKBLUE + line + bcolors.ENDC)
    #                 else:
    #                     print(line)
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    print("We scan %s for stupid errors." % (file) )

    # check if the file exists, at least
    if not os.path.isfile(file):
        raise ValueError("Stupidest error of all: we did not find the INI file.")

    wavelet         = get_ini_parameter(file, 'Wavelet', 'wavelet', str, default="CDF40")        
        
    # since 05 Jul 2023, g is set automatically, unless we do something stupid.
    if wavelet == 'CDF20':
        g_default = 2
    elif wavelet=='CDF22':
        g_default = 3
    elif wavelet=='CDF40':
        g_default = 4
    elif wavelet=='CDF42':
        g_default = 5
    elif wavelet=='CDF44' or wavelet=='CDF62':
        g_default = 7
    else:
        g_default = 1
        
    jmax            = get_ini_parameter(file, 'Blocks', 'max_treelevel', int)
    jmin            = get_ini_parameter(file, 'Blocks', 'min_treelevel', int, default=1)
    adapt_mesh      = get_ini_parameter(file, 'Blocks', 'adapt_tree', int, default=1)
    ceps            = get_ini_parameter(file, 'Blocks', 'eps')
    bs              = get_ini_parameter(file, 'Blocks', 'number_block_nodes', int, vector=True)
    g               = get_ini_parameter(file, 'Blocks', 'number_ghost_nodes', int, default=g_default)
    g_rhs           = get_ini_parameter(file, 'Blocks', 'number_ghost_nodes_rhs', int, default=g)
    dealias         = get_ini_parameter(file, 'Blocks', 'force_maxlevel_dealiasing', int)
    Neqn            = get_ini_parameter(file, 'Blocks', 'number_equations', int)
    dim             = get_ini_parameter(file, 'Domain', 'dim', int)
    L               = get_ini_parameter(file, 'Domain', 'domain_size', vector=True)
    discretization  = get_ini_parameter(file, 'Discretization', 'order_discretization', str)
    physics_type    = get_ini_parameter(file, 'Physics', 'physics_type', str)
    time_step_method = get_ini_parameter( file, 'Time', 'time_step_method', str, default="RungeKuttaGeneric")
    CFL              = get_ini_parameter( file, 'Time', 'CFL', float, default=1.0)
    CFL_eta          = get_ini_parameter( file, 'Time', 'CFL_eta', float, default=0.99)
    CFL_nu           = get_ini_parameter( file, 'Time', 'CFL_nu', float, default=0.99*2.79/(float(dim)*np.pi**2))
    c0           = get_ini_parameter( file, 'ACM-new', 'c_0', float)
    nu           = get_ini_parameter( file, 'ACM-new', 'nu', float)
    ceta         = get_ini_parameter( file, 'VPM', 'C_eta', float, default=0.0)
    penalized    = get_ini_parameter( file, 'VPM', 'penalization', bool, default=False)
    geometry     = get_ini_parameter( file, 'VPM', 'geometry', str, default='default')
    sponged      = get_ini_parameter( file, 'Sponge', 'use_sponge', bool, default=False)
    csponge      = get_ini_parameter( file, 'Sponge', 'C_sponge', float, default=0.0)
    sponge_type  = get_ini_parameter( file, 'Sponge', 'sponge_type', str, default='default')
    L_sponge     = get_ini_parameter( file, 'Sponge', 'L_sponge', default=0.0)
    time_max     = get_ini_parameter( file, 'Time', 'time_max', float)
    time_stepper = get_ini_parameter( file, 'Time', 'time_step_method', str, default="RungeKuttaGeneric")
    CFL          = get_ini_parameter( file, 'Time', 'CFL', float, default=0.5)
    CFL_nu       = get_ini_parameter( file, 'Time', 'CFL_nu', float, default=0.99*2.79/(float(dim)*np.pi**2) )
    CFL_eta      = get_ini_parameter( file, 'Time', 'CFL_eta', float, default=0.99)
    filter_type  = get_ini_parameter( file, 'Discretization', 'filter_type', str, default='no_filter')
    filter_freq  = get_ini_parameter( file, 'Discretization', 'filter_freq', int, default=-1)
    
    
    dx = L[0]*2**-jmax/(bs[0])
    keta = np.sqrt(ceta*nu)/dx
    
    
    print("======================================================================================")
    print("Bs= %i   g= %i  g_rhs= %i   dim= %i   Jmax= %i   L= %2.2f %s==> dx= %2.3e   N_equi= %i   N= %i per unit length%s" % 
          (bs[0],g,g_rhs, dim,jmax,L[0],bcolors.OKBLUE, dx, int(L[0]/dx), int(1.0/dx), bcolors.ENDC))
    print("equidistant grids: Jmin=%i^%i, Jmax=%i^%i" % (int(bs[0]*2**jmin), dim, int(bs[0]*2**jmax), dim) )
    print("discretization= %s" % (discretization))
    print("T_max = %2.2f   CFL= %2.2f   CFL_eta= %2.2f   CFL_nu= %2.3f   time_stepper= %s" % (time_max, CFL, CFL_eta, CFL_nu, time_stepper))
    
    
    print("use_penalization= %i   geometry= %s   C_eta= %2.2e %s    ==> K_eta = %2.2f%s" % 
          (penalized, geometry, ceta, bcolors.OKBLUE, keta, bcolors.ENDC))
    if sponged:
        print("use_sponge=%i   type=%s   C_sponge=%2.2e   L_sponge=%2.2f %s==> Ntau  = %2.2f%s" % 
              (sponged, sponge_type, csponge, L_sponge, bcolors.OKBLUE, L_sponge/(c0*csponge), bcolors.ENDC))
    print("C_0   = %2.2f   delta_shock= %2.2f dx     nu=%e" % (c0, c0*ceta/dx, nu))
    print("C_eps = %2.2e   wavelet= %s    dealias=%i    adapt_mesh=%i" % (ceps, wavelet, dealias, adapt_mesh))
    
    print("dt_CFL= %2.3e" % (CFL*dx/c0))
    print("filter_type= %s filter_freq=%i" % (filter_type, filter_freq))
    
    if geometry == "Insect":
        h_wing = get_ini_parameter( file, 'Insects', 'WingThickness', float, 0.0)
        print('--- insect ----')
        print('h_wing/dx = %2.2f' % (h_wing/dx))
    
    print("======================================================================================")
    
    if physics_type == 'ACM-new' and dim == 3 and Neqn != 4:
        err("For 3D ACM, you MUST set number_equations=4 (ux,uy,uz,p)")
        
    if physics_type == 'ACM-new' and dim == 2 and Neqn != 3:
        err("For 2D ACM, you MUST set number_equations=3 (ux,uy,p)")
    
    if len(bs) > 1:
        bs = bs[0]

    if bs % 2 != 0:
        warn('The block size is bs=%i which is an ODD number.' % (bs) )

    if bs < 3:
        warn('The block size is bs=%i is very small or even negative.' % (bs) )        
          
    if (wavelet == "CDF22") and g<3:
        warn("Not enough ghost nodes for wavelet %s g=%i < 3" % (wavelet, g) )
    if (wavelet == "CDF42") and g<5:
        warn("Not enough ghost nodes for wavelet %s g=%i < 5" % (wavelet, g) )        
    if (wavelet == "CDF44" or wavelet == "CDF62") and g<7:
        warn("Not enough ghost nodes for wavelet %s g=%i < 7" % (wavelet, g) )
    if (wavelet == "CDF40") and g<4:
        warn("Not enough ghost nodes for wavelet %s g=%i < 4" % (wavelet, g) )
        
   
    if time_step_method == "RungeKuttaChebychev":
        if CFL_eta < 999:
            warn('are you sure you did not forget to adjustl CFL_eta for the RKC scheme???')
        if CFL_nu < 999:
            warn('are you sure you did not forget to adjustl CFL_nu for the RKC scheme???')
        if CFL != 0.75:
            warn('are you sure you did not forget to adjustl CFL for the RKC scheme??? often we used 0.75.')    
            
    if time_step_method == "RungeKuttaGeneric":
        if CFL_eta > 1.0:
            warn('are you sure you did not forget to adjustl CFL_eta for the RK scheme? it may be unstable.')
        if CFL_nu > 0.99*2.79/(float(dim)*np.pi**2):
            warn('are you sure you did not forget to adjustl CFL_nu for the RK scheme? it may be unstable.')
        if CFL >  1.0:
            warn('are you sure you did not forget to adjustl CFL for the RK scheme? it may be unstable.')    
            
    # if somebody modifies the standard parameter file, users have to update their
    # ini files they use. this is often forgoten and obnoxious. Hence, if we find
    # value sthat no longer exist, warn the user.
    if exists_ini_parameter( file, "Blocks", "number_data_fields" ) :
        warn('Found deprecated parameter: [Blocks]::number_data_fields')

    if exists_ini_parameter( file, "Physics", "initial_cond" ) :
        warn('Found deprecated parameter: [Physics]::initial_cond')

    if exists_ini_parameter( file, "Dimensionality", "dim" ) :
        warn('Found deprecated parameter: [Dimensionality]::dim')

    if exists_ini_parameter( file, "DomainSize", "Lx" ) :
        warn('Found deprecated parameter: [DomainSize]::Lx')

    if exists_ini_parameter( file, "Time", "time_step_calc" ) :
        warn('Found deprecated parameter: [Time]::time_step_calc')
        
    if exists_ini_parameter( file, "ACM", "forcing" ):
        warn('Found deprecated parameter: [ACM]::forcing')
        
    if exists_ini_parameter( file, "ACM", "forcing_type" ):
        warn('Found deprecated parameter: [ACM]::forcing_type')
        
    if exists_ini_parameter( file, "ACM", "p_mean_zero" ):
        warn('Found deprecated parameter: [ACM]::p_mean_zero')
        
    if exists_ini_parameter( file, "ACM", "compute_laplacian" ):
        warn('Found deprecated parameter: [ACM]::compute_laplacian')
        
    if exists_ini_parameter( file, "ACM", "compute_nonlinearity" ):
        warn('Found deprecated parameter: [ACM]::compute_nonlinearity')
    
    if exists_ini_parameter( file, "Blocks", "adapt_mesh" ):
        warn('Found deprecated parameter: [Blocks]::adapt_mesh ===> adapt_tree')
   
    HIT = get_ini_parameter( file, 'ACM-new', 'use_HIT_linear_forcing', bool, default=False)
    if HIT:
        print(type(HIT))
        print(HIT)
        warn('You use HIT linear forcing, which is HIGHLY EXPERIMENTAL')

    jmax = get_ini_parameter( file, 'Blocks', 'max_treelevel', int)

    if jmax > 18:
        warn('WABBIT can compute at most 18 refinement levels, you set more!')

    if sponged:
        # default value is TRUE so if not found, all is well
        mask_time_dependent = get_ini_parameter( file, 'VPM', 'mask_time_dependent_part', int, default=1)

        if mask_time_dependent != 1:
            warn("""you use sponge, but mask_time_dependent_part=0! The sponge
            is treated as if it were time dependent because it does not have
            to be at the maximum refinement level.""")



    # loop over ini file and check that each non-commented line with a "=" contains the trailing semicolon ";"
    with open(file) as f:
        # loop over all lines
        linenumber = 0
        for line in f:
            # remove trailing & leading spaces
            line = line.strip()
            linenumber += 1
            if line != "" :
                if line[0] != "!" and line[0] != "#" and line[0] != ";" :
                    if "=" in line and ";" not in line:
                        warn('It appears the line #%i does not contain the semicolon' % (linenumber) )

    restart = get_ini_parameter( file, 'Physics', 'read_from_files', int)
    print("read_from_files=%i" %(restart))

    if restart == 1:
        info("This simulation is being resumed from file")

        infiles = get_ini_parameter( file, 'Physics', 'input_files', str)
        infiles = infiles.split()
        for file in infiles:
            print(file)
            if not os.path.isfile(file):
                raise ValueError("CRUTIAL: read_from_files=1 but infiles NOT found!.")
    else:
        info("This simulation is being started from initial condition (and not from file)")

#
def get_ini_parameter( inifile, section, keyword, dtype=float, vector=False, default=None, matrix=False, verbose=False ):
    """
    From a given ini file, read [Section]::keyword and return the value
    If the value is not found, an error is raised, if no default is given.
    
    Input:
    ------
    
        inifile:  string
            File to read from
        section: string
            Section to find in file that contains the value eg "Hallo" to find in [Hallo]
        keyword: string
            Variable name to find (e.g. [Hallo]  param=1;)
        dtype: type
            return value data type
        vector: bool
            If true, we allow returning vectors (eg param=1 2 2 3; in the file), without the option, the conversion to
            dtype likely fails. Vectors can also be in the syntax:
                vct=1 2 2;   \n 
                vct=1,2 2;        \n             
                vct=(/1 2 2/)   \n                 
                vct=(/1,2,2/)   \n 
            because the fortran code can read both with and without commas.
        matrix: bool
            If true, return a matrix. A matrix is a vector that spans more than one line. It cannot be read using
            the default python CONFIGPARSER module, so we do it manually here.
            The matrix format is:
                matrix=(/ 2 ,3 4,4\n
                8,3,3,5\n
                8,2,2,2/)\n
            Again, the use of commas is optional. The fortran code seems to be more restrictive: I think it
            can only read values separated by a SINGLE SPACE. If one row has a different length, reading will fail.
        default: value
            If the entry in ini file is not found, this value is returned instead. If no default is given, not finding raises ValueError
        
    
    Output:
    -------
        the value, maybe an np.ndarray
    
    """
    import configparser
    import os
    import numpy as np

    # check if the file exists, at least
    if not os.path.isfile(inifile):
        raise ValueError("Stupidest error of all: we did not find the INI file.")


    # a matrix is something that starts with (/ (FORTRAN style) and it extends
    # over many lines. this is incompatible with the python ini files parser
    if matrix:
        fid = open( inifile, 'r')      
        found_section, found = False, False
        rows = []
        
        for line in fid:
            # remove leading and trailing spaces from line
            line = line.strip()
            
            # this will read the second and following rows.
            if found == True:
                # is this the last line of the matrix?
                if '/)' in line:
                    found = False
                # some vectors are separated by commas ',', remove them.
                line = line.replace(',', ' ')
                line = line.replace('/)', '')
                line = line.replace(';', '')
                rows.append( [float(i) for i in line.split()] )
                
            if line == "":
                line = "blabla"
            
            # is the line commented out?
            if (line[0] == ";" or line[0] == "!" or line[0] == "#"):
                line = "blabla"
                
            # did we find the section?
            if '['+section+']' in line:
                found_section = True
                
            # first row, if found keyword.
            if found_section:
                if keyword+"=" in line:
                    
                    if not '(/' in line:
                        raise ValueError("You try to read a matrix, and we found the keyword, but it does not seem to be a matrix..")
                    
                    # remove first vct=
                    line = line.replace(keyword+"=", "")
                    # remove (/
                    line = line.replace('(/', '')                                        
                    # some vectors are separated by commas ',', remove them.
                    line = line.replace(',', ' ')                    
                    # next couple of line is all matrix elements
                    found = True
                    # this is the first row
                    rows.append( [float(i) for i in line.split()] )
        fid.close()
        
        # convert 
        matrix = np.array( rows, dtype=float )
        return matrix


    # 2022: We do no longer use the buildin ini files parser, because it has trouble 
    # with MATRIX definitions. Unfortunately, we used an incompatible syntax (I regret this very much)
    
    # -------------------------------------------------------------------------
    # # initialize parser object
    # config = configparser.ConfigParser(allow_no_value=True)
    # # read (parse) inifile.
    # config.read(inifile)

    # # use configparser to find the value
    # value_string = config.get( section, keyword, fallback='UNKNOWN')
    # -------------------------------------------------------------------------
      
    
    value_string = 'UNKNOWN'
    
    fid = open( inifile, 'r')      
    found_section = False
    
    for line in fid:
        # remove leading and trailing spaces from line
        line = line.strip()
        
        # remove trailing comments (including the ';')
        if ';' in line:
            line = line[0:line.index(';')]
              
        if line == "":
            line = "blabla"
        
        # is the line commented out?
        if (line[0] == ";" or line[0] == "!" or line[0] == "#"):
            line = "blabla"
            
        # did we find the section?
        if '['+section+']' in line:
            found_section = True
            continue
        
        # if we found the section previously, and we now find a DIFFERENT one, well,
        # then we did not find the keyword
        if found_section and ('[' in line and ']' in line):
            found_section = False
            break
            
        # first row, if found keyword.
        if found_section:
            if keyword+"=" in line:
                # remove first vct=
                line = line.replace(keyword+"=", "")
                
                # this is the result...
                value_string = line
                # we're done
                break
    fid.close()
        
    
    
    # check if that worked. If not value is found, UNKNOWN is returned. if the value field
    # is empty, then ";" is returned
    if (value_string==";" or 'UNKNOWN' in value_string) and default is None:
        raise ValueError("NOT FOUND! file=%s section=%s keyword=%s" % (inifile, section, keyword) )

    if (value_string==";" or 'UNKNOWN' in value_string or value_string=='') and default is not None:
        if verbose:
            print("Returning default!")
        return dtype(default)

   

    if not vector:
        if verbose:
            print(value_string)
            
        # configparser returns "0.0;" so remove trailing ";"
        if ";" in value_string:
            i = value_string.find(';')
            value_string = value_string[:i]
            
        if dtype is bool:
            if value_string=="1" or value_string=="yes":
                return True
            else:
                return False
        
        return dtype(value_string)
    else:
                       
        # remove everything after ; character
        if ";" in value_string:
            i = value_string.find(';')
            value_string = value_string[:i]
                
        # some vectors are written as vect=(/2, 3, 4/)
        value_string = value_string.replace('(/', '')
        value_string = value_string.replace('/)', '')
        
        # some vectors are separated by commas ',', remove them.
        value_string = value_string.replace(',', ' ')
        
        # you can use the strip() to remove trailing and leading spaces.
        value_string.strip()
        l = value_string.split()
        return np.asarray( [float(i) for i in l] )



#
def exists_ini_parameter( inifile, section, keyword ):
    """ check if a given parameter in the ini file exists or not. can be used to detect
        deprecated entries somebody removed
    """
    found_section = False
    found_parameter = False

    # read jobfile
    with open(inifile) as f:
        # loop over all lines
        for line in f:

            # once found, do not run to next section
            if found_section and line[0] == "[":
                found_section = False

            # until we find the section
            if "["+section+"]" in line:
                found_section = True

            # only if were in the right section the keyword counts
            if found_section and keyword+"=" in line:
                found_parameter = True

    return found_parameter

#
def exists_ini_section( inifile, section ):
    """ check if a given parameter in the ini file exists or not. can be used to detect
        deprecated entries somebody removed
    """
    found_section = False

    # read jobfile
    with open(inifile) as f:
        # loop over all lines
        for line in f:
            # until we find the section
            if "["+section+"]" in line and line[0]!=";" and line[0]!="!" and line[0]!="#":
                found_section = True


    return found_section


def replace_ini_value(file, section, keyword, new_value):
    """
    replace ini value: Sets a value in an INI file. Useful for scripting of preprocessing
    

    Parameters
    ----------
    file : string
        The INI file
    section : string
        Parameter file section.
    keyword : string
        Actual parameter to set/change.
    new_value : string
        The new value. Note we use strings here but WABBIT/FLUSI may interpret the value as number

    Returns
    -------
    None.

    """
    found_section, found_keyword = False, False
    i = 0


    with open(file, 'r') as f:
        # read a list of lines into data
        data = f.readlines()
       
        

    # loop over all lines
    for line in data:
        line = line.lstrip().rstrip()
        if len(line) > 0:
            if line[0] != ';':
                if '['+section+']' in line:
                    found_section = True
                    
                if ';' in line:
                    line_nocomments = line[0:line.index(';')]
                else:
                    line_nocomments = ""
                
                    
                if '[' in line_nocomments and ']' in line_nocomments and not '['+section+']' in line_nocomments and found_section:
                    # left section again
                    found_section = False         
                    break
                    
                if keyword+'=' in line and found_section:
                    # found keyword in section
                    found_keyword = True
                    old_value = line[ line.index(keyword+"="):line.index(";") ]
     
                    line = line.replace(old_value, keyword+'='+new_value)
                    data[i] = line+'\n'
                    
                    print("changed: "+old_value+" to: "+keyword+'='+new_value)
                    break
        i += 1
       
                    
    if found_keyword:
        # .... and write everything back
        with open(file, 'w') as f:
            f.writelines( data )


#
def prepare_resuming_backup( inifile ):
    """ we look for the latest *.h5 files
        to resume the simulation, and prepare the INI file accordingly.
        Some errors are caught.
    """
    import numpy as np
    import os
    import glob
    import flusi_tools

    # does the ini file exist?
    if not os.path.isfile(inifile):
        raise ValueError("Inifile not found!")

    Tmax = get_ini_parameter(inifile, "Time", "time_max", float)
    dim  = get_ini_parameter(inifile, "Domain", "dim", int)

    # This code currenty only works with ACMs
    physics_type = get_ini_parameter(inifile, "Physics", "physics_type", str)

    if physics_type != "ACM-new":
        raise ValueError("ERROR! backup resuming is available only for ACM")


    if dim == 2:
        state_vector_prefixes = ['ux', 'uy', 'p']
    else:
        state_vector_prefixes = ['ux', 'uy', 'uz', 'p']
    


    # if used, take care of passive scalar as well
    if exists_ini_parameter( inifile, 'ACM-new', 'use_passive_scalar' ):
        scalar = get_ini_parameter(inifile, 'ACM-new', 'use_passive_scalar', bool, default=False)
        if scalar:
            n_scalars = get_ini_parameter(inifile, 'ConvectionDiffusion', 'N_scalars', int, default=0)

            for i in range(n_scalars):
                state_vector_prefixes.append( "scalar%i" % (i+1) )



    # find list of H5 files for first prefix.
    files = glob.glob( state_vector_prefixes[0] + "*.h5" )
    files.sort()

    if not files:
        raise ValueError( "Something is wrong: no h5 files found for resuming" )
        
    # first, we try the latest snapshots (obviously)
    # it can happen (disk quota) that the code cannot complete writing this backup.
    index = -1
    timestamp = flusi_tools.get_timestamp_name( files[index] )
    t0 = float(timestamp) / 1e6
    
    # is this complete ? 
    snapshot_complete = True
    for prefix in state_vector_prefixes:
        if not os.path.isfile( prefix + '_' + timestamp + '.h5'):
            snapshot_complete = False
            print('For snapshot %s we did not find %s!! -> trying another one' % (timestamp, prefix))
    
    # if not, we try the second latest, if it exists
    if not snapshot_complete:
        if len(files) >= 2:
            index = -2
            timestamp = flusi_tools.get_timestamp_name( files[index] )
            t0 = float(timestamp) / 1e6
            
            snapshot_complete = True
            for prefix in state_vector_prefixes:
                if not os.path.isfile( prefix + '_' + timestamp + '.h5'):
                    snapshot_complete = False
                    print('For snapshot %s we did not find all required input files!! -> trying another one' % (timestamp))
            
        else:
            raise ValueError("We did not find a complete snapshot to resume from...you'll have to start over.")
   
    # if we still were unable to resume...well, then its time to give up (if both snapshots are incomplete, you may have forgotten
    # to save enough data, simply)
    if not snapshot_complete:
        raise ValueError("We did not find a complete snapshot to resume from (tried -1 and -2)...you'll have to start over.")

    print('Latest file is:           ' + files[index])    
    print('Latest file is at time:   %f' % (t0))

    # if we find the dt.t file, we now at what time the job ended.
    # otherwise, just resume the latest H5 files
    if os.path.isfile('dt.t'):

        d = np.loadtxt('dt.t')
        t1 = d[-1,0]
        print('Last time stamp in logs is: %f' % (t1))

        # time check when resuming a backup
        if t0 > t1:
            print( "Something is wrong: the latest H5 file is at LATER time than the log files. Is this the right data?" )

        if t0 < 1.0e-6:
            print("Something is wrong: the latest H5 file is almost at t=0. That means no backup has been saved?" )

        if t1 > t0:
            print('Warning: the latest H5 file is younger than the last entry in the log: we will have to compute some times twice.')

        if abs(t1-t0) < 1.0e-4:
            print('Good news: timestamp in H5 file and time in log file match!')

        if t1 >= 0.9999*Tmax or t0 >= 0.9999*Tmax:
            raise ValueError( "Something is wrong: the run seems to be already finnished!" )

    # check if all required input files exist
    for prefix in state_vector_prefixes:
        if not os.path.isfile( prefix + '_' + timestamp + '.h5'):
            raise ValueError( "file not found!!!! " + prefix + '_' + timestamp + '.h5' )


    # create the string we will put in the ini file
    infiles_string = ""
    for prefix in state_vector_prefixes:
        infiles_string += prefix + '_' + timestamp + '.h5' + ' '

    # remove trailing space:
    infiles_string = infiles_string.strip()
    # add colon
    infiles_string += ';'
    # information (debug)
    print(infiles_string)

    f1 = open( inifile, 'r')
    f2 = open( inifile+'.tmptmp', 'w')
    found, okay1, okay2 = False, False, False

    for line in f1:
        # remove trailing space:
        line_cpy = line.strip()

        if '[Physics]' in line_cpy:
            found = True

        if 'read_from_files=' in line_cpy and found and line_cpy[0] != ";":
            line = "read_from_files=1;\n"
            okay1 = True

        if 'input_files=' in line_cpy and found and line_cpy[0] != ";":
            line = "input_files=" + infiles_string + "\n"
            okay2 = True

        f2.write( line )

    f1.close()
    f2.close()

    if okay1 and okay2:
        os.rename( inifile+'.tmptmp', inifile )


#
def block_level_distribution_file( file ):
    """ Read a 2D/3D wabbit file and return a list of how many blocks are at the different levels
    """
    import h5py
    import numpy as np

    # open the h5 wabbit file
    fid = h5py.File(file,'r')

    # read treecode table
    b = fid['block_treecode'][:]
    treecode = np.array(b, dtype=float)

    # close file
    fid.close()

    # number of blocks
    Nb = treecode.shape[0]

    # min/max level. required to allocate list!
    jmin, jmax = get_max_min_level( treecode )
    counter = np.zeros(jmax+1)

    # fetch level for each block and count
    for i in range(Nb):
        J = treecode_level(treecode[i,:])
        counter[J] += 1

    return counter


#
def read_wabbit_hdf5(file, verbose=True, return_iteration=False):
    """ Read a wabbit-type HDF5 of block-structured data.

    Return time, x0, dx, box, data, treecode.
    Get number of blocks and blocksize as

    N, Bs = data.shape[0], data.shape[1]
    """
    import h5py
    import numpy as np

    if verbose:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Reading file %s" % (file) )

    fid = h5py.File(file,'r')
    b = fid['coords_origin'][:]
    x0 = np.array(b, dtype=float)

    b = fid['coords_spacing'][:]
    dx = np.array(b, dtype=float)

    b = fid['blocks'][:]
    data = np.array(b, dtype=float)

    b = fid['block_treecode'][:]
    treecode = np.array(b, dtype=float)

    # get the dataset handle
    dset_id = fid.get('blocks')
    
    # from the dset handle, read the attributes
    time = dset_id.attrs.get('time')
    iteration = dset_id.attrs.get('iteration')
    box = dset_id.attrs.get('domain-size')
    version=dset_id.attrs.get('version')


    fid.close()

    jmin, jmax = get_max_min_level( treecode )
    N = data.shape[0]
    Bs = data.shape[1:]
    Bs = np.asarray(Bs[::-1]) # we have to flip the array since hdf5 stores in [Nz, Ny, Nx] order
    
    if version == 20200408 or version == 20231602:
        Bs = Bs-1
        #print("!!!Warning old (old branch: newGhostNodes) version of wabbit format detected!!!")
    else:
        print("This file includes redundant points")
        
    if verbose:
        print("Time=%e it=%i N=%i Bs[0]=%i Bs[1]=%i Jmin=%i Jmax=%i" % (time, iteration, N, Bs[0], Bs[1], jmin, jmax) )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")


    if return_iteration:
        return time, x0, dx, box, data, treecode, iteration[0]
    else:
        return time, x0, dx, box, data, treecode

#
def read_treecode_hdf5(file):
    """ Read a wabbit-type HDF5 of block-structured data.
    same as read_wabbit_hdf5, but reads ONLY the treecode array.
    """
    import h5py
    import numpy as np

    fid = h5py.File(file,'r')

    b = fid['block_treecode'][:]
    treecode = np.array(b, dtype=float)

    return treecode

#
def write_wabbit_hdf5( file, time, x0, dx, box, data, treecode, iteration = 0, dtype=np.float64  ):
    """ Write data from wabbit to an HDF5 file
        Note: hdf5 saves the arrays in [Nz, Ny, Nx] order!
        So: data.shape = Nblocks, Bs[3], Bs[2], Bs[1]
    """
    import h5py
    import numpy as np


    Level = np.size(treecode,1)
    if len(data.shape)==4:
        # 3d data
        Bs = np.zeros([3,1])
        N, Bs[0], Bs[1], Bs[2] = data.shape
        Bs = Bs[::-1]
        print( "Writing to file=%s max=%e min=%e size=%i %i %i " % (file, np.max(data), np.min(data), Bs[0], Bs[1], Bs[2]) )

    else:
        # 2d data
        Bs = np.zeros([2,1])
        N, Bs[0], Bs[1] = data.shape
        Bs = Bs[::-1]
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Writing file %s" % (file) )
        print("Time=%e it=%i N=%i Bs[0]=%i Bs[1]=%i Level=%i Domain=[%d, %d]" % (time, iteration, N, Bs[0], Bs[1],Level, box[0], box[1]) )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~")


    fid = h5py.File( file, 'w')

    fid.create_dataset( 'coords_origin', data=x0, dtype=dtype )
    fid.create_dataset( 'coords_spacing', data=dx, dtype=dtype )
    fid.create_dataset( 'blocks', data=data, dtype=dtype )
    fid.create_dataset( 'block_treecode', data=treecode, dtype=int )

    fid.close()

    fid = h5py.File(file,'a')
    dset_id = fid.get( 'blocks' )
    dset_id.attrs.create( "version", 20231602) # this is used to distinguish wabbit file formats
    dset_id.attrs.create('time', time, dtype=dtype)
    dset_id.attrs.create('iteration', iteration)
    dset_id.attrs.create('domain-size', box, dtype=dtype )
    dset_id.attrs.create('total_number_blocks', N )
    fid.close()

#


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
    data={'time': [],'x0':[],'dx':[],'treecode':[]}
    # we loop over all files in the given directory
    for file in os.listdir(dir):
        # filter out the good ones (ending with .h5)
        if file.endswith(".h5"):
            # from the file we can get the fieldname
            fieldname=re.split('_',file)[0]
            print(fieldname)
            time, x0, dx, box, field, treecode = read_wabbit_hdf5(os.path.join(dir, file))
            #increase the counter
            data['time'].append(time[0])
            data['x0'].append(x0)
            data['dx'].append(dx)
            data['treecode'].append(treecode)
            if fieldname not in data:
                # add the new field to the dictionary
                data[fieldname]=[]
                data[fieldname].append(field)
            else: # append the field to the existing data field
                data[fieldname].append(field)
            it=it+1
    # the size of the domain
    data['box']=box
    #return time, x0, dx, box, data, treecode
    return data



def add_convergence_labels(dx, er):
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

    for i in range(len(dx)-1):
        x = 10**( 0.5 * ( np.log10(dx[i]) + np.log10(dx[i+1]) ) )
        y = 10**( 0.5 * ( np.log10(er[i]) + np.log10(er[i+1]) ) )
        order = "%2.1f" % ( convergence_order(dx[i:i+1+1],er[i:i+1+1]) )
        plt.text(x, y, order, horizontalalignment='center', verticalalignment='center',
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
        A[i,0] = N[i]
        B[i] = err[i]

    x, residuals, rank, singval  = np.linalg.lstsq(A, B, rcond=None)

    return x


def plot_wabbit_dir(d, **kwargs):
    import glob

    files = glob.glob(d+'/*.h5')
    files.sort()
    for file in files:
        plot_wabbit_file(file, **kwargs)


# given a treecode tc, return its level
def treecode_level( tc ):
    level = 0
    for k in range(len(tc)):        
        if (tc[k] >= 0):
            level += 1
        else:
            break
    return(level)



# for a treecode list, return max and min level found
def get_max_min_level( treecode ):

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
def plot_1d_cut( file, y ):
    # read data
    time, x0, dx, box, data, treecode = read_wabbit_hdf5( file )

    # get number of blocks and blocksize
    N, Bs = data.shape[0], data.shape[1:]

    dim = len( data.shape )-1
    
    if dim != 2:
        raise ValueError("Sadly, we do this only for 2D fields right now")
        
    if y < 0.0 or y >= box[1]:
        raise ValueError("Sadly, you request a y value out of the domain..")
        
    y_found = []
    # first check if any blocks contain the y value at all 
    for i in range(N):
        y_vct = np.arange(Bs[0])*dx[i,0] + x0[i,0]
        
        if np.min( np.abs(y_vct-y) ) < dx[i,1]/2.0:
            iy = np.argmin( np.abs(y_vct-y) )
            y_found.append( y_vct[iy] )
            
    print(y_found)
    y_new = y_found[0]
    print('snapped to y=%f' % (y_new))
    
    x_values, f_values = [],[]
    
    for i in range(N):
        
        x_vct = np.arange(Bs[1])*dx[i,1] + x0[i,1]
        y_vct = np.arange(Bs[0])*dx[i,0] + x0[i,0]
        
        if np.min( np.abs(y_vct-y_new) ) < dx[i,0]/100.0:
            iy = np.argmin( np.abs(y_vct-y) )
            x_values.append( x_vct )
            f_values.append( data[i,iy,:].copy() )
            
    x_values = np.hstack(x_values)
    f_values = np.hstack(f_values)
    
    x_values, f_values = zip(*sorted(zip(x_values, f_values)))
           
    return x_values, f_values
#    import matplotlib.pyplot as plt
#    plt.figure()
#    plt.plot(x_values, f_values, '-')

#
def plot_wabbit_file( file, savepng=False, savepdf=False, cmap='rainbow', caxis=None,
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
        filename_pdf = file.replace('h5','pdf')
    if filename_png is None:
        filename_png = file.replace('h5','png')

    
    cb = []
    # read procs table, if we want to draw the grid only
    if gridonly:
        fid = h5py.File(file,'r')

        # read procs array from file
        b = fid['procs'][:]
        procs = np.array(b, dtype=float)

        if gridonly_coloring in ['refinement-status', 'refinement_status']:
            b = fid['refinement_status'][:]
            ref_status = np.array(b, dtype=float)

        if gridonly_coloring == 'lgt_id':
            b = fid['lgt_ids'][:]
            lgt_ids = np.array(b, dtype=float)
            
        fid.close()

    # read data
    time, x0, dx, box, data, treecode = read_wabbit_hdf5( file )

    # get number of blocks and blocksize
    N, Bs = data.shape[0], data.shape[1:]

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
    jmin, jmax = get_max_min_level( treecode )



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
                level = treecode_level( treecode[i,:] )
                if (jmax-jmin>0):
                    c = 0.9 - 0.75*(level-jmin)/(jmax-jmin)
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
                    tag = ""
                    for jj in range(treecode.shape[1]):
                        if treecode[i,jj] != -1:
                            tag += "%1.1i" % treecode[i,jj]

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
                raise ValueError("ERROR! The value for gridonly_coloring is unkown")

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
            plt.savefig( filename_png, dpi=dpi, transparent=True, bbox_inches='tight' )

        if savepdf:
            plt.savefig( filename_pdf, bbox_inches='tight', dpi=dpi )
    else:
        if savepng:
            plt.savefig( file.replace('.h5','-grid.png'), dpi=dpi, transparent=True, bbox_inches='tight' )

        if savepdf:
            plt.savefig( file.replace('.h5','-grid.pdf'), bbox_inches='tight' )

    return ax,cb,hplot


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
    time, x0, dx, box, data, treecode = read_wabbit_hdf5( fname_wabbit )
    Bs = data.shape[1]
    Jflusi = (np.log2(ny/(Bs-1)))
    print("Flusi resolution: %i %i %i so desired level is Jmax=%f" % (data_ref.shape[0], data_ref.shape[2], data_ref.shape[2], Jflusi) )

    if dim==2:
        # squeeze 3D flusi field (where dim0 == 1) to true 2d data
        data_ref = data_ref[0,:,:].copy().transpose()
        box_ref = box_ref[1:2].copy()

    # convert wabbit to dense field
    data_dense, box_dense = dense_matrix( x0, dx, data, treecode, dim )
    
    if data_dense.shape[0] < data_ref.shape[0]:
        # both datasets have different size
        s = int( data_ref.shape[0] / data_dense.shape[0] )
        data_ref = data_ref[::s, ::s].copy()
        raise ValueError("ERROR! Both fields are not a the same resolutionn")

    if data_dense.shape[0] > data_ref.shape[0]:
        warn("WARNING! The reference solution is not fine enough for the comparison! UPSAMPLING!")
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
        raise ValueError("ERROR! Both fields are not a the same resolutionn")

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
        time1, x01, dx1, box1, data1, treecode1 = read_wabbit_hdf5( fname_ref )
        time2, x02, dx2, box2, data2, treecode2 = read_wabbit_hdf5( fname_dat )
    
        data1, box1 = dense_matrix( x01, dx1, data1, treecode1, dim )
        data2, box2 = dense_matrix( x02, dx2, data2, treecode2, dim )
        
        if (len(data1) != len(data2)) or (np.linalg.norm(box1-box2)>1e-15):
           raise ValueError("ERROR! Both fields are not a the same resolution")

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
    time, x0, dx, box, data, treecode = read_wabbit_hdf5( fname_in )

    # convert blocks to complete matrix
    field, box = dense_matrix(  x0, dx, data, treecode, dim=dim )

    # write data to FLUSI-type hdf file
    if fname_out:
        insect_tools.write_flusi_HDF5( fname_out, time, box, field)
    else:        
        dx = [b/(np.size(field,k)) for k,b in enumerate(box)]
        X = [np.arange(0,np.size(field,k))*dx[k] for k,b in enumerate(box)]
        return field, box, dx, X

#
def compare_two_grids( treecode1, treecode2 ):
    """ Compare two grids. The number returned is the % of blocks from treecode1
    which have also been found in treecode2 """
    import numpy as np

    common_blocks = 0

    for i in range(treecode1.shape[0]):
        # we look for this tree code in the second array
        code1 = treecode1[i,:]

        for j in range(treecode2.shape[0]):
            code2 = treecode2[j,:]
            if np.linalg.norm( code2-code1 ) < 1.0e-13:
                # found code1 in the second array
                common_blocks += 1
                break

    print( "Nblocks1=%i NBlocks2=%i common blocks=%i" % (treecode1.shape[0], treecode2.shape[0], common_blocks) )

    return common_blocks / treecode1.shape[0]


#
def overwrite_block_data_with_level(treecode, data):
    """On all blocks of the data array, replace any function values by the level of the block"""

    if len(data.shape) == 4:
        N = treecode.shape[0]
        for i in range(N):
            level = treecode_level(treecode[i,:])
            data[i,:,:,:] = float( level )

    elif len(data.shape) == 3:

        N = treecode.shape[0]
        for i in range(N):
            level = treecode_level(treecode[i,:])
            data[i,:,:] = float( level )

    return data


#
def dense_matrix(  x0, dx, data, treecode, dim=2, verbose=True, new_format=False ):

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
    jmin, jmax = get_max_min_level( treecode )
    if jmin != jmax:
        raise ValueError("ERROR! not an equidistant grid yet...")

    
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
# calculates treecode from the index of the block
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
        err("The given directory does not exist!")

    files = glob.glob(directory+'/*.h5')
    files.sort()
    for file in files:
        c = command % file
        os.system(c)

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
        err("The given directory does not exist!")

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
                err("Number of Intervals must be a power of 2!")
        else:
            err("datasize must be multiple of Bs!")
            
    # 3) check dimension of array:
    if Ndim < 2 or Ndim > 3:
        err("dimensions are wrong")
    #########################################################

    # assume periodicity:
    data = np.zeros(Nsize+1, dtype=dtype)
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

                    # lower = [ibx, iby, ibz]* (Bs - 1)
                    lower = [ibx, iby, ibz]* (Bs-1)
                    lower = np.asarray(lower, dtype=int)
                    upper = lower + Bs

                    treecode.append(blockindex2treecode([ibx, iby, ibz], 3, level))
                    bdata.append( data[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] )
    else:
        for ibx in range(Nintervals[0]):
            for iby in range(Nintervals[1]):
                x0.append([ibx, iby]*Lintervals)
                dx.append(Lintervals/(Bs-1))
                
                lower = [ibx, iby]* (Bs - 1)
                lower = np.asarray(lower, dtype=int)
                upper = lower + Bs
                
                treecode.append(blockindex2treecode([ibx, iby], 2, level))
                bdata.append( data[lower[0]:upper[0], lower[1]:upper[1]] )


    x0 = np.asarray(x0, dtype=dtype)
    dx = np.asarray(dx, dtype=dtype)
    treecode   = np.asarray(treecode, dtype=int)
    block_data = np.asarray(bdata, dtype=dtype)

    write_wabbit_hdf5(fname, time, x0, dx, box, block_data, treecode, iteration, dtype )
    return fname

#

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
            err("Number of Grid points has to be a power of 2!")
            
    # Note we have to flip  n here because Bs = [BsX, BsY]
    # The order of Bs is choosen like it is in WABBIT.
    # NB: while this definition is the one from a redundant grid,
    # it is used the same in the uniqueGrid ! The funny thing is that in the latter
    # case, we store the 1st ghost node to the H5 file - this is required for visualization.
    return n[::-1]//2**level + 1



def read_Bs_from_file(file):
    import h5py
    
    fid = h5py.File(file, 'r')
    b = fid['blocks'][:]            
    dset_id = fid.get('blocks')
    Nb, Bs, Bs = dset_id.shape            
    fid.close() 
        
    return Bs
