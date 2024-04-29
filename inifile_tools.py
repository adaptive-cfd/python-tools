"""
This script contains functions that deal with modfying the ini files
"""
import numpy as np
import bcolors


#
def check_parameters_for_stupid_errors( file ):
    """
    For a given WABBIT parameter file, check for the most common stupid errors
    the user can commit: Jmax<Jmin, negative time steps, etc.
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
        bcolors.err("For 3D ACM, you MUST set number_equations=4 (ux,uy,uz,p)")
        
    if physics_type == 'ACM-new' and dim == 2 and Neqn != 3:
        bcolors.err("For 2D ACM, you MUST set number_equations=3 (ux,uy,p)")
   
    if len(bs) > 1:
        bs = bs[0]

    if bs % 2 != 0:
        bcolors.warn('The block size is bs=%i which is an ODD number.' % (bs) )

    if bs < 3:
        bcolors.warn('The block size is bs=%i is very small or even negative.' % (bs) )
        
          
    if (wavelet == "CDF22") and g<3:
        bcolors.warn("Not enough ghost nodes for wavelet %s g=%i < 3" % (wavelet, g) )
    if (wavelet == "CDF42") and g<5:
        bcolors.warn("Not enough ghost nodes for wavelet %s g=%i < 5" % (wavelet, g) )        
    if (wavelet == "CDF44" or wavelet == "CDF62") and g<7:
        bcolors.warn("Not enough ghost nodes for wavelet %s g=%i < 7" % (wavelet, g) )
    if (wavelet == "CDF40") and g<4:
        bcolors.warn("Not enough ghost nodes for wavelet %s g=%i < 4" % (wavelet, g) )
        
   
    if time_step_method == "RungeKuttaChebychev":
        if CFL_eta < 999:
            bcolors.warn('are you sure you did not forget to adjustl CFL_eta for the RKC scheme???')
        if CFL_nu < 999:
            bcolors.warn('are you sure you did not forget to adjustl CFL_nu for the RKC scheme???')
        if CFL != 0.75:
            bcolors.warn('are you sure you did not forget to adjustl CFL for the RKC scheme??? often we used 0.75.')    
            
    if time_step_method == "RungeKuttaGeneric":
        if CFL_eta > 1.0:
            bcolors.warn('are you sure you did not forget to adjustl CFL_eta for the RK scheme? it may be unstable.')
        if CFL_nu > 0.99*2.79/(float(dim)*np.pi**2):
            bcolors.warn('are you sure you did not forget to adjustl CFL_nu for the RK scheme? it may be unstable.')
        if CFL >  1.0:
            bcolors.warn('are you sure you did not forget to adjustl CFL for the RK scheme? it may be unstable.')    
            
    # if somebody modifies the standard parameter file, users have to update their
    # ini files they use. this is often forgoten and obnoxious. Hence, if we find
    # value sthat no longer exist, warn the user.
    if exists_ini_parameter( file, "Blocks", "number_data_fields" ) :
        bcolors.warn('Found deprecated parameter: [Blocks]::number_data_fields')

    if exists_ini_parameter( file, "Physics", "initial_cond" ) :
        bcolors.warn('Found deprecated parameter: [Physics]::initial_cond')

    if exists_ini_parameter( file, "Dimensionality", "dim" ) :
        bcolors.warn('Found deprecated parameter: [Dimensionality]::dim')

    if exists_ini_parameter( file, "DomainSize", "Lx" ) :
        bcolors.warn('Found deprecated parameter: [DomainSize]::Lx')

    if exists_ini_parameter( file, "Time", "time_step_calc" ) :
        bcolors.warn('Found deprecated parameter: [Time]::time_step_calc')
        
    if exists_ini_parameter( file, "ACM", "forcing" ):
        bcolors.warn('Found deprecated parameter: [ACM]::forcing')
        
    if exists_ini_parameter( file, "ACM", "forcing_type" ):
        bcolors.warn('Found deprecated parameter: [ACM]::forcing_type')
        
    if exists_ini_parameter( file, "ACM", "p_mean_zero" ):
        bcolors.warn('Found deprecated parameter: [ACM]::p_mean_zero')
        
    if exists_ini_parameter( file, "ACM", "compute_laplacian" ):
        bcolors.warn('Found deprecated parameter: [ACM]::compute_laplacian')
        
    if exists_ini_parameter( file, "ACM", "compute_nonlinearity" ):
        bcolors.warn('Found deprecated parameter: [ACM]::compute_nonlinearity')
    
    if exists_ini_parameter( file, "Blocks", "adapt_mesh" ):
        bcolors.warn('Found deprecated parameter: [Blocks]::adapt_mesh ===> adapt_tree')
   
    HIT = get_ini_parameter( file, 'ACM-new', 'use_HIT_linear_forcing', bool, default=False)
    if HIT:
        print(type(HIT))
        print(HIT)
        bcolors.warn('You use HIT linear forcing, which is HIGHLY EXPERIMENTAL')

    jmax = get_ini_parameter( file, 'Blocks', 'max_treelevel', int)

    if jmax > 18:
        bcolors.warn('WABBIT can compute at most 18 refinement levels, you set more!')

    if sponged:
        # default value is TRUE so if not found, all is well
        mask_time_dependent = get_ini_parameter( file, 'VPM', 'mask_time_dependent_part', int, default=1)

        if mask_time_dependent != 1:
            bcolors.warn("""you use sponge, but mask_time_dependent_part=0! The sponge
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
                        bcolors.warn('It appears the line #%i does not contain the semicolon' % (linenumber) )

    restart = get_ini_parameter( file, 'Physics', 'read_from_files', int)
    print("read_from_files=%i" %(restart))

    if restart == 1:
        bcolors.info("This simulation is being resumed from file")

        infiles = get_ini_parameter( file, 'Physics', 'input_files', str)
        infiles = infiles.split()
        for file in infiles:
            print(file)
            if not os.path.isfile(file):
                raise ValueError("CRUTIAL: read_from_files=1 but infiles NOT found!.")
    else:
        bcolors.info("This simulation is being started from initial condition (and not from file)")

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
        raise ValueError(bcolors.FAIL + "ERROR! backup resuming is available only for ACM" + bcolors.ENDC)


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
