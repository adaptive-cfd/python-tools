#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:00:31 2018

@author: engels
"""
import bcolors

print("----------------------------------------")
print("%sZAY%s submission preflight  " %(bcolors.OKGREEN, bcolors.ENDC))
print("----------------------------------------")


import sys, os
import inifile_tools
import numpy as np

sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))


# fetch jobfile from call
jobfile = sys.argv[1]


if os.path.isfile( jobfile ):
    auto_resub=False

    # read jobfile
    with open(jobfile) as f:
        # loop over all lines
        for line in f:
            if "#SBATCH --ntasks=" in line:
                cpuline = line
            elif "#SBATCH --time=" in line:
                wtimeline = line
            elif "INIFILE=" in line:
                iniline = line
            elif "AUTO_RESUB=" in line:
                line = line.replace('"','').replace('AUTO_RESUB=','')
                if float(line) > 0:
                    auto_resub=True


        progfile = ""
        paramsfile = iniline.replace('INIFILE=','').replace('"','').replace('\n','')


    cpuline = cpuline.replace("#SBATCH --ntasks=","")
    cpulist = cpuline.split()
    ncpu = float(cpulist[0])

    wtimelist = wtimeline.replace("#SBATCH --time=","").split()
    wtimelist = wtimelist[0].split(":")
    wtime = float(wtimelist[0])*3600 + float(wtimelist[1])*60 + float(wtimelist[2])

    core_per_node = 40
    maxmem = ncpu*4.8 #GB

    if not os.path.isfile(paramsfile):
        print('paramsfile check  : %snot found%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('paramsfile check  : %sfound%s' % (bcolors.OKGREEN,bcolors.ENDC) )


    print("program           = %s%s%s" % (bcolors.OKBLUE, progfile, bcolors.ENDC) )
    print("paramsfile        = %s%s%s" % (bcolors.OKBLUE, paramsfile, bcolors.ENDC) )
    print("ncpu              = %s%i%s" % (bcolors.OKBLUE, ncpu, bcolors.ENDC) )
    print("wtime (jobfile)   = %s%i%s sec (%2.2f hours)" % (bcolors.OKBLUE, wtime, bcolors.ENDC, wtime/3600.0) )
    wtime_ini = inifile_tools.get_ini_parameter(paramsfile, "Time", "walltime_max", float)
    # hours to seconds
    wtime_ini *= 3600.0
    print("wtime (inifile)   = %s%i%s sec (%2.2f hours)" % (bcolors.OKBLUE, wtime_ini, bcolors.ENDC, wtime_ini/3600.0) )

    if auto_resub:
        print('RESUBMISSION      : %sAutomatic resubmission is ACTIVE%s' % (bcolors.WARNING,bcolors.ENDC) )
    else:
        print('RESUBMISSION      : %sAutomatic resubmission is DEACTIVTÀTED!!%s' % (bcolors.WARNING,bcolors.ENDC) )


    if abs(ncpu/core_per_node - float(round(ncpu/core_per_node))) > 0.0:
        print('Complete node(s)  : %sYou did not specify N*48 CPUS%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('Complete node(s)  : %sokay%s' % (bcolors.OKGREEN,bcolors.ENDC) )

    if wtime_ini > wtime:
        print('walltime *.SH/INI : %s Walltime in ini file greater than walltime in job file!%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('walltime *.SH/INI : %sokay%s' % (bcolors.OKGREEN,bcolors.ENDC) )

    print("----------------------------------------")

    eps  = inifile_tools.get_ini_parameter( paramsfile, 'Blocks', 'eps',float )

    Jmax = inifile_tools.get_ini_parameter( paramsfile, 'Blocks', 'max_treelevel', int)
    L = inifile_tools.get_ini_parameter( paramsfile, 'Domain', 'domain_size', float, vector=True)
    Bs = inifile_tools.get_ini_parameter( paramsfile, 'Blocks', 'number_block_nodes', int, vector=True)
    CFL = inifile_tools.get_ini_parameter( paramsfile, 'Time', 'CFL', float)

    c0 =  inifile_tools.get_ini_parameter( paramsfile, 'ACM-new', 'c_0', float)
    nu =  inifile_tools.get_ini_parameter( paramsfile, 'ACM-new', 'nu', float)
    ceta =  inifile_tools.get_ini_parameter( paramsfile, 'VPM', 'C_eta', float, default=0.0)
    penalized = inifile_tools.get_ini_parameter( paramsfile, 'VPM', 'penalization', bool)
    csponge =  inifile_tools.get_ini_parameter( paramsfile, 'Sponge', 'C_sponge', float, default=0.0)
    sponged =  inifile_tools.get_ini_parameter( paramsfile, 'Sponge', 'use_sponge', bool)

    geometry =  inifile_tools.get_ini_parameter( paramsfile, 'VPM', 'geometry', str, default='none')

    if len(Bs)==1:
        dx = L[0]*(2**-Jmax)/(Bs[0]-1)
    else:
        dx = min(L*(2**-Jmax)/(Bs-1))

    keta = np.sqrt(ceta*nu)/dx

    print( "dx               = %2.2e" % (dx))
    print( "Jmax             = %i" % (Jmax))
    print( "eps              = %2.2e" % (eps))
    print( "c0               = %2.2f" % (c0))
    print( "C_eta            = %2.2e" % (ceta))
    print( "K_eta            = %s%2.2f%s" % (bcolors.OKGREEN,keta,bcolors.ENDC))
    print( "C_sponge         = %2.2e" % (csponge))

    if geometry == "Insect":
        t =  inifile_tools.get_ini_parameter( paramsfile, 'Insects', 'WingThickness', float, default=0.0)
        print( "wing thickness   = %2.2f (%2.2f dx)" % (t, t/dx))

    print("----------------------------------------")
    print('Launching wabbit_tools ini file check now:')
    print("----------------------------------------")
    inifile_tools.check_parameters_for_stupid_errors( paramsfile )



else:
    bcolors.err("Jobfile %s not found" % (jobfile))
    raise ValueError( )

