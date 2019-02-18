#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:00:31 2018

@author: engels
"""
class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

def err( msg ):
    print("")
    print( bcolors.FAIL + "ERROR! " + bcolors.ENDC + msg)
    print("")

def warn( msg ):
    print("")
    print( bcolors.WARNING + "WARNING! " + bcolors.ENDC + msg)
    print("")

print("----------------------------------------")
print("%sIRENE%s submission preflight  " %(bcolors.OKGREEN, bcolors.ENDC))
print("----------------------------------------")

import sys
import os
import wabbit_tools
import numpy as np

# fetch jobfile from call
jobfile = sys.argv[1]


if os.path.isfile( jobfile ):
    auto_resub=False

    # read jobfile
    with open(jobfile) as f:
        # loop over all lines
        for line in f:
            if "ccc_mprun" in line:
                runline = line
            elif "#MSUB -n " in line:
                cpuline = line
            elif "#MSUB -T" in line:
                wtimeline = line
            elif "INIFILE=" in line:
                iniline = line
            elif "MEMORY=" in line:
                memline = line
            elif "AUTO_RESUB=" in line:
                line = line.replace('"','').replace('AUTO_RESUB=','')
                if float(line) > 0:
                    auto_resub=True



    if "./wabbit" in runline:
        # OLD STYLE: one line call
        # now runline contains the actual run:
        #       ccc_mprun ./wabbit suzuki.ini --memory=550.0GB
        runlist = runline.split()

        progfile   = runlist[1]
        paramsfile = runlist[2]
        memory     = float( runlist[3].replace('--memory=','').replace('GB','').replace('gb','') )
    else:
        # NEW STYLE: as on turing
        progfile = ""
        paramsfile = iniline.replace('INIFILE=','').replace('"','').replace('\n','')
        memory = float( memline.replace('"','').replace('GB','').replace('MEMORY=','').replace('\n',''))


    cpulist = cpuline.split()
    ncpu = float(cpulist[2])

    wtimelist = wtimeline.split()
    wtime = float(wtimelist[2])

    core_per_node = 48
    maxmem = ncpu*3.75 #GB

    if not os.path.isfile(paramsfile):
        print('paramsfile check  : %snot found%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('paramsfile check  : %sfound%s' % (bcolors.OKGREEN,bcolors.ENDC) )


    print("program           = %s%s%s" % (bcolors.OKBLUE, progfile, bcolors.ENDC) )
    print("paramsfile        = %s%s%s" % (bcolors.OKBLUE, paramsfile, bcolors.ENDC) )
    print("memory in call    = %s%2.2f%s GB" % (bcolors.OKBLUE, memory, bcolors.ENDC) )
    print("max memory        = %s%i%s GB" % (bcolors.OKBLUE, maxmem, bcolors.ENDC) )
    print("max memory (safe) = %s%i%s GB" % (bcolors.OKBLUE, maxmem-5.0, bcolors.ENDC) )
    print("ncpu              = %s%i%s" % (bcolors.OKBLUE, ncpu, bcolors.ENDC) )
    print("wtime (jobfile)   = %s%i%s sec (%2.2f hours)" % (bcolors.OKBLUE, wtime, bcolors.ENDC, wtime/3600.0) )
    wtime_ini = wabbit_tools.get_ini_parameter(paramsfile, "Time", "walltime_max", float)
    # hours to seconds
    wtime_ini *= 3600.0
    print("wtime (inifile)   = %s%i%s sec (%2.2f hours)" % (bcolors.OKBLUE, wtime_ini, bcolors.ENDC, wtime_ini/3600.0) )

    if auto_resub:
        print('RESUBMISSION      : %sAutomatic resubmission is ACTIVE%s' % (bcolors.WARNING,bcolors.ENDC) )
    else:
        print('RESUBMISSION      : %sAutomatic resubmission is DEACTIVTÀTED!!%s' % (bcolors.WARNING,bcolors.ENDC) )


    if memory >= 0.98*maxmem:
        print('Memory check      : %sEXCEEDED%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('Memory check      : %sokay%s' % (bcolors.OKGREEN,bcolors.ENDC) )



    if abs(ncpu/core_per_node - float(round(ncpu/core_per_node))) > 0.0:
        print('Complete node(s)  : %sYou did not specify N*48 CPUS%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('Complete node(s)  : %sokay%s' % (bcolors.OKGREEN,bcolors.ENDC) )



    if wtime_ini > wtime:
        print('walltime *.SH/INI : %s Walltime in ini file greater than walltime in job file!%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('walltime *.SH/INI : %sokay%s' % (bcolors.OKGREEN,bcolors.ENDC) )





    Jmax = wabbit_tools.get_ini_parameter( paramsfile, 'Blocks', 'max_treelevel', int)
    L = wabbit_tools.get_ini_parameter( paramsfile, 'Domain', 'domain_size', float, vector=True)
    Bs = wabbit_tools.get_ini_parameter( paramsfile, 'Blocks', 'number_block_nodes', int)
    CFL = wabbit_tools.get_ini_parameter( paramsfile, 'Time', 'CFL', float)

    c0 =  wabbit_tools.get_ini_parameter( paramsfile, 'ACM-new', 'c_0', float)
    nu =  wabbit_tools.get_ini_parameter( paramsfile, 'ACM-new', 'nu', float)
    ceta =  wabbit_tools.get_ini_parameter( paramsfile, 'VPM', 'C_eta', float)
    penalized = wabbit_tools.get_ini_parameter( paramsfile, 'VPM', 'penalization', bool)
    csponge =  wabbit_tools.get_ini_parameter( paramsfile, 'Sponge', 'C_sponge', float)
    sponged =  wabbit_tools.get_ini_parameter( paramsfile, 'Sponge', 'use_sponge', bool)

    geometry =  wabbit_tools.get_ini_parameter( paramsfile, 'VPM', 'geometry', str)

    dx = L[0]*(2**-Jmax)/(Bs-1)

    keta = np.sqrt(ceta*nu)/dx


    print( "c0               = %2.2f" % (c0))
    print( "C_eta            = %2.2e" % (ceta))
    print( "K_eta            = %s%2.2f%s" % (bcolors.OKGREEN,keta,bcolors.ENDC))
    print( "C_sponge         = %2.2e" % (csponge))

    if geometry == "Insect":
        t =  wabbit_tools.get_ini_parameter( paramsfile, 'Insects', 'WingThickness', float)
        print( "wing thickness   = %2.2f (%2.2f dx)" % (t, t/dx))

    print('Launching wabbit_tools ini file check now:')
    wabbit_tools.check_parameters_for_stupid_errors( paramsfile )



else:
    err("Jobfile %s not found" % (jobfile))
    raise ValueError( )

