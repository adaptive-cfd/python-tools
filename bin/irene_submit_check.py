#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:00:31 2018

@author: engels
"""
import bcolors

print("----------------------------------------")
print("%sIRENE%s submission preflight  " %(bcolors.OKGREEN, bcolors.ENDC))
print("----------------------------------------")

import sys, os
import inifile_tools

sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))


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
    else:
        # NEW STYLE: as on turing
        progfile = ""
        paramsfile = iniline.replace('INIFILE=','').replace('"','').replace('\n','')


    cpulist = cpuline.split()
    ncpu = float(cpulist[2])

    wtimelist = wtimeline.split()
    wtime = float(wtimelist[2])

    core_per_node = 48
    

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
    inifile_tools.check_parameters_for_stupid_errors( paramsfile )



else:
    bcolors.err("Jobfile %s not found" % (jobfile))
    raise ValueError( )

