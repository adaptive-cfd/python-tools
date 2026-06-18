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
            elif "MEMPERCORE=" in line:
                mempercoreline = line
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

    core_per_node = 128  # for Rome, byebye Skylake with 48 cores
    mempercore_max = 1.781  # GB
    mempercore_lim = 1.7  # GB
    if 'mempercoreline' in locals():
        mempercorelist = mempercoreline.split('=')
        mempercore = float( mempercorelist[1].replace('"','').replace('GB','').replace('\n','') )
    elif 'memline' in locals():
        memlist = memline.split('=')
        totalmem = float( memlist[1].replace('"','').replace('GB','').replace('\n','') )
        mempercore = totalmem / ncpu
    else:
        bcolors.err(f"Please specify either MEMORY or MEMPERCORE in your job submission script for launching so that we can check whether your memory request is reasonable and does not exceed the maximum per core (which is {mempercore_max:.3f} GB, but it is recommended to set below {mempercore_lim:.3f} GB).")
        raise ValueError("Missing MEMORY or MEMPERCORE specification in job submission script.")
    if mempercore > mempercore_max:
        print('Memory per core  : %sYou requested %.3f GB per core, which exceeds the usage limit of %.3f GB per core!%s' % (bcolors.FAIL, mempercore, mempercore_max, bcolors.ENDC) )
        raise ValueError("Requested memory per core exceeds allowed limit.")
    elif mempercore > mempercore_lim:
        print('Memory per core  : %sYou requested %.3f GB per core, which exceeds the recommended limit of %.3f GB per core!%s' % (bcolors.WARNING, mempercore, mempercore_lim, bcolors.ENDC) )

    if not os.path.isfile(paramsfile):
        print('paramsfile check  : %snot found%s' % (bcolors.FAIL,bcolors.ENDC) )
    else:
        print('paramsfile check  : %sfound%s' % (bcolors.OKGREEN,bcolors.ENDC) )


    print("program           = %s%s%s" % (bcolors.OKBLUE, progfile, bcolors.ENDC) )
    print("paramsfile        = %s%s%s" % (bcolors.OKBLUE, paramsfile, bcolors.ENDC) )
    print("ncpu              = %s%i%s" % (bcolors.OKBLUE, ncpu, bcolors.ENDC) )
    print('Memory per core   = %s%.3f GB per core%s' % (bcolors.OKBLUE, mempercore, bcolors.ENDC) )
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
        print('Complete node(s)  : %sYou did not specify N*%i CPUS%s' % (bcolors.FAIL, core_per_node, bcolors.ENDC) )
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
    raise ValueError("Jobfile not found.")

