#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:06:06 2021

@author: engels
"""

import glob
import os
import time
import subprocess
import argparse

# waiting time between checks (seconds):
TIME_WAIT_CHECK = 30
# if file is older than (seconds), warn:
TIME_WARNING = 30 * 60
# warn every seconds, until the kill time is reached
TIME_WARNING_INTERVAL = 10 * 60
# kill run if older than (seconds):
TIME_KILL = 60 * 60

def age_of_youngest_file( directory='./' ):
    """
    Returns the age of the youngest file in a directory in seconds.

    Parameters
    ----------
    directory : TYPE, optional
        DESCRIPTION. The default is './'.

    Returns
    -------
    float, age in seconds

    """

    # we do not simply use all files: only actual output of code.
    files = glob.glob( directory+'/*.h5', recursive=False)
    files.extend(glob.glob( directory+'/*.t', recursive=False))
    files.extend(glob.glob( directory+'/*.dat', recursive=False))
    files.extend(glob.glob( directory+'/*.out', recursive=False))
    files.extend(glob.glob( directory+'/*.err', recursive=False))
    
    if not files:
        # empty list
        # return no age - wait longer.
        return 0.0
    
    for file in files:
        # check if this is a symlink, if so remove it
        # used because dead symlinks cause errors
        if os.path.islink(file):
            files.remove(file)

    
    youngest_file = max(files, key=os.path.getctime)

    return(time.time() - os.path.getctime(youngest_file))


def SendMail(subject: str, body: str, recipient):
    body_str_encoded_to_byte = body.encode()
    return_stat = subprocess.run([f"mail", f"-s {subject}", recipient], input=body_str_encoded_to_byte)
    # print(return_stat) 
    
def GetJobStatus(ID: int):
    s = subprocess.check_output( "myjobstatus.sh %i" % (ID), shell=True )
    s = str(s).rstrip().lstrip().replace("b'", "").replace("\\n'","")
    return s

age_of_youngest_file( directory='./' )

# after killing, the job is not automatically resubmited, we need to call automatic_resubmission.sh to do that


#---------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobfile", help="""The name of the jobfile. Should be job.sh for technical reasons: The resubmission
                    is done using automatic_resubmission.sh.""", default='job.sh')
parser.add_argument("-d", "--dir", help="""Data directory of the job, used to check for activity.""")
parser.add_argument("-i", "--ID", help="""The ID of the job. We use this for the cancel command: mycancel $ID. This, of course,
                    is used only if the job fails""", type=int)
parser.add_argument("-m", "--mail", help="Mail adress to send updates to", default="thomas.engels@ens.fr")
parser.add_argument("--cluster", help="Name of cluster", default="Irene")
args = parser.parse_args()
#---------------------------------------------------------------------------------------------------------------------------------------

ID = args.ID
jobfile = args.jobfile
jobdir = args.dir
print('%i %s %s' % (ID, jobfile, jobdir))

TIME_LASTMAIL = 0.0
a = 0.0

stat = GetJobStatus(ID)

kill_job = False

while stat != "ERR":
    # to reduce disk access, wait a bit
    time.sleep(TIME_WAIT_CHECK)
    
    # check if the job is still running 
    stat = GetJobStatus(ID)
    
    # watchdog does not do anything unless the job is running:
    if stat == "RUN":
        a = age_of_youngest_file(jobdir)
        
        if a > TIME_KILL:
            print("KILLING")
            SendMail(subject="%s:AUTOKILL:%s" % (args.cluster.upper(), jobdir), body="KILLED\nKilled the job %i %s\nAge=%f min" %(ID, jobdir, a/60.0), recipient=args.mail)
            # time to die.
            kill_job = True
            # exit while loop (even before checking for a warning.)
            break
        
        if a > TIME_WARNING and time.time()-TIME_LASTMAIL>TIME_WARNING_INTERVAL:
            print("WARNING")
            SendMail(subject="%s:WARNING:%s" % (args.cluster.upper(), jobdir), body="WARNING\nWe are about to kill the job %i %s\nAge=%f min" %(ID, jobdir, a/60.0), recipient=args.mail)
            TIME_LASTMAIL = time.time()
            

    fid = open('watchdog.log', 'w')
    fid.write("Watchdog ID=%i status=%s age=%2.3f minutes  dir=%s \n" % (ID, stat, a/60.0, jobdir))
    fid.close()  
        

if kill_job:
    subprocess.run("ccc_mdel %i" % (ID))

