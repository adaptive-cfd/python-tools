"""
We want to set colors for logging often so instead of including it in every file here it is
"""

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def warn( msg ):
    print( WARNING + "WARNING! " + ENDC + msg)

def err( msg ):
    print( FAIL + "CRITICAL! " + ENDC + msg)

def info( msg ):
    print( OKBLUE + "Information:  " + ENDC + msg)
