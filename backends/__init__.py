import os
import sys

#Default keras backend
_BACKEND = 'keras'

#Check 'SLICE_BACKEND Key exist
if 'SLICE_BACKEND' in os.environ:
    _backend = os.environ['SLICE_BACKEND']

    #Set _Backend
    if _backend:
        _BACKEND = _backend

#If ackend is keras import keras backend functions
if _BACKEND == 'keras':
    sys.stderr.write('NetSlice is using keras backend.\n')
    from .keras_backend import *

#Function to return backend value
def backend():
    return _BACKEND