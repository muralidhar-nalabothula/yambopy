#
# License-Identifier: GPL
#
# Copyright (C) 2024 The Yambo Team
#
# Authors: HPC, AMS, FP, RR
#
# This file is part of the yambopy project
#
import numpy as np

def CmplxType(var):
    """ Distinguish between double and float for storing residuals and eigenvector with the same precision as in the Yambo database
    """
    if var.dtype=='float32':   return np.complex64
    elif var.dtype=='float64': return np.complex128
    else: raise TypeError('\n[ERROR] Variable type not recognized. It should be either float (float32) or double (float64).\n')


def float2Cmplex(arr):
    ### return a view (not copy) of complex array from float array 
    ### with last dimention == 2
    ### If the last dimension is not contiguous, a copy is created.
    assert arr.shape[-1] == 2, "Last dimention of float array must be 2 for complex view"
    # // Check if the last dimension is contiguous
    if not arr[...,:].flags['C_CONTIGUOUS']:
        return arr[...,0] + 1j*arr[...,1]
    else :
        if arr.dtype == np.float64: return arr.view(np.complex128)[...,0]
        elif arr.dtype == np.float32: return arr.view(np.complex64)[...,0]
        else : arr[...,0] + 1j*arr[...,1]
