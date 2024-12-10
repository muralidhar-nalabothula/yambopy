# Create a two float array to a single complex float array 
import numpy as np


def float2Cmplex(arr):
    ### return a view (not copy) of complex array from float array 
    ### with last dimention == 2
    ### If the last dimension is not contiguous, a copy is created.
    if arr.shape[-1] != 2:
        printf("Last dimention of float array must be 2 for complex view")
        exit()
    # // Check if the last dimension is contiguous
    if not arr[...,:].flags['C_CONTIGUOUS']:
        return arr[...,0] + 1j*arr[...,1]
    else :
        if arr.dtype == np.float64: return arr.view(np.complex128)[...,0]
        elif arr.dtype == np.float32: return arr.view(np.complex64)[...,0]
        else : arr[...,0] + 1j*arr[...,1]


