## Contains functions that transfer data to fro to numpy 

import numpy as np
import torch as pt

## check if any gpu is available 
## (i) Cuda or HIP
gpu_device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
## (ii) incase of Apple silicon gpu
if pt.backends.mps.is_available() and pt.backends.mps.is_built():
    gpu_device = pt.device("mps")

def to_pytorch_tensor(arr, gpu=True):
    ## convert a numpy_array/list/pytorch_tensor to pytorch tensor.
    ## if gpu = True, the output will be on gpu (if exists)
    if type(arr) == list:
        if gpu : return pt.Tensor(arr).to(gpu_device)
        else : return pt.Tensor(arr)
    elif type(arr) == pt.Tensor:
        if gpu and arr.is_cpu: return arr.to(gpu_device)
        elif not gpu and not arr.is_cpu: return arr.detach().cpu().numpy()
        else : return arr
    else :
        assert type(arr) == np.ndarray, "Can only pass numpy.ndarray or torch.Tensor or valid list"
        if gpu: return pt.from_numpy(arr).to(gpu_device)
        else : return pt.from_numpy(arr)

def to_npy_array(arr):
    ## convert a pytorch_tensor/list to numpy tensor.
    ## if gpu = True, the output will be on gpu (if exists)
    if type(arr) == list:
        return np.array(arr)
    elif type(arr) == pt.Tensor:
        return arr.detach().cpu().numpy()
    else :
        assert type(arr) == np.ndarray, "Can only pass numpy.ndarray or torch.Tensor or valid list"
        return arr
