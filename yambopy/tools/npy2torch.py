## Contains functions that transfer data to fro to numpy 

import numpy as np
try:
    import torch as pt
    torch_exist = True
    has_gpu = True
    gpu_device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
except ImportError:
    import numpy as pt
    torch_exist = False
    has_gpu = False
    gpu_device = 'cpu'


def npy2torch(arr, gpu=True):
    ## convert a numpy tensor to pytorch tensor.
    ## if gpu = True, the output will be on gpu (if exists)
    if torch_exist:
        if has_gpu and gpu: return pt.from_numpy(arr).to(gpu_device)
        else : return pt.from_numpy(arr)
    else : return arr

def torch2npy(arr):
    ## convert a numpy tensor to pytorch tensor.
    ## if gpu = True, the output will be on gpu (if exists)
    if torch_exist: return arr.detach().cpu().numpy()
    else : return arr
