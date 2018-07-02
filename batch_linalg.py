import math
from torch import nn
from torch.autograd import Function
import torch

import batch_linalg_cuda as blc

#torch.manual_seed(42)

#No inheritance from "Function" for now, because I have no gradients.

def batchSolve(bA, bb):
    if (not bA.is_cuda()) or (not bb.is_cuda()):
        raise ValueError('Only applicable to cuda tensors.')

    if (bA.dtype == torch.float32) and (bb.dtype == torch.float32):
        return blc.batchSolveSingle(bA, bb)
    elif (bA.dype == torch.float64) and (bb.dtype == torch.float64):
        return blc.batchSolveDouble(bA, bb)
    else:
        raise ValueError('Both tensors must have the same dtype')


