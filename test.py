import torch
import batch_linalg_cuda as bl
import numpy as np

bl.demo()


bA = torch.cuda.FloatTensor([[[1., 0.], [0., 1.]], [[2., 0.], [0., 3.]]])
bb = torch.cuda.FloatTensor([[3., 4.], [4., 6.]])
#bx = torch.cuda.DoubleTensor([[0., 0.], [0., 0.]])

bx = bl.batchSolveSingle(bA, bb)

print(bA)
print(bA.size())
print(bA.dtype)
print(bb)
print(bx)


q = bx.cpu().numpy()


#Just checking that it's not some pointer magic that makes it work, and the results can be used later.
print(q)

###############

n = 10
b = 100000

bb = torch.ones(b, n).cuda()
bA = torch.randn(b,n,n).cuda()*1e-3 +  torch.eye(n).cuda()

bx = bl.batchSolveSingle(bA, bb)

print(bA.size())
print(bx.size())

by = torch.bmm(bA.view(-1, n, n), bx.view(-1, n, 1)).view(b, n)
print(torch.sum(by - bb)/(b*n))



