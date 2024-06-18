import torch
import torch.add as TorchAdd
import torch.matmul as TorchMatul

a=torch.tensor([1])
b=torch.tensor([2])
print("#########################case1#########################")
func = TorchAdd
func(a,b)

print("#########################case2#########################")
func = TorchMatul
func(a,b)
