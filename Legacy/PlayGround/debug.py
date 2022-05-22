import torch
import torch.nn.functional as F

input_List = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype = torch.float)
output_List = torch.tensor([[1], [0], [1], [0]], dtype = torch.long)


print(input_List.size(), output_List.size())
output_List = output_List.t().contiguous().squeeze()
print(output_List)

l = F.cross_entropy(input_List, output_List)
print(l)


input_List = torch.tensor([[1], [0], [0]], dtype = torch.float)
output_List = torch.tensor([[2], [0], [1]], dtype = torch.long)



print(input_List.size(), output_List.size())

l = F.mse_loss(input_List, output_List)
print(l)
