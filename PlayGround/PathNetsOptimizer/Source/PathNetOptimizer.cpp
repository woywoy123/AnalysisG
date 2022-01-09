#include <torch/extension.h>
#include <iostream>
#include <vector>

void Combinatorial(int n, int k, int num, std::vector<int>* out)
{ 
  if (n == 0){
    if (k == 0){ (*out).push_back(num); }
    return; 
  }
  if (n -1 >= k){ Combinatorial(n-1, k, num, out); }
  if (k > 0){ Combinatorial(n -1, k -1, num | ( 1 << (n -1)), out); }
}

std::vector<torch::Tensor> PathCombination(torch::Tensor AdjMatrix, int Nodes)
{
  auto options = torch::TensorOptions().device(AdjMatrix.device().type());
  torch::Tensor msk = torch::pow(2, torch::arange(Nodes, options)); 
  AdjMatrix.to(torch::kFloat);  

  // Create the binary combinatorial 
  std::vector<int> Binary;
  for (int i = 1; i < Nodes; i++){Combinatorial(Nodes, i+1, 0, &Binary);}

  std::vector<torch::Tensor> Output; 
  std::vector<torch::Tensor> Output_Matrix; 
  // Convert the Binary to a tensor 
  for (int i = 0; i < Binary.size(); i++)
  {
    torch::Tensor combi = torch::tensor(Binary[i], options).unsqueeze(-1).bitwise_and(msk).ne(0).to(torch::kFloat).view({Nodes, 1}); 
    torch::Tensor proj = AdjMatrix.matmul(combi);
    
    proj.index_put_({proj == combi.sum(0)}, 0); 
    proj.index_put_({proj > 0}, 1); 
    Output.push_back(proj);

    torch::Tensor proj_T = proj.matmul(proj.t()); 
    Output_Matrix.push_back(proj_T);
  }
  
  torch::Tensor out = torch::stack(Output); 
  torch::Tensor out2 = torch::stack(Output_Matrix); 
  
  return {out, out2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("PathCombination", &PathCombination, "Path Combinatorial");
}
