#include <torch/extension.h>
#include <iostream>


void Combinatorial(int n, int k, int num, std::vector<int>* out)
{ 
  if (n == 0){
    if (k == 0){ (*out).push_back(num); }
    return; 
  }
  if (n -1 >= k){ Combinatorial(n-1, k, num, out); }
  if (k > 0){ Combinatorial(n -1, k -1, num | ( 1 << (n -1)), out); }
}

torch::Tensor PathCombination(torch::Tensor AdjMatrix, int Nodes)
{
  // Create the binary combinatorial 
  std::vector<int> Binary;
  for (int i = 1; i < Nodes; i++){Combinatorial(Nodes, i+1, 0, &Binary);}
  
  auto options = torch::TensorOptions().device(AdjMatrix.device().type());
  std::vector<torch::Tensor> Output; 
  // Convert the Binary to a tensor 
  for (int i = 0; i < Binary.size(); i++)
  {

    torch::Tensor msk = torch::pow(2, torch::arange(Nodes, options)); 
    torch::Tensor proj = torch::tensor(Binary[i], options).unsqueeze(-1).bitwise_and(msk).ne(0).to(torch::kInt); 
    Output.push_back(proj);
  }
  
  torch::Tensor out = torch::stack(Output); 
  return out;
}










PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("PathCombination", &PathCombination, "Path Combinatorial");
}
