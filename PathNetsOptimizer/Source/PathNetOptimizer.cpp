#include <torch/extension.h>
#include <iostream>
#include <vector>

void Combinatorial(int n, int k, int num, std::vector<torch::Tensor>* out, torch::Tensor msk, int node)
{ 
  if (n == 0){
    if (k == 0){ (*out).push_back(torch::tensor(num).unsqueeze(-1).bitwise_and(msk).ne(0).to(torch::kFloat).view({node, 1})); }
    return; 
  }
  if (n -1 >= k){ Combinatorial(n-1, k, num, out, msk, node); }
  if (k > 0){ Combinatorial(n -1, k -1, num | ( 1 << (n -1)), out, msk, node); }
}


std::vector<torch::Tensor> PathCombination(torch::Tensor AdjMatrix, int Nodes, int choose)
{
  auto options = torch::TensorOptions().device(AdjMatrix.device().type());
  torch::Tensor msk = torch::pow(2, torch::arange(Nodes, options)); 
  AdjMatrix.to(torch::kFloat);  

  // Create the binary combinatorial 
  std::vector<torch::Tensor> Binary;
  for (int i = 1; i < choose; i++){Combinatorial(Nodes, i+1, 0, &Binary, msk, Nodes);}

  std::vector<torch::Tensor> Output; 
  std::vector<torch::Tensor> Output_Matrix; 
  // Convert the Binary to a tensor 
  for (int i = 0; i < Binary.size(); i++)
  {
    torch::Tensor proj = AdjMatrix.matmul(Binary[i]);
    
    proj.index_put_({proj == Binary[i].sum(0)}, 0); 
    proj.index_put_({proj > 0}, 1); 
    Output.push_back(proj);

    torch::Tensor proj_T = proj.matmul(proj.t()); 
    Output_Matrix.push_back(proj_T);
  }
  
  torch::Tensor out = torch::stack(Output); 
  torch::Tensor out2 = torch::stack(Output_Matrix); 
  
  return {out, out2};
}



torch::Tensor Combination(int Nodes, int choose, std::string device)
{
  auto options = torch::TensorOptions();
  if (device == "cuda"){options.device(torch::kCUDA);}
  else{ options.device(torch::kCPU); }

  torch::Tensor msk = torch::pow(2, torch::arange(Nodes, options)); 

  // Create the binary combinatorial 
  std::vector<torch::Tensor> Binary;
  for (int i = 1; i < choose; i++){Combinatorial(Nodes, i+1, 0, &Binary, msk, Nodes);}

  if (device == "cuda") {return torch::stack(Binary).to(torch::kCUDA);}
  else{return torch::stack(Binary);}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("PathCombination", &PathCombination, "Path Combinatorial");
  m.def("PathCombination", &Combination, "Path Combinatorial");
}
