#include <torch/extension.h>
#include <vector>
#include <iostream>

using namespace torch::indexing; 

void Combinatorial(
        int n, 
        int k, 
        int num, 
        std::vector<torch::Tensor>* out, 
        torch::TensorOptions options, 
        torch::Tensor msk)
{ 
	if (n == 0)
	{
		if (k == 0)
		{ 
			torch::Tensor tmp = torch::tensor(num, options);
			tmp = tmp.unsqueeze(-1).bitwise_and(msk).ne(0).to(torch::kInt);
			out -> push_back(tmp);
		}
		return; 
	}
	if (n -1 >= k){ Combinatorial(n-1, k, num, out, options, msk); }
	if (k > 0){ Combinatorial(n -1, k -1, num | ( 1 << (n -1)), out, options, msk); }
}


torch::Tensor PathCombinatorialCPU(
        int n, 
        unsigned int max, 
        std::string device)
{
	torch::TensorOptions options = torch::TensorOptions();
	if (device == "cuda"){options = options.device(torch::kCUDA);}
	torch::Tensor msk = torch::pow(2, torch::arange(n, options));

	std::vector<torch::Tensor> nodes; 
	for (unsigned int k = 1; k < max +1; ++k)
	{
		Combinatorial(n, k, 0, &nodes, options, msk);
	}
	
	return torch::stack(nodes).to(options);
}

torch::Tensor MassFromPxPyPzE(torch::Tensor v)
{
  v = v.pow(2);  
  v = v.view({-1, 4});
  
  torch::Tensor px = v.index({Slice(), Slice(0, 1, 1)}); 
  torch::Tensor py = v.index({Slice(), Slice(1, 2, 2)}); 
  torch::Tensor pz = v.index({Slice(), Slice(2, 3, 3)}); 
  torch::Tensor e = v.index({Slice(), Slice(3, 4, 4)}); 
  
  torch::Tensor s2 = e - px - py - pz;
  return torch::sqrt(s2.abs()); 
}

torch::Tensor PathVectorCPU(
        torch::Tensor AdjMatrix, 
        torch::Tensor FourVector)
{
	std::vector<torch::Tensor> MassCombi; 
	for (unsigned int i = 0; i < AdjMatrix.sizes()[0]; ++i)
	{
		torch::Tensor x = torch::sum(FourVector.index({AdjMatrix[i] == 1}), {0});
		MassCombi.push_back(x);
	}
	return torch::stack(MassCombi);
}

torch::Tensor PathMassCPU(
        torch::Tensor AdjMatrix, 
        torch::Tensor FourVector)
{
	return MassFromPxPyPzE(PathVectorCPU(AdjMatrix, FourVector)); 
}

std::vector<torch::Tensor> IncomingEdgeVectorCPU(
        torch::Tensor AdjMatrix, 
        torch::Tensor IncomingEdges, 
        torch::Tensor Index)
{
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU); 
    const unsigned int adj = AdjMatrix.size(0);
    const unsigned int nodes = AdjMatrix.size(1);
    torch::Tensor Pmu = torch::zeros({adj*nodes, 4}, options);
    torch::Tensor Pmu_adj = torch::zeros({adj*nodes, 1}, options);
    Index = Index.to(options);
    IncomingEdges = IncomingEdges.to(options);
    AdjMatrix = AdjMatrix.to(options);

    for (unsigned int i = 0; i < nodes; i++)
    {
        for (unsigned int index = 0; index < adj; index++)
        {
            for (unsigned int j = 0; j < nodes; j++)
            {
                unsigned int ni = Index[j + i*nodes][0].item<int>();
                if ( ni != i ){continue;}
                for (unsigned int dim = 0; dim < 4; dim++)
                {
                    Pmu[index + i*adj][dim] += AdjMatrix[index][j]*IncomingEdges[j + i*nodes][dim];
                }
                Pmu_adj[index + i*adj][0] = Index[j + i*nodes][0];
            }
        }
    }

    return {Pmu, Pmu_adj};
}

std::vector<torch::Tensor> IncomingEdgeMassCPU(
        torch::Tensor AdjMatrix, 
        torch::Tensor IncomingEdges, 
        torch::Tensor Index)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kLong);
    std::vector<torch::Tensor> V = IncomingEdgeVectorCPU(AdjMatrix, IncomingEdges, Index);
    return {MassFromPxPyPzE(V[0]), V[1].to(options)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("PathCombinatorial",  &PathCombinatorialCPU, "Path Combinatorial");
    m.def("PathVector",         &PathVectorCPU, "Summation of four vectors");
    m.def("PathMass",           &PathMassCPU, "Invariant Mass"); 
    m.def("IncomingEdgeVector", &IncomingEdgeVectorCPU, "Computes the aggregated vector for different combinatorial of incoming edges");
	m.def("IncomingEdgeMass",   &IncomingEdgeMassCPU, "Computes the invariant mass of summed edge combinatorials");
}
