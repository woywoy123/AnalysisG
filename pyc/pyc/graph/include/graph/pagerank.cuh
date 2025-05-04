#ifndef CUPAGERANK_H
#define CUPAGERANK_H

#include <map>
#include <string>
#include <torch/torch.h>

namespace graph_ {
    std::map<std::string, torch::Tensor> page_rank(
            torch::Tensor* edge_index, torch::Tensor* edge_scores, 
            double alpha, double threshold, double norm_low, long timeout, int num_cls
    );

    std::map<std::string, torch::Tensor> page_rank_reconstruction(
            torch::Tensor* edge_index, torch::Tensor* edge_scores, torch::Tensor* pmc,  
            double alpha, double threshold, double norm_low, long timeout, int num_cls
    );


}

#endif
