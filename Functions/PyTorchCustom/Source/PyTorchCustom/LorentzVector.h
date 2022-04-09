#include <torch/extension.h>
#include <iostream>

torch::Tensor ToPxPyPzE(float pt, float eta, float phi, float e, std::string device = ""); 


