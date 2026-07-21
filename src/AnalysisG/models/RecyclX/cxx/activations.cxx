#include <pyc/pyc.h>
#include <utils.h>

NetOps::NetOps(network v0){this -> ntwk = v0;}
NetOps::NetOps(network v0, long l1, long l2){
    this -> ntwk = v0; 
    this -> _l1 = l1;
    this -> _l2 = (l2 < 0) ? l1 : l2; 
}

std::string NetOps::Name(int x1, int x1p, int depth){
    std::string op = ""; 
    op += "][src|" + std::to_string(x1)    + "|+>|";  
    op +=   "dst|" + std::to_string(x1p)   + "]";
    op += "[depth|" + std::to_string(depth)+ "]";  
    switch(this -> ntwk){
        case network::linear:    return "[linear"    + op;
        case network::layernorm: return "[layernorm" + op; 
        case network::dropout:   return "[dropout"   + op; 
        case network::relu:      return "[relu"      + op; 
        case network::silu:      return "[silu"      + op; 
        case network::sigmoid:   return "[sigmoid"   + op; 
        case network::prelu:     return "[prelu"     + op; 
        case network::leakyrelu: return "[leakyrelu" + op; 
        case network::tanh:      return "[tanh"      + op; 
        case network::invalid:   return "[invalid"   + op; 
        default: break;
    }
    return "-1"; 
}

void NetOps::Apply(torch::nn::Sequential* nn, NetOps* opx){
    std::string name = this -> Name(opx -> _l1, opx -> _l2, opx -> depth); 
    switch(this -> ntwk){
        case network::linear:    (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n1, opx)); return; 
        case network::layernorm: (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n2, opx)); return;  
        case network::dropout:   (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n3, opx)); return; 
        case network::relu:      (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n4, opx)); return; 
        case network::silu:      (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n5, opx)); return; 
        case network::sigmoid:   (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n6, opx)); return; 
        case network::prelu:     (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n7, opx)); return; 
        case network::leakyrelu: (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n8, opx)); return;  
        case network::tanh:      (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n9, opx)); return; 
        case network::invalid:   abort(); 
        default: break;
    }
}


torch::nn::Linear utils::make_Fx(torch::nn::Linear* tn, NetOps* opx){
    return torch::nn::Linear(torch::nn::LinearOptions(opx -> _l1, opx -> _l2).bias(true)); 
}

torch::nn::LayerNorm utils::make_Fx(torch::nn::LayerNorm* tn, NetOps* opx){
    return torch::nn::LayerNorm(torch::nn::LayerNormOptions({opx -> _l1})); 
}

torch::nn::Dropout utils::make_Fx(torch::nn::Dropout* tn, NetOps* opx){
    double k = double(opx -> _l1) / 100.0; 
    return torch::nn::Dropout(torch::nn::DropoutOptions({k})); 
}

torch::nn::ReLU utils::make_Fx(torch::nn::ReLU* tn, NetOps* opx){
    return torch::nn::ReLU();
}

torch::nn::LeakyReLU utils::make_Fx(torch::nn::LeakyReLU* tn, NetOps* opx){
    return torch::nn::LeakyReLU();
}

torch::nn::Tanh utils::make_Fx(torch::nn::Tanh* tn, NetOps* opx){
    return torch::nn::Tanh();
}

torch::nn::SiLU utils::make_Fx(torch::nn::SiLU* tn, NetOps* opx){
    return torch::nn::SiLU();
}

torch::nn::Sigmoid utils::make_Fx(torch::nn::Sigmoid* tn, NetOps* opx){
    return torch::nn::Sigmoid();
}

torch::nn::PReLU utils::make_Fx(torch::nn::PReLU* tn, NetOps* opx){
    return torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(int(opx -> _l1))); 
}

torch::nn::Sequential* utils::make_Network(std::string title, std::vector<NetOps> prm){
    prm.push_back(NetOps(network::invalid,  prm[prm.size()-1]._l1)); 
    std::vector<std::pair<std::string, torch::nn::Module>> data = {}; 
    torch::nn::Sequential* nn = new torch::nn::Sequential(); 
    for (size_t x(0); x < prm.size()-1; ++x){prm[x].Apply(nn, &prm[x]);}
    return nn;
}

