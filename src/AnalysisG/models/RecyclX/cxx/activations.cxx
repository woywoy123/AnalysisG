#include <pyc/pyc.h>
#include <utils.h>

NetOps::NetOps(network v0){this -> nt = v0;}
NetOps::NetOps(network v0, long v1){
    this -> nt = v0; 
    this -> _l1 = v1;
}

std::string NetOps::Name(int nxt){
    std::string op = "[" + std::to_string(this -> _l1) + "->" + std::to_string(nxt) + "]";  
    switch(this -> nt){
        case network::linear:    return "linear"    + op;
        case network::layernorm: return "layernorm" + op; 
        case network::dropout:   return "dropout"   + op; 
        case network::relu:      return "relu"      + op; 
        case network::silu:      return "silu"      + op; 
        case network::sigmoid:   return "sigmoid"   + op; 
        case network::prelu:     return "prelu"     + op; 
        case network::leakyrelu: return "leakyrelu" + op; 
        case network::tanh:      return "tanh"      + op; 
        case network::invalid:   return "invalid"   + op; 
        default: break;
    }
    return "-1"; 
}

void NetOps::Apply(torch::nn::Sequential* nn, int nxt){
    std::string name = this -> Name(nxt); 
    switch(this -> nt){
        case network::linear:    (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n1, this -> _l1, nxt)); return; 
        case network::layernorm: (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n2, this -> _l1, nxt)); return;  
        case network::dropout:   (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n3, this -> _l1, nxt)); return; 
        case network::relu:      (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n4, this -> _l1, nxt)); return; 
        case network::silu:      (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n5, this -> _l1, nxt)); return; 
        case network::sigmoid:   (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n6, this -> _l1, nxt)); return; 
        case network::prelu:     (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n7, this -> _l1, nxt)); return; 
        case network::leakyrelu: (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n8, this -> _l1, nxt)); return;  
        case network::tanh:      (*nn) -> push_back(name.c_str(), utils::make_Fx(this -> n9, this -> _l1, nxt)); return; 
        case network::invalid:   abort(); 
        default: break;
    }
}









torch::nn::Linear    utils::make_Fx(torch::nn::Linear* tn, long int src, long int dst){
    return torch::nn::Linear(torch::nn::LinearOptions(src, (dst < 0) ? src : dst).bias(true)); 
}

torch::nn::LayerNorm utils::make_Fx(torch::nn::LayerNorm* tn, long int src, long int dst){
    return torch::nn::LayerNorm(torch::nn::LayerNormOptions({src})); 
}

torch::nn::Dropout  utils::make_Fx(torch::nn::Dropout* tn, long int n, long int  dst){
    double k = double(n) / 100.0; 
    return torch::nn::Dropout(torch::nn::DropoutOptions({k})); 
}

torch::nn::ReLU     utils::make_Fx(torch::nn::ReLU* tn, long int n, long int dst){
    return torch::nn::ReLU();
}

torch::nn::LeakyReLU utils::make_Fx(torch::nn::LeakyReLU* tn, long int n, long int dst){
    return torch::nn::LeakyReLU();
}

torch::nn::Tanh     utils::make_Fx(torch::nn::Tanh* tn, long int n, long int dst){
    return torch::nn::Tanh();
}

torch::nn::SiLU     utils::make_Fx(torch::nn::SiLU* tn, long int n, long int dst){
    return torch::nn::SiLU();
}

torch::nn::Sigmoid  utils::make_Fx(torch::nn::Sigmoid* tn, long int n, long int dst){
    return torch::nn::Sigmoid();
}

torch::nn::PReLU    utils::make_Fx(torch::nn::PReLU* tn, long int n, long int dst){
    return torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(n)); 
}

torch::nn::Sequential* utils::make_Network(std::string title, std::vector<NetOps> prm){
    std::vector<std::pair<std::string, torch::nn::Module>> data = {}; 
    torch::nn::Sequential* nn = new torch::nn::Sequential(); 
    for (size_t x(0); x < prm.size()-1; ++x){
        long ln = 0; 
        if (!x && prm[x+1]._l1 > 0){ln = prm[x+1]._l1;}
        std::string nm = std::to_string(x); 
        long int nxt = prm[x+1]._l1;
        prm[x].Apply(nn, nxt); 
        

         

    }  
    


    return nullptr; 
}

