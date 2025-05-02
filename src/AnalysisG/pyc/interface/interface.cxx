#include <pyc/pyc.h>

#include <tools/tensor_cast.h>
#include <tools/vector_cast.h>

#ifdef PYC_CUDA
#include <utils/utils.cuh>
#else 
#include <utils/utils.h>
#endif

neutrino::neutrino(){this -> type = "nunu";}
neutrino::neutrino(double px, double py, double pz){
    particle_t* p = &this -> data; 
    p -> px = px; p -> py = py; p -> pz = pz; p -> e = this -> e; 
    p -> polar = true; 
    this -> type = "nunu";
}

neutrino::~neutrino(){
    if (this -> bquark){delete this -> bquark;}
    if (this -> lepton){delete this -> lepton;}
    this -> bquark = nullptr;
    this -> lepton = nullptr; 
}


torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor>* inpt){
    torch::Dict<std::string, torch::Tensor> out;  
    std::map<std::string, torch::Tensor>::iterator itr = inpt -> begin(); 
    for (; itr != inpt -> end(); ++itr){out.insert(itr -> first, itr -> second);}
    return out; 
}

torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor> inpt){
    return pyc::std_to_dict(&inpt); 
}

torch::Tensor pyc::tensorize(std::vector<double>* inpt){
    torch::TensorOptions ops = torch::TensorOptions(torch::kCPU); 
    return build_tensor(inpt, torch::kDouble, double(), &ops).view({-1, int(inpt -> size())});
}

torch::Tensor pyc::tensorize(std::vector<long>* inpt){
    torch::TensorOptions ops = torch::TensorOptions(torch::kCPU); 
    return build_tensor(inpt, torch::kLong, long(), &ops).view({-1, int(inpt -> size())});
}

torch::Tensor pyc::tensorize(std::vector<std::vector<double>>* inpt){
    std::vector<torch::Tensor> mrg; 
    for (size_t x(0); x < inpt -> size(); ++x){mrg.push_back(pyc::tensorize(&(*inpt)[x]));}
    return torch::cat(mrg, {0}); 
}

torch::Tensor pyc::tensorize(std::vector<std::vector<long>>* inpt){
    std::vector<torch::Tensor> mrg; 
    for (size_t x(0); x < inpt -> size(); ++x){mrg.push_back(pyc::tensorize(&(*inpt)[x]));}
    return torch::cat(mrg, {0}); 
}


std::vector<neutrino*> construct_particle(torch::Tensor* inpt, torch::Tensor* ln, torch::Tensor* bn, std::vector<double>* dst){

    std::vector<std::vector<double>> pmc; 
    std::vector<signed long> s = tensor_size(inpt); 
    tensor_to_vector(inpt, &pmc, &s, double(0));
    std::vector<std::vector<long>> _l, _b; 
    if (ln){
        s = tensor_size(ln); 
        tensor_to_vector(ln, &_l, &s, long(0));
        tensor_to_vector(bn, &_b, &s, long(0));
   }
  
    size_t o = 0;  
    for (size_t x(0); x < dst -> size(); ++x){o += (*dst)[x] != 0;}
    if (!o){return {};}

    std::vector<neutrino*> out(o, nullptr); o = 0; 
    for (size_t x(0); x < pmc.size(); ++x){
        if (!(*dst)[x]){continue;}
        neutrino* nx = new neutrino(pmc[x][0], pmc[x][1], pmc[x][2]); 
        nx -> min   = (*dst)[x];
        nx -> b_idx = (bn) ? _b[x][0] : x / 6;
        nx -> l_idx = (ln) ? _l[x][0] : x / 6;
        out[o] = nx; ++o; 
    }
    return out; 
}

std::vector<std::pair<neutrino*, neutrino*>> pyc::nusol::NuNu(
       std::vector<std::vector<double>>* pmc_b1, std::vector<std::vector<double>>* pmc_b2, 
       std::vector<std::vector<double>>* pmc_l1, std::vector<std::vector<double>>* pmc_l2, 
       std::vector<double>* met               , std::vector<double>* phi, 
       std::vector<std::vector<double>>* mass1, std::vector<std::vector<double>>* mass2,
       std::string dev, const double null, const double step, const double tolerance, const unsigned int timeout
){
    torch::Tensor b1   = pyc::tensorize(pmc_b1); 
    torch::Tensor b2   = pyc::tensorize(pmc_b2); 
    torch::Tensor l1   = pyc::tensorize(pmc_l1); 
    torch::Tensor l2   = pyc::tensorize(pmc_l2); 
    torch::Tensor m1   = pyc::tensorize(mass1); 
    torch::Tensor m2   = pyc::tensorize(mass2); 
    torch::Tensor met_ = pyc::tensorize(met).view({-1, 1}); 
    torch::Tensor phi_ = pyc::tensorize(phi).view({-1, 1}); 
    
    b1   = changedev(dev, &b1  ); 
    b2   = changedev(dev, &b2  ); 
    l1   = changedev(dev, &l1  ); 
    l2   = changedev(dev, &l2  ); 
    m1   = changedev(dev, &m1  ); 
    m2   = changedev(dev, &m2  ); 
    met_ = changedev(dev, &met_); 
    phi_ = changedev(dev, &phi_); 

    torch::Tensor metxy = torch::cat({pyc::transform::separate::Px(met_, phi_), pyc::transform::separate::Py(met_, phi_)}, {-1}); 
    torch::Dict<std::string, torch::Tensor> nus = pyc::nusol::NuNu(b1, b2, l1, l2, metxy, null, m1, m2, step, tolerance, timeout);     
    torch::Tensor nu1 = nus.at("nu1").view({-1, 3}); 
    torch::Tensor nu2 = nus.at("nu2").view({-1, 3}); 
    torch::Tensor dis = nus.at("distances").view({-1}); 

    std::vector<double> dist; 
    tensor_to_vector(&dis, &dist); 

    std::vector<std::pair<neutrino*, neutrino*>> out; 
    std::vector<neutrino*> nu1_ = construct_particle(&nu1, nullptr, nullptr, &dist);  
    std::vector<neutrino*> nu2_ = construct_particle(&nu2, nullptr, nullptr, &dist); 
    for (size_t x(0); x < nu1_.size(); ++x){out.push_back({nu1_[x], nu2_[x]});}
    return out; 
}

std::vector<std::pair<neutrino*, neutrino*>> pyc::nusol::combinatorial(
        std::vector<double>* met_, std::vector<double>* phi_, std::vector<std::vector<double>>* pmc_, 
        std::vector<long>*   bth_,  std::vector<long>* is_b_, std::vector<long>* is_l_, std::string dev, 
        double mT, double mW, double null, double perturb, long steps
){

    std::vector<std::vector<long>> edge_index_ = {}; 
    edge_index_.push_back({});
    edge_index_.push_back({}); 
    for (size_t x(0); x < bth_ -> size(); ++x){
        long ix = (*bth_)[x]; 
        for (size_t y(0); y < bth_ -> size(); ++y){
            long iy = (*bth_)[y]; 
            if (iy != ix){continue;}
            edge_index_[0].push_back(long(x)); 
            edge_index_[1].push_back(long(y));
        }
    }
    if (!steps){std::cout << "FAILURE steps parameter set to 0" << std::endl; abort();}

    torch::Tensor met        = pyc::tensorize(met_); 
    torch::Tensor phi        = pyc::tensorize(phi_); 
    torch::Tensor pmc        = pyc::tensorize(pmc_); 
    torch::Tensor isb        = pyc::tensorize(is_b_); 
    torch::Tensor isl        = pyc::tensorize(is_l_); 
    torch::Tensor bth        = pyc::tensorize(bth_); 
    torch::Tensor edge_index = pyc::tensorize(&edge_index_); 

    pmc        = changedev(dev, &pmc);
    bth        = changedev(dev, &bth).view({-1});
    met        = changedev(dev, &met).view({-1, 1});
    phi        = changedev(dev, &phi).view({-1, 1}); 
    edge_index = changedev(dev, &edge_index);  

    torch::Tensor pid   = torch::cat({changedev(dev, &isl).view({-1, 1}), changedev(dev, &isb).view({-1, 1})}, {-1}); 
    torch::Tensor metxy = torch::cat({pyc::transform::separate::Px(met, phi), pyc::transform::separate::Py(met, phi)}, {-1}); 

    torch::Dict<std::string, torch::Tensor> nus = pyc::nusol::combinatorial(edge_index, bth, pmc, pid, metxy, mT, mW, null, perturb, steps); 
    if (!nus.contains("nu1")){return {};}
    torch::Tensor nu1 = nus.at("nu1");
    torch::Tensor nu2 = nus.at("nu2");
    torch::Tensor l1  = nus.at("l1");
    torch::Tensor l2  = nus.at("l2");
    torch::Tensor b1  = nus.at("b1");
    torch::Tensor b2  = nus.at("b2");
    torch::Tensor dis = nus.at("distances");

    std::vector<double> dist; 
    tensor_to_vector(&dis, &dist); 
    std::vector<neutrino*> nu1_ = construct_particle(&nu1, &l1, &b1, &dist);  
    std::vector<neutrino*> nu2_ = construct_particle(&nu2, &l2, &b2, &dist); 
    std::vector<std::pair<neutrino*, neutrino*>> out = {}; 
    for (size_t x(0); x < nu1_.size(); ++x){out.push_back({nu1_[x], nu2_[x]});}
    return out; 
}

