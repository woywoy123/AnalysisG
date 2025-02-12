#include <tools/tensor_cast.h>
#include <tools/vector_cast.h>
#include <pyc/cupyc.h>

#include "combinatorial.h"

combinatorial::combinatorial(){this -> name = "combinatorial";}
combinatorial::~combinatorial(){}
selection_template* combinatorial::clone(){return (selection_template*)new combinatorial();}

void combinatorial::merge(selection_template* sl){
    combinatorial* slt = (combinatorial*)sl; 
    merge_data(&this -> output, &slt -> output);  
}

bool combinatorial::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev;
    std::vector<particle_template*> tops = evn -> Tops; 
    if (tops.size() != 4){return false;}
    int num_leps = 0; 
    for (size_t x(0); x < tops.size(); ++x){
        std::vector<particle_template*> ch_ = this -> vectorize(&tops[x] -> children);  
        for (size_t i(0); i < ch_.size(); ++i){
            bool lp = ch_[i] -> is_lep; 
            if (!lp){continue;}
            num_leps += lp;
            break; 
        }
    }
    return num_leps == 2; // || num_leps == 1;
}

torch::Tensor tensorize(std::vector<double>* inpt){
    torch::TensorOptions ops = torch::TensorOptions(torch::kCPU); 
    return build_tensor(inpt, torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, int(inpt -> size())});
}

torch::Tensor tensorize(std::vector<long>* inpt){
    torch::TensorOptions ops = torch::TensorOptions(torch::kCPU); 
    return build_tensor(inpt, torch::kLong, long(), &ops).to(torch::kCUDA).view({-1, int(inpt -> size())});
}

torch::Tensor pxpypze(particle_template* pc){
    std::vector<double> pmc = {pc -> px, pc -> py, pc -> pz, pc -> e};
    return tensorize(&pmc); 
}


nu* construct_particle(torch::Tensor* inpt, std::vector<double>* dst){

    std::vector<std::vector<double>> pmc; 
    std::vector<signed long> s = tensor_size(inpt); 
    tensor_to_vector(inpt, &pmc, &s, double(0));

    int idx = -1; 
    double lst = 0; 
    for (size_t x(0); x < pmc.size(); ++x){
        double d = dst -> at(x); 
        if (!d){continue;}
        bool tx = lst > d;
        if (!tx){continue;}
        idx = x;
        lst = d; 
    }
    if (idx == -1){return nullptr;}
    std::vector<double> solx = pmc[idx]; 
    nu* nx = new nu(solx[0], solx[1], solx[2]); 
    nx -> min = lst; 
    return nx; 
}


std::vector<nu*> combinatorial::build_nus(
    std::vector<long>* isb_, std::vector<long>* isl_,
    std::vector<particle_template*>* bqs, std::vector<particle_template*>* leps, 
    double met, double phi
){
    std::vector<double> _phi = {phi}; 
    std::vector<double> _met = {met};
    torch::Tensor phi_ = tensorize(&_phi); 
    torch::Tensor met_ = tensorize(&_met); 
    torch::Tensor metxy = torch::cat({
            pyc::transform::separate::Px(met_, phi_), 
            pyc::transform::separate::Py(met_, phi_)
    }, {-1}); 

    std::vector<torch::Tensor> pmv = {}; 
    for (size_t x(0); x < leps -> size(); ++x){pmv.push_back(pxpypze((*leps)[x]));}
    for (size_t x(0); x <  bqs -> size(); ++x){pmv.push_back(pxpypze((*bqs)[x]));}
    torch::Tensor pid = torch::cat({tensorize(isl_).view({-1, 1}), tensorize(isb_).view({-1, 1})}, {-1}); 
    torch::Tensor bth = torch::zeros_like(tensorize(isb_)).view({-1}); 
    torch::Tensor pmc = torch::cat(pmv, {0}); 

    unsigned int lx = bqs -> size() + leps -> size(); 
    std::vector<std::vector<long>> edge_index = {{}, {}}; 
    for (size_t x(0); x < lx; ++x){
        for (size_t y(0); y < lx; ++y){
            edge_index[0].push_back(x);
            edge_index[1].push_back(y); 
        }
    }

    torch::Tensor src  = tensorize(&edge_index[0]).view({1, -1}); 
    torch::Tensor dst  = tensorize(&edge_index[1]).view({1, -1}); 
    torch::Tensor topo = torch::cat({src, dst}, {0}); 

    double mw = this -> massw / this -> scale; 
    double mt = this -> masstop / this -> scale;  

    torch::Dict<std::string, torch::Tensor> nuxt;
    nuxt = pyc::nusol::combinatorial(topo, bth, pmc, pid, metxy, mt, mw, 0.95, 0.995, this -> steps, 1e-8); 
    torch::Tensor nu1 = nuxt.at("nu1"); 
    torch::Tensor nu2 = nuxt.at("nu2"); 
    torch::Tensor distx = nuxt.at("distances"); 

    std::vector<double> w_mass, t_mass, dist; 
    tensor_to_vector(&distx, &dist); 
    if (dist[0] == 0){return {};}
    
    nu* nu1_ = construct_particle(&nu1, &dist);  
    nu* nu2_ = construct_particle(&nu2, &dist);  
    if (!nu1_ || !nu2_){
        if (nu1_){delete nu1_;}
        if (nu2_){delete nu2_;}
        return {}; 
    }

    torch::Tensor l1  = pmc.index({nuxt.at("l1").view({-1})}); 
    torch::Tensor l2  = pmc.index({nuxt.at("l2").view({-1})});
    torch::Tensor b1  = pmc.index({nuxt.at("b1").view({-1})}); 
    torch::Tensor b2  = pmc.index({nuxt.at("b2").view({-1})}); 

    torch::Tensor w1 = nu1 + l1; 
    torch::Tensor w2 = nu2 + l2; 
    torch::Tensor wmass = pyc::physics::cartesian::combined::M(torch::cat({w1, w2}, {0})).view({-1}); 
    tensor_to_vector(&wmass, &w_mass); 

    torch::Tensor t1 = nu1 + l1 + b1; 
    torch::Tensor t2 = nu2 + l2 + b2; 
    torch::Tensor tmass = pyc::physics::cartesian::combined::M(torch::cat({t1, t2}, {0})).view({-1}); 
    tensor_to_vector(&tmass, &t_mass); 

    l1 = nuxt.at("l1").view({-1}); 
    l2 = nuxt.at("l2").view({-1}); 
    std::vector<long> l1_, l2_; 
    tensor_to_vector(&l1, &l1_); 
    tensor_to_vector(&l2, &l2_); 

    nu1_ -> exp_wmass = w_mass[0] / 1000; 
    nu1_ -> exp_tmass = t_mass[0] / 1000; 
    nu1_ -> min = dist[0]; 
    nu1_ -> idx = l1_[0]; 

    nu2_ -> exp_wmass = w_mass[1] / 1000; 
    nu2_ -> exp_tmass = t_mass[1] / 1000; 
    nu2_ -> min = dist[0]; 
    nu2_ -> idx = l2_[0]; 
    return {nu1_, nu2_};
}

std::vector<nu*> combinatorial::get_baseline(
    std::vector<particle_template*>* bqs, std::vector<particle_template*>* lpt, 
    std::vector<double>* tps, std::vector<double>* wbs, double met, double phi
){
    std::vector<double> _phi = {phi}; 
    std::vector<double> _met = {met}; 

    std::vector<double> tm1 = {(*tps)[0], (*wbs)[0]}; 
    torch::Tensor m1t = tensorize(&tm1); 

    std::vector<double> tm2 = {(*tps)[1], (*wbs)[1]}; 
    torch::Tensor m2t = tensorize(&tm2); 

    torch::Tensor b1t = pxpypze((*bqs)[0]);
    torch::Tensor b2t = pxpypze((*bqs)[1]);

    torch::Tensor l1t = pxpypze((*lpt)[0]);
    torch::Tensor l2t = pxpypze((*lpt)[1]);

    torch::Tensor _phit = tensorize(&_phi); 
    torch::Tensor _mett = tensorize(&_met); 
    torch::Tensor metxy = torch::cat({
            pyc::transform::separate::Px(_mett, _phit), 
            pyc::transform::separate::Py(_mett, _phit)
    }, {-1});

    torch::Dict res = pyc::nusol::NuNu(b1t, b2t, l1t, l2t, metxy, 1e-10, m1t, m2t);   
    torch::Tensor nu1 = res.at("nu1").view({-1, 3});
    torch::Tensor nu2 = res.at("nu2").view({-1, 3});
    torch::Tensor dst = res.at("distances").view({-1}); 

    std::vector<double> distance; 
    tensor_to_vector(&dst, &distance); 
    if (distance[0] == 0){return {};}
    nu* nu1_ = construct_particle(&nu1, &distance);  
    nu* nu2_ = construct_particle(&nu2, &distance);  
    if (nu1_ && nu2_){return {nu1_, nu2_};}
    if (nu1_){delete nu1_;}
    if (nu2_){delete nu2_;}
    return {}; 
}

bool combinatorial::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::string hash = evn -> hash; 

    // ------------ find the tops that decay leptonically --------------- //    
    std::vector<particle_template*> nus, leps, bs, tps; 
    std::vector<particle_template*> tops = evn -> Tops; 
    std::vector<double> tmass, wmass; 
    for (size_t x(0); x < tops.size(); ++x){
        particle_template* b_   = nullptr; 
        particle_template* nu_  = nullptr;
        particle_template* lep_ = nullptr; 

        std::vector<particle_template*> ch_ = this -> vectorize(&tops[x] -> children); 
        for (size_t i(0); i < ch_.size(); ++i){
            if (ch_[i] -> is_lep){lep_ = ch_[i]; continue;}
            if (ch_[i] -> is_nu){  nu_ = ch_[i]; continue;}
            if (ch_[i] -> is_b){    b_ = ch_[i]; continue;}
        }
        if (!b_ || !nu_ || !lep_){continue;}
        bs.push_back(b_);  
        nus.push_back(nu_); 
        leps.push_back(lep_); 

        particle_template* tpsx = nullptr; 
        this -> sum(&ch_, &tpsx); 
        tps.push_back(tpsx); 
        tmass.push_back(tops[x] -> mass); 
        wmass.push_back((*lep_ + *nu_).mass); 
        if (leps.size() == 2 && bs.size() == 2){break;}
    }

    particle_template* all_children = nullptr; 
    this -> sum(&evn -> Children, &all_children); 
    if (!all_children){return false;}

    particle_template* all_nus = nullptr; 
    this -> sum(&nus, &all_nus); 
    if (!all_nus){return false;}

    event_data* evx = &this -> output[hash]; 
    evx -> delta_met    = (all_children -> pt - evn -> met) / 1000; 
    evx -> delta_metnu  = (all_nus -> pt - evn -> met) / 1000; 
    evx -> observed_met = evn -> met / 1000; 
    evx -> neutrino_met = all_nus -> pt / 1000;

    for (size_t x(0); x < nus.size(); ++x){
        evx -> truth_neutrinos.push_back(new nu(&nus[x] -> data));
        evx -> bquark.push_back(new particle(&bs[x] -> data));
        evx -> lepton.push_back(new particle(&leps[x] -> data)); 
        evx -> tops.push_back(new particle(&tps[x] -> data)); 
    }

    evx -> cobs_neutrinos = this -> get_neutrinos(&bs, &leps,    evn -> met,     evn -> phi); 
    evx -> cmet_neutrinos = this -> get_neutrinos(&bs, &leps, all_nus -> pt, all_nus -> phi); 

    evx -> robs_neutrinos = this -> get_baseline(&bs, &leps, &tmass, &wmass,    evn -> met,     evn -> phi); 
    evx -> rmet_neutrinos = this -> get_baseline(&bs, &leps, &tmass, &wmass, all_nus -> pt, all_nus -> phi); 
    return true; 
}
