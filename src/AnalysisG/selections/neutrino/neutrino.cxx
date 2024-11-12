#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <tools/tensor_cast.h>
#include <tools/vector_cast.h>
#include <nusol/nusol-cuda.h>
#include "neutrino.h"

neutrino::neutrino(){this -> name = "neutrino";}
neutrino::~neutrino(){}

selection_template* neutrino::clone(){
    return (selection_template*)new neutrino();
}

void neutrino::merge(selection_template* sl){
    neutrino* slt = (neutrino*)sl; 

    merge_data(&this -> delta_met  , &slt -> delta_met);  
    merge_data(&this -> delta_metnu, &slt -> delta_metnu);  
    merge_data(&this -> obs_met    , &slt -> obs_met);  
    merge_data(&this -> nus_met    , &slt -> nus_met);  
    merge_data(&this -> dist_nu    , &slt -> dist_nu);  

    merge_data(&this -> pdgid      , &slt -> pdgid);  
    merge_data(&this -> tru_topmass, &slt -> tru_topmass);  
    merge_data(&this -> tru_wmass  , &slt -> tru_wmass);  
    merge_data(&this -> nusol_tmass, &slt -> nusol_tmass);  
    merge_data(&this -> nusol_wmass, &slt -> nusol_wmass);  
    merge_data(&this -> exp_topmass, &slt -> exp_topmass);  
    merge_data(&this -> exp_wmass  , &slt -> exp_wmass);  
}

bool neutrino::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev;
    if (evn -> hash != "0xd0edab7810edc54f"){return false;}
    std::vector<top*> tops = this -> upcast<top>(&evn -> Tops); 
    if (tops.size() != 4){return false;}
    int num_leps = 0; 
    for (size_t x(0); x < tops.size(); ++x){
        std::map<std::string, particle_template*> ch = tops[x] -> children; 
        std::vector<top_children*> ch_ = this -> upcast<top_children>(&ch);  
        for (size_t i(0); i < ch_.size(); ++i){
            bool lp = ch_[i] -> is_lep; 
            if (!lp){continue;}
            num_leps += lp;
            break; 
        }
    }
    return num_leps == 2; // || num_leps == 1;
}

std::vector<nu> neutrino::build_nus(
                std::vector<long>* isb_, std::vector<long>* isl_,
                std::vector<double>* pt_, std::vector<double>* eta_, 
                std::vector<double>* phi_, std::vector<double>* energy_,
                double met, double phi, double scale
){
    std::vector<double> _phi = {phi}; 
    std::vector<double> _met = {met / scale};
    std::vector<std::vector<long>> edge_index = {{}, {}}; 
    for (size_t x(0); x < isb_ -> size(); ++x){
        for (size_t y(0); y < isb_ -> size(); ++y){
            edge_index[0].push_back(x);
            edge_index[1].push_back(y); 
        }
    }

    torch::TensorOptions ops = torch::TensorOptions(torch::kCPU); 
    torch::Tensor src  = build_tensor(&edge_index[0], torch::kLong, long(), &ops).view({1, -1}); 
    torch::Tensor dst  = build_tensor(&edge_index[1], torch::kLong, long(), &ops).view({1, -1}); 
    torch::Tensor topo = torch::cat({src, dst}, {0}).to(torch::kCUDA); 

    torch::Tensor phit = build_tensor(&_phi  , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor mett = build_tensor(&_met  , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor metxy = torch::cat({transform::cuda::Px(mett, phit), transform::cuda::Py(mett, phit)}, {-1}); 

    torch::Tensor m_pt  = build_tensor(pt_    , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_eta = build_tensor(eta_   , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_phi = build_tensor(phi_   , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_enx = build_tensor(energy_, torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_isb = build_tensor(isb_   , torch::kLong  , long(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_isl = build_tensor(isl_   , torch::kLong  , long(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor pmc  = transform::cuda::PxPyPzE(m_pt, m_eta, m_phi, m_enx); 
    torch::Tensor pid  = torch::cat({m_isl, m_isb}, {-1}); 
    torch::Tensor bth  = torch::zeros_like(m_isl).view({-1}); 

    std::cout << std::endl; 
    double mw = this -> massw / scale; 
    double mt = this -> masstop / scale;  
    std::map<std::string, torch::Tensor> nuxt;
    nuxt = nusol::cuda::combinatorial(topo, bth, pmc, pid, metxy, mt, mw, 0.0, 0.95, 0.95, 1e-8, this -> steps); 
    torch::Tensor combi = nuxt["combi"].sum({-1}) > 0;
    if (!combi.index({combi}).size({0})){return {};}
    torch::Tensor b1 = nuxt["combi"].index({combi, 0}).to(torch::kInt); 
    torch::Tensor b2 = nuxt["combi"].index({combi, 1}).to(torch::kInt); 
    torch::Tensor l1 = nuxt["combi"].index({combi, 2}).to(torch::kInt); 
    torch::Tensor l2 = nuxt["combi"].index({combi, 3}).to(torch::kInt);

    torch::Tensor w1 = nuxt["nu_1f"] + pmc.index({l1}); 
    torch::Tensor w2 = nuxt["nu_2f"] + pmc.index({l2}); 
    torch::Tensor wmass = physics::cuda::M(torch::cat({w1, w2}, {0})).view({-1}); 

    torch::Tensor t1 = nuxt["nu_1f"] + pmc.index({l1}) + pmc.index({b1}); 
    torch::Tensor t2 = nuxt["nu_2f"] + pmc.index({l2}) + pmc.index({b2}); 
    torch::Tensor tmass = physics::cuda::M(torch::cat({t1, t2}, {0})).view({-1}); 

    nuxt["nu_1f"] = nuxt["nu_1f"].view({-1}); 
    nuxt["ms_1f"] = nuxt["ms_1f"].view({-1}); 
    nuxt["nu_2f"] = nuxt["nu_2f"].view({-1}); 
    nuxt["ms_2f"] = nuxt["ms_2f"].view({-1}); 
    nuxt["min"]   = nuxt["min"].view({-1});

    std::vector<double> w_mass, t_mass, nu1f_, nu2f_, expm1_, expm2_, dist; 
    tensor_to_vector(&wmass, &w_mass); 
    tensor_to_vector(&tmass, &t_mass); 
    tensor_to_vector(&nuxt["min"], &dist); 
    tensor_to_vector(&nuxt["nu_1f"], &nu1f_); 
    tensor_to_vector(&nuxt["nu_2f"], &nu2f_); 
    tensor_to_vector(&nuxt["ms_1f"], &expm1_); 
    tensor_to_vector(&nuxt["ms_2f"], &expm2_); 
  
    l1 = l1.view({-1}); 
    l2 = l2.view({-1}); 
    std::vector<int> l1_, l2_; 
    tensor_to_vector(&l1, &l1_); 
    tensor_to_vector(&l2, &l2_); 

    std::vector<nu> out;  
    out.push_back(nu(nu1f_[0], nu1f_[1], nu1f_[2], nu1f_[3])); 
    out[0].exp_wmass   = w_mass[0]; 
    out[0].exp_tmass   = t_mass[0]; 
    out[0].nusol_wmass = expm1_[0]; 
    out[0].nusol_tmass = expm1_[1]; 
    out[0].min = dist[0]; 
    out[0].idx = l1_[0]; 

    out.push_back(nu(nu2f_[0], nu2f_[1], nu2f_[2], nu2f_[3])); 
    out[1].exp_wmass   = w_mass[1]; 
    out[1].exp_tmass   = t_mass[1]; 
    out[1].nusol_wmass = expm2_[0]; 
    out[1].nusol_tmass = expm2_[1]; 
    out[1].min = dist[0]; 
    out[1].idx = l2_[0]; 
    return out; 
}

bool neutrino::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::string hash = evn -> hash; 

    // ------------ find the tops that decay leptonically --------------- //    
    std::vector<top*> matched;  
    std::vector<top_children*> nus, leps, bs;
    std::vector<top*> tops = this -> upcast<top>(&evn -> Tops); 
    for (size_t x(0); x < tops.size(); ++x){
        top_children* b_   = nullptr; 
        top_children* nu_  = nullptr;
        top_children* lep_ = nullptr; 
        std::vector<top_children*> add = {}; 
        std::map<std::string, particle_template*> ch = tops[x] -> children; 
        std::vector<top_children*> ch_ = this -> upcast<top_children>(&ch);  
        for (size_t i(0); i < ch_.size(); ++i){
            if (ch_[i] -> is_lep){lep_ = ch_[i]; continue;}
            if (ch_[i] -> is_nu){  nu_ = ch_[i]; continue;}
            if (ch_[i] -> is_b){    b_ = ch_[i]; continue;}
            add.push_back(ch_[i]);  // need to add any additional jets to the b-quark!
        }
        
        if (!b_ || !nu_ || !lep_){continue;}
        matched.push_back(tops[x]); 
        bs.push_back(b_);  
        nus.push_back(nu_); 
        leps.push_back(lep_); 
        this -> tru_topmass[hash].push_back(this -> sum(&ch_));
        this -> tru_wmass[hash].push_back((*nu_ + *lep_).mass/this -> scale); 
    }

    particle_template* all_children = nullptr; 
    this -> sum(&evn -> Children, &all_children); 
    if (!all_children){return false;}

    particle_template* all_nus = nullptr; 
    this -> sum(&nus, &all_nus); 
    if (!all_nus){return false;}

    double met = evn -> met; 
    double phi = evn -> phi; 
    
    this -> delta_met[hash] = std::abs(all_children -> pt - met) / this -> scale; 
    this -> delta_metnu[hash] = std::abs(all_nus -> pt - met) / this -> scale; 
    this -> nus_met[hash] = all_nus -> pt / this -> scale;
    this -> obs_met[hash] = met / this -> scale; 

    std::vector<nu> ch_nus = this -> get_neutrinos(&leps, &bs, -all_nus -> pt, std::abs(3.14159 - all_nus -> phi), this -> scale); 
    for (nu &p : ch_nus){
        this -> pdgid[hash].push_back(p.leppid); 
        this -> nusol_tmass[hash].push_back(p.nusol_tmass);
        this -> nusol_wmass[hash].push_back(p.nusol_wmass);

        this -> exp_topmass[hash].push_back(p.exp_tmass);
        this -> exp_wmass[hash].push_back(p.exp_wmass);
        this -> dist_nu[hash] = p.min;
    }
    return true; 
}
