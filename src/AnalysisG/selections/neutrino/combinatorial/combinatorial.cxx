#include <tools/tensor_cast.h>
#include <tools/vector_cast.h>
#include <pyc/cupyc.h>
#include "combinatorial.h"

combinatorial::combinatorial(){this -> name = "combinatorial";}
combinatorial::~combinatorial(){}

selection_template* combinatorial::clone(){
    return (selection_template*)new combinatorial();
}

void combinatorial::merge(selection_template* sl){
    combinatorial* slt = (combinatorial*)sl; 

    merge_data(&this -> delta_met  , &slt -> delta_met);  
    merge_data(&this -> delta_metnu, &slt -> delta_metnu);  
    merge_data(&this -> obs_met    , &slt -> obs_met);  
    merge_data(&this -> nus_met    , &slt -> nus_met);  
    merge_data(&this -> dist_nu    , &slt -> dist_nu);  

    merge_data(&this -> pdgid      , &slt -> pdgid);  
    merge_data(&this -> tru_topmass, &slt -> tru_topmass);  
    merge_data(&this -> tru_wmass  , &slt -> tru_wmass);  

    merge_data(&this -> exp_topmass, &slt -> exp_topmass);  
    merge_data(&this -> exp_wmass  , &slt -> exp_wmass);  
}

bool combinatorial::selection(event_template* ev){
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

std::vector<nu> combinatorial::build_nus(
                std::vector<long>* isb_, std::vector<long>* isl_,
                std::vector<double>* pt_, std::vector<double>* eta_, 
                std::vector<double>* phi_, std::vector<double>* energy_,
                double met, double phi, double scale
){
    std::vector<double> _phi = {phi}; 
    std::vector<double> _met = {met};
    std::vector<std::vector<long>> edge_index = {{}, {}}; 
    for (size_t x(0); x < pt_ -> size(); ++x){
        for (size_t y(0); y < pt_ -> size(); ++y){
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
    torch::Tensor metxy = torch::cat({
            pyc::transform::separate::Px(mett, phit), 
            pyc::transform::separate::Py(mett, phit)
    }, {-1}); 

    torch::Tensor m_pt  = build_tensor(pt_    , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_eta = build_tensor(eta_   , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_phi = build_tensor(phi_   , torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_enx = build_tensor(energy_, torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_isb = build_tensor(isb_   , torch::kLong  , long(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor m_isl = build_tensor(isl_   , torch::kLong  , long(), &ops).to(torch::kCUDA).view({-1, 1}); 
    torch::Tensor pmc   = pyc::transform::separate::PxPyPzE(m_pt, m_eta, m_phi, m_enx); 
    torch::Tensor pid   = torch::cat({m_isl, m_isb}, {-1}); 
    torch::Tensor bth   = torch::zeros_like(m_isl).view({-1}); 

    double mw = this -> massw / scale; 
    double mt = this -> masstop / scale;  
    torch::Dict<std::string, torch::Tensor> nuxt;

    std::cout << "::::" << std::endl;
    std::cout << topo << std::endl; 
    std::cout << bth << std::endl;
    std::cout << pid << std::endl;
    std::cout << metxy << std::endl;
    std::cout << pmc << std::endl;
    std::cout << mw << " " << mt << std::endl; 

    nuxt = pyc::nusol::combinatorial(topo, bth, pmc, pid, metxy, mt, mw, 0.95, 0.95, this -> steps, 1e-10); 

    std::cout << nuxt.at("nu1") << std::endl; 




    torch::Tensor l1  = pmc.index({nuxt.at("l1").view({-1})}); 
    torch::Tensor l2  = pmc.index({nuxt.at("l2").view({-1})});
    torch::Tensor b1  = pmc.index({nuxt.at("b1").view({-1})}); 
    torch::Tensor b2  = pmc.index({nuxt.at("b2").view({-1})}); 

    torch::Tensor nu1 = torch::cat({nuxt.at("nu1"), nuxt.at("nu1").pow(2).sum({-1}, true).sqrt()}, {-1}); 
    torch::Tensor nu2 = torch::cat({nuxt.at("nu2"), nuxt.at("nu2").pow(2).sum({-1}, true).sqrt()}, {-1}); 
    torch::Tensor distx = nuxt.at("distances"); 

    torch::Tensor w1 = nu1 + l1; 
    torch::Tensor w2 = nu2 + l2; 
    torch::Tensor wmass = pyc::physics::cartesian::combined::M(torch::cat({w1, w2}, {0})).view({-1}); 

    torch::Tensor t1 = nu1 + l1 + b1; 
    torch::Tensor t2 = nu2 + l2 + b2; 
    torch::Tensor tmass = pyc::physics::cartesian::combined::M(torch::cat({t1, t2}, {0})).view({-1}); 

    std::vector<double> w_mass, t_mass, nu1f_, nu2f_, dist; 
    tensor_to_vector(&distx, &dist); 
    //if (dist[0] >= -10){return {};}

    tensor_to_vector(&wmass, &w_mass); 
    tensor_to_vector(&tmass, &t_mass); 
    tensor_to_vector(&nu1, &nu1f_); 
    tensor_to_vector(&nu2, &nu2f_); 

    l1 = nuxt.at("l1").view({-1}); 
    l2 = nuxt.at("l2").view({-1}); 
    std::vector<long> l1_, l2_; 
    tensor_to_vector(&l1, &l1_); 
    tensor_to_vector(&l2, &l2_); 

    std::vector<nu> out;  
    out.push_back(nu(nu1f_[0], nu1f_[1], nu1f_[2], nu1f_[3])); 
    out[0].exp_wmass   = w_mass[0] / 1000; 
    out[0].exp_tmass   = t_mass[0] / 1000; 
    out[0].min = dist[0]; 
    out[0].idx = l1_[0]; 

    out.push_back(nu(nu2f_[0], nu2f_[1], nu2f_[2], nu2f_[3])); 
    out[1].exp_wmass   = w_mass[1] / 1000; 
    out[1].exp_tmass   = t_mass[1] / 1000; 
    out[1].min = dist[0]; 
    out[1].idx = l2_[0]; 
    return out;
}

bool combinatorial::strategy(event_template* ev){
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
        this -> tru_wmass[hash].push_back((*nu_ + *lep_).mass / 1000); 
        if (leps.size() == 2 && bs.size() == 2){break;}
    }

    particle_template* all_children = nullptr; 
    this -> sum(&evn -> Children, &all_children); 
    if (!all_children){return false;}

    particle_template* all_nus = nullptr; 
    this -> sum(&nus, &all_nus); 
    if (!all_nus){return false;}

    double met = evn -> met; 
    double phi = evn -> phi; 
   
    //std::cout << std::endl;
    //std::cout << met << " " << phi << std::endl;
    //std::cout << all_nus -> pt << " " << all_nus -> phi << std::endl;  
    this -> delta_met[hash] = std::abs(all_children -> pt - met) / 1000; 
    this -> delta_metnu[hash] = std::abs(all_nus -> pt - met) / 1000; 
    this -> nus_met[hash] = all_nus -> pt / 1000;
    this -> obs_met[hash] = met / 1000;

    std::vector<nu> ch_nus = this -> get_neutrinos(&leps, &bs, all_nus -> pt, all_nus -> phi, this -> scale); 
    for (nu &p : ch_nus){
        this -> pdgid[hash].push_back(p.leppid); 

        this -> exp_topmass[hash].push_back(p.exp_tmass);
        this -> exp_wmass[hash].push_back(p.exp_wmass);
        this -> dist_nu[hash] = p.min;
    }
    return true; 
}
