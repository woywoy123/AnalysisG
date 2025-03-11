#include "validation.h"

#include <tools/tensor_cast.h>
#include <tools/vector_cast.h>
#include <pyc/pyc.h>

validation::validation(){this -> name = "validation";}
validation::~validation(){}
selection_template* validation::clone(){return (selection_template*)new validation();}

void validation::merge(selection_template* sl){
    validation* slt = (validation*)sl; 
    merge_data(&this -> data_out, &slt -> data_out); 
}

bool validation::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<top*> tops = this -> upcast<top>(&evn -> Tops);
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
    return num_leps == 2; //  || num_leps == 1;  
}


torch::Tensor tensorize(std::vector<double>* inpt){
    torch::TensorOptions ops = torch::TensorOptions(torch::kCPU); 
    return build_tensor(inpt, torch::kDouble, double(), &ops).to(torch::kCUDA).view({-1, int(inpt -> size())});
}

torch::Tensor pxpypze(particle_template* pc){
    std::vector<double> pmc = {pc -> px, pc -> py, pc -> pz, pc -> e};
    return tensorize(&pmc); 
}


std::vector<nu*> construct_particle(torch::Tensor* inpt, std::vector<double>* dst){
    std::vector<std::vector<double>> pmc; 
    std::vector<signed long> s = tensor_size(inpt); 
    tensor_to_vector(inpt, &pmc, &s, double(0));
    std::vector<nu*> outpt = {}; 
    outpt.reserve(dst -> size()); 
    for (size_t x(0); x < pmc.size(); ++x){
        double d = dst -> at(x); 
        if (!d){continue;}
        std::vector<double> solx = pmc[x]; 
        nu* nx = new nu(solx[0], solx[1], solx[2]); 
        nx -> distance = d; 
        outpt.push_back(nx); 
    }
    return outpt; 
}

double compute_chi2(nu* v, nu* vt){
    double vx2 = std::pow(v -> px - vt -> px, 2); 
    double vy2 = std::pow(v -> py - vt -> py, 2); 
    double vz2 = std::pow(v -> pz - vt -> pz, 2); 
    return vx2 + vy2 + vz2; 
}

std::vector<nu*> validation::build_neutrinos(
                std::vector<bquark*>* bqs, std::vector<lepton*>* lpt, 
                std::vector<tquark*>* tps, std::vector<boson*>* wbs,
                double met, double phi, std::vector<nu*>* truth
){
    std::vector<double> _phi = {phi}; 
    std::vector<double> _met = {met}; 

    std::vector<double> m1, m2; 
    if (tps && wbs){
        m1.push_back((*tps)[0] -> mass); 
        m1.push_back((*wbs)[0] -> mass); 
        m2.push_back((*tps)[1] -> mass); 
        m2.push_back((*wbs)[1] -> mass); 
    }
    else {
        m1 = {this -> masstop, this -> massw}; 
        m2 = {this -> masstop, this -> massw}; 
    }

    torch::Tensor m1t = tensorize(&m1); 
    torch::Tensor m2t = tensorize(&m2); 

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

    #ifdef PYC_CUDA
    torch::Dict res = pyc::nusol::NuNu(b1t, b2t, l1t, l2t, metxy, 10e-10, m1t, m2t);   
    torch::Tensor nu1 = res.at("nu1").view({-1, 3});
    torch::Tensor nu2 = res.at("nu2").view({-1, 3});
    torch::Tensor dst = res.at("distances").view({-1}); 
    torch::Tensor pst = res.at("passed"); 

    std::vector<double> px; 
    tensor_to_vector(&pst, &px); 
    bool passed = px[0] == 1; 
    if (!passed){return {};}

    std::vector<double> distance; 
    tensor_to_vector(&dst, &distance); 
    std::vector<nu*> nu1_ = construct_particle(&nu1, &distance);  
    std::vector<nu*> nu2_ = construct_particle(&nu2, &distance);  
    if (!nu1_.size() && !nu2_.size()){return {};}
    nu* nu_1t = truth -> at(0); 
    nu* nu_2t = truth -> at(1); 

    int bst(0);  
    double chi2(-1);
    for (unsigned int x(0); x < nu1_.size(); ++x){
        nu* nx1 = nu1_[x]; 
        nu* nx2 = nu2_[x]; 
        double v1 = compute_chi2(nx1, nu_1t); 
        double v2 = compute_chi2(nx2, nu_2t); 
        if (!x){chi2 = v1 + v2; continue;}
        if (chi2 < v1 + v2){continue;}
        chi2 = v1 + v2; 
        bst = x; 
    }

    for (unsigned int x(0); x < nu1_.size(); ++x){
        if (x == bst){continue;}
        delete nu1_[x]; 
        delete nu2_[x]; 
    }
    return {nu1_[bst], nu2_[bst]}; 
    #else
    return {}; 
    #endif

}

bool validation::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::string hash_ = evn -> hash; 
    package* pkl = &this -> data_out[hash_];

    pkl -> met = evn -> met; 
    pkl -> phi = evn -> phi; 

    // ------------ find the tops that decay leptonically --------------- //    
    std::vector<particle_template*> b_jets; 
    std::vector<particle_template*> b_truthjets; 

    std::vector<particle_template*> jets_tops; 
    std::vector<particle_template*> ljets_tops; 
    std::vector<particle_template*> truthjets_tops; 
    
    std::vector<particle_template> reco_bosons; 
    std::vector<particle_template*> reco_leptons; 

    std::vector<top*> tops = this -> upcast<top>(&evn -> Tops); 
    for (size_t x(0); x < tops.size(); ++x){
        top* txp = tops[x]; 
        particle_template* b_   = nullptr; 
        particle_template* nu_  = nullptr;
        particle_template* lep_ = nullptr; 
        std::map<std::string, particle_template*> ch = txp -> children; 
        std::vector<top_children*> ch_ = this -> upcast<top_children>(&ch);  
        for (size_t i(0); i < ch_.size(); ++i){
            if (ch_[i] -> is_lep){lep_ = (particle_template*)ch_[i]; continue;}
            if (ch_[i] -> is_nu){  nu_ = (particle_template*)ch_[i]; continue;}
            if (ch_[i] -> is_b){    b_ = (particle_template*)ch_[i]; continue;}
        }
        if (!b_ || !nu_ || !lep_){continue;}

        particle_template bosonx = (*nu_ + *lep_); 
        pkl -> truth_tops.push_back(new tquark(&tops[x] -> data));
        pkl -> truth_bosons.push_back(new boson(&bosonx.data)); 
        pkl -> truth_bquarks.push_back(new bquark(&b_ -> data));
        pkl -> truth_leptons.push_back(new lepton(&lep_ -> data));
        pkl -> truth_nus.push_back(new nu(&nu_ -> data));

        b_ = nullptr; 
        std::vector<truthjet*> tjets = txp -> TruthJets;
        for (size_t j(0); j < tjets.size(); ++j){
           if (!tjets[j] -> is_b){continue;}
           b_ = (particle_template*)tjets[j]; 
           break; 
        }

        if (b_){
            std::vector<particle_template*> mtx; 
            this -> downcast(&tjets, &mtx); 
            mtx.push_back(nu_); 
            mtx.push_back(lep_);             

            particle_template* tj_top = nullptr; 
            this -> sum(&mtx, &tj_top); 
            truthjets_tops.push_back(tj_top); 

            b_truthjets.push_back(b_); 
        }

        b_ = nullptr; 
        std::vector<jet*> jets = txp -> Jets; 
        for (size_t j(0); j < jets.size(); ++j){
           if (!jets[j] -> is_b){continue;}
           b_ = (particle_template*)jets[j]; 
           break; 
        }

        if (b_){
            std::vector<particle_template*> mtx; 
            this -> downcast(&jets, &mtx); 
            mtx.push_back(nu_); 
            mtx.push_back(lep_);             

            particle_template* j_top = nullptr; 
            this -> sum(&mtx, &j_top); 
            jets_tops.push_back(j_top); 

            b_jets.push_back(b_); 
        }

        std::vector<particle_template*> reco_lep = this -> vectorize(&lep_ -> children); 
        if (!reco_lep.size() || !b_){continue;}
        lep_ = reco_lep[0]; 
        reco_leptons.push_back(lep_); 
        
        std::vector<particle_template*> mtk; 
        this -> downcast(&jets, &mtk); 
        mtk.push_back(nu_); 
        mtk.push_back(lep_);             

        particle_template* jl_top = nullptr; 
        this -> sum(&mtk, &jl_top); 
        ljets_tops.push_back(jl_top); 

        bosonx = (*nu_ + *lep_); 
        reco_bosons.push_back(bosonx);
    }
    
    pkl -> c1_reconstructed_children_nu = this -> build_neutrinos(
                &pkl -> truth_bquarks , &pkl -> truth_leptons, 
                &pkl -> truth_tops    , &pkl -> truth_bosons, 
                pkl -> met, pkl -> phi, &pkl -> truth_nus
    ); 

    pkl -> c2_reconstructed_children_nu = this -> build_neutrinos(
                &pkl -> truth_bquarks , &pkl -> truth_leptons, 
                nullptr, nullptr, 
                pkl -> met, pkl -> phi, &pkl -> truth_nus
    ); 

    if (truthjets_tops.size() == 2){
        pkl -> truth_jets_top.push_back(new tquark(&truthjets_tops[0] -> data));
        pkl -> truth_jets_top.push_back(new tquark(&truthjets_tops[1] -> data));

        pkl -> truth_bjets.push_back(new bquark(&b_truthjets[0] -> data)); 
        pkl -> truth_bjets.push_back(new bquark(&b_truthjets[1] -> data)); 

        pkl -> c1_reconstructed_truthjet_nu = this -> build_neutrinos(
                &pkl -> truth_bjets   , &pkl -> truth_leptons,  
                &pkl -> truth_jets_top, &pkl -> truth_bosons, 
                pkl -> met, pkl -> phi, &pkl -> truth_nus
        ); 

        pkl -> c2_reconstructed_truthjet_nu = this -> build_neutrinos(
                &pkl -> truth_bjets   , &pkl -> truth_leptons, 
                nullptr, nullptr, 
                pkl -> met, pkl -> phi, &pkl -> truth_nus
        ); 
    }

    if (jets_tops.size() == 2){
        pkl -> jets_top.push_back(new tquark(&jets_tops[0] -> data));
        pkl -> jets_top.push_back(new tquark(&jets_tops[1] -> data));

        pkl -> bjets.push_back(new bquark(&b_jets[0] -> data)); 
        pkl -> bjets.push_back(new bquark(&b_jets[1] -> data)); 

        pkl -> c1_reconstructed_jetchild_nu = this -> build_neutrinos(
                &pkl -> bjets   , &pkl -> truth_leptons,  
                &pkl -> jets_top, &pkl -> truth_bosons, 
                pkl -> met, pkl -> phi, &pkl -> truth_nus
        ); 

        pkl -> c2_reconstructed_jetchild_nu = this -> build_neutrinos(
                &pkl -> bjets   , &pkl -> truth_leptons, 
                nullptr, nullptr, 
                pkl -> met, pkl -> phi, &pkl -> truth_nus
        ); 
    }

    if (reco_bosons.size() == 2){
        pkl -> reco_bosons.push_back(new boson(&reco_bosons[0].data)); 
        pkl -> reco_bosons.push_back(new boson(&reco_bosons[1].data)); 

        pkl -> reco_leptons.push_back(new lepton(&reco_leptons[0] -> data)); 
        pkl -> reco_leptons.push_back(new lepton(&reco_leptons[1] -> data)); 

        pkl -> lepton_jets_top.push_back(new tquark(&ljets_tops[0] -> data)); 
        pkl -> lepton_jets_top.push_back(new tquark(&ljets_tops[1] -> data)); 

        pkl -> c1_reconstructed_jetlep_nu = this -> build_neutrinos(
                &pkl -> bjets          , &pkl -> reco_leptons,  
                &pkl -> lepton_jets_top, &pkl -> reco_bosons, 
                pkl -> met,  pkl -> phi, &pkl -> truth_nus
        ); 

        pkl -> c2_reconstructed_jetlep_nu = this -> build_neutrinos(
                &pkl -> bjets   , &pkl -> reco_leptons, 
                nullptr, nullptr, 
                pkl -> met, pkl -> phi, &pkl -> truth_nus
        ); 
    }
    return true; 
}




