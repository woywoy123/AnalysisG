#include <bsm_4tops/event.h>
#include <pyc/pyc.h>
#include "met.h"

met::met(){this -> name = "met";}
met::~met(){this -> safe_delete(&this -> storage);}
selection_template* met::clone(){return new met();}
bool met::selection(event_template* ev){return true;}
void met::merge(selection_template* sl){
    auto lamb =[this](std::vector<angle_t>* data, std::string name){
        std::vector<std::vector<double>> pln_w1 = {}; 
        std::vector<std::vector<double>> pln_t1 = {}; 

        std::vector<std::vector<double>> pln_w2 = {}; 
        std::vector<std::vector<double>> pln_t2 = {}; 
        std::vector<std::vector<double>> pln_t1xt2 = {};

        std::vector<double> dR_nu_lep1 = {};
        std::vector<double> dR_nu_lep2 = {};
        std::vector<bool> has_null = {}; 
        for (size_t x(0); x < data -> size(); ++x){
            angle_t* xt = &data -> at(x); 
            pln_w1.push_back(xt -> pln_w1); 
            pln_w2.push_back(xt -> pln_w2); 

            pln_t1.push_back(xt -> pln_t1); 
            pln_t2.push_back(xt -> pln_t2); 
            pln_t1xt2.push_back(xt -> pln_t1xt2); 

            dR_nu_lep1.push_back(xt -> dR_nu_lep1); 
            dR_nu_lep2.push_back(xt -> dR_nu_lep2); 
            has_null.push_back(xt -> has_null); 
        }

        this -> write(&pln_w1   , name + "_plane_w1"); 
        this -> write(&pln_w2   , name + "_plane_w2"); 

        this -> write(&pln_t1   , name + "_plane_t1"); 
        this -> write(&pln_t2   , name + "_plane_t2"); 
        this -> write(&pln_t1xt2, name + "_plane_t1xt2"); 

        this -> write(&dR_nu_lep1, name + "_dR_nu_lep1"); 
        this -> write(&dR_nu_lep2, name + "_dR_nu_lep2"); 
        this -> write(&has_null  , name + "_has_null"); 
    }; 


    met* slt = (met*)sl;
    packet_t* pkl = &slt -> data; 
    
    
    this -> write(&pkl -> missing_evn_px, "missing_event_px"); 
    this -> write(&pkl -> missing_evn_py, "missing_event_py"); 

    this -> write(&pkl -> missing_det_px, "missing_detector_px"); 
    this -> write(&pkl -> missing_det_py, "missing_detector_py"); 
    this -> write(&pkl -> missing_det_pz, "missing_detector_pz"); 

    this -> write(&pkl -> missing_nus_px, "missing_neutrino_px"); 
    this -> write(&pkl -> missing_nus_py, "missing_neutrino_py"); 
    this -> write(&pkl -> missing_nus_pz, "missing_neutrino_pz"); 

    this -> write(&pkl -> num_neutrino    , "num_neutrino"); 
    this -> write(&pkl -> num_leptons     , "num_leptons"); 
    this -> write(&pkl -> num_leptons_reco, "num_leptons_reco"); 

    this -> write(&pkl -> top_index    , "top_index"); 
    this -> write(&pkl -> mass_tru_top , "mass_tru_top"); 
    this -> write(&pkl -> mass_tru_top3, "mass_tru_top_blnu"); 

    this -> write(&pkl -> tru_nu, "truth_neutrino", particle_enum::pmc); 
    this -> write(&pkl -> chi2_sols      , "chi2_solutions"); 
    this -> write(&pkl -> top_index_sols1, "top_index_solutions_1"); 
    this -> write(&pkl -> top_index_sols2, "top_index_solutions_2"); 

    this -> write(&pkl -> top_mass_sols1, "top_mass_solutions_1"); 
    this -> write(&pkl -> top_mass_sols2, "top_mass_solutions_2"); 

    this -> write(&pkl -> nu1, "reco_neutrino_1", particle_enum::pmc); 
    this -> write(&pkl -> nu2, "reco_neutrino_2", particle_enum::pmc); 

    lamb(&pkl -> agnR, "reco"); 
    lamb(&pkl -> agnT, "truth"); 
}

angle_t met::angle(
        particle_template* nu1, particle_template* l1, particle_template* b1, 
        particle_template* nu2, particle_template* l2, particle_template* b2, 
        std::vector<double> det_, std::vector<double> met_
){
    auto dot = [this](std::vector<double> pln1, std::vector<double> pln2) -> double{
        double nul = 0; 
        for (size_t x(0); x < 3; ++x){nul += pln1[x] * pln2[x];}
        return nul; 
    }; 

    auto minus = [this](std::vector<double> pln1, std::vector<double> pln2, double sign) -> std::vector<double>{
        std::vector<double> out = {}; 
        for (size_t x(0); x < 3; ++x){out.push_back(pln1[x] + sign * pln2[x]);}
        return out; 
    }; 


    auto sum = [this, minus](std::vector<double> pln1, std::vector<std::vector<double>> pln2, double sign) -> std::vector<double>{
        std::vector<double> out = pln1; 
        for (size_t x(0); x < pln2.size(); ++x){out = minus(out, pln2[x], sign);}
        return out; 
    }; 

    auto lamb =[this, dot](std::vector<double> pln1, std::vector<double> pln2) -> double{
        double m11 = std::pow(dot(pln1, pln1), 0.5); 
        double m22 = std::pow(dot(pln2, pln2), 0.5);
        double m12 = dot(pln1, pln2) / (m11 * m22); 
        return std::asin(m12)*(180.0/3.141592653589793238463); 
    }; 

    auto chx = [this](std::vector<double> pln) -> double {
        double o = 0; 
        for (size_t x(0); x < pln.size(); ++x){o += std::pow(pln[x]*0.001, 2);}
        return std::pow(o, 0.5); 
    }; 




    angle_t angl;
    if (!nu1 || !l1 || !b1 || !nu2 || !l2 || !b2){return angl;}
    std::vector<double> pmc_n1 = this -> pmc(nu1); 
    std::vector<double> pmc_b1 = this -> pmc(b1); 
    std::vector<double> pmc_l1 = this -> pmc(l1); 

    std::vector<double> pmc_n2 = this -> pmc(nu2); 
    std::vector<double> pmc_b2 = this -> pmc(b2); 
    std::vector<double> pmc_l2 = this -> pmc(l2); 

    std::vector<double> pln_w1 = sum(det_, {pmc_n2, pmc_n1}, 1); 
    std::vector<double> pln_w2 = minus(met_, pln_w1, +1); 

    std::vector<double> pln_t1 = minus(met_  , pln_w1, +1); 
    std::vector<double> pln_t2 = minus(pmc_n2, pmc_n1, -1); 

    double dz = std::abs(det_[2] - (pmc_b1[2] + pmc_l1[2] + pmc_n1[2] + pmc_b2[2] + pmc_l2[2] + pmc_n2[2]))*0.001;
    double dy = std::abs(met_[1] + det_[1] + (pmc_n1[1] + pmc_n2[1]))*0.001; 
    double dx = std::abs(met_[0] + det_[0] + (pmc_n1[0] + pmc_n2[0]))*0.001; 
    double r  = std::pow(dx*dy*dz, 0.3); 
    angl.pln_w1 = {};  
    angl.pln_t1 = {};
    angl.pln_w2 = {}; 
    angl.pln_t2 = {}; 
    angl.pln_t1xt2  = this -> cross(pln_t2, det_, true); 
    angl.dR_nu_lep1 = r; 
    angl.dR_nu_lep2 = b1 -> DeltaR(l1) + b2 -> DeltaR(l2); 
    angl.has_null = false;
    return angl;  
}

bool met::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    std::vector<particle_template*> dleps = {}; 
    merge_data(&dleps, &evn -> Electrons); 
    merge_data(&dleps, &evn -> Muons); 

    std::vector<particle_template*> nus = {}; 
    for (size_t x(0); x < evn -> Children.size(); ++x){
        if (!evn -> Children[x] -> is_nu){continue;}
        nus.push_back(evn -> Children[x]);        
    }

    this -> data.tru_nu = nus; 
    this -> data.num_neutrino = nus.size(); 
    this -> data.num_leptons  = dleps.size(); 

    this -> data.missing_nus_px = this -> cart_px(nus); 
    this -> data.missing_nus_py = this -> cart_py(nus); 
    this -> data.missing_nus_pz = this -> cart_pz(nus); 

    this -> data.missing_det_px = this -> cart_px(evn -> DetectorObjects); 
    this -> data.missing_det_py = this -> cart_py(evn -> DetectorObjects); 
    this -> data.missing_det_pz = this -> cart_pz(evn -> DetectorObjects); 

    this -> data.missing_evn_px = this -> cart_px(evn -> met, evn -> phi); 
    this -> data.missing_evn_py = this -> cart_py(evn -> met, evn -> phi); 

    std::vector<double> det_ = {this -> cart_px(evn -> DetectorObjects), this -> cart_py(evn -> DetectorObjects), this -> cart_pz(evn -> DetectorObjects)}; 
    std::vector<double> met_ = {this -> cart_px(evn -> met, evn -> phi), this -> cart_py(evn -> met, evn -> phi), 0}; 

    double mx = det_[0] + met_[0]; 
    double my = det_[1] + met_[1]; 
    double metx_ = std::pow(mx*mx + my*my, 0.5); 
    double phix_ = std::atan2(my, mx); 

    std::map<int, std::map<std::string, particle_template*>> tops_ = this -> match_tops(evn -> Tops, dleps); 
    std::tuple<std::vector<neutrino*>, std::vector<neutrino*>> nux = this -> reconstruction(evn -> DetectorObjects, metx_, phix_); 
    std::vector<neutrino*> nu1 = std::get<0>(nux); 
    std::vector<neutrino*> nu2 = std::get<1>(nux); 

    std::map<int, std::map<int, std::tuple<particle_template*, neutrino*>>> chi2_map_n1; 
    std::map<int, std::map<int, std::tuple<particle_template*, neutrino*>>> chi2_map_n2; 

    std::map<int, particle_template*> leptons; 
    std::map<int, particle_template*> quarks; 

    std::map<int, std::map<std::string, particle_template*>>::iterator itt;

    for (itt = tops_.begin(); itt != tops_.end(); ++itt){
        if (!itt -> second.size()){continue;}
        std::vector<particle_template*> vc = this -> vectorize(&itt -> second); 
        
        particle_template* nut = nullptr; 
        particle_template* lep = nullptr; 
        particle_template* bjt = nullptr; 

        for (size_t x(0); x < vc.size(); ++x){
            if (vc[x] -> is_nu){ nut = vc[x]; continue;}
            if (vc[x] -> is_lep){lep = vc[x]; continue;}
            if (vc[x] -> is_b){  bjt = vc[x];}
        }
        double mass = 0; 
        if (nut && lep && bjt){
            std::vector<particle_template*> nvx = {nut, lep, bjt};  
            mass = this -> sum(&nvx); 
        }
        leptons[itt -> first] = lep;
        quarks[itt -> first]  = bjt; 
        this -> data.mass_tru_top.push_back(this -> sum(&vc)); 
        this -> data.mass_tru_top3.push_back(mass); 
        this -> data.top_index.push_back(itt -> first);  
        for (size_t x(0); x < nu1.size(); ++x){chi2_map_n1[x][itt -> first] = {nut, nu1[x]};}
        for (size_t x(0); x < nu2.size(); ++x){chi2_map_n2[x][itt -> first] = {nut, nu2[x]};}
    }

    std::map<double, std::map<int, std::tuple<particle_template*, neutrino*>>> sols_map; 
    for (size_t x(0); x < nu1.size(); ++x){
        std::map<int, std::tuple<particle_template*, neutrino*>>::iterator itn1; 
        std::map<int, std::tuple<particle_template*, neutrino*>>::iterator itn2; 
        for (itn1 = chi2_map_n1[x].begin(); itn1 != chi2_map_n1[x].end(); ++itn1){
            for (itn2 = chi2_map_n2[x].begin(); itn2 != chi2_map_n2[x].end(); ++itn2){
                int top1 = itn1 -> first; int top2 = itn2 -> first;
                if (top1 >= top2){continue;}

                neutrino* n1r = std::get<1>(itn1 -> second); 
                neutrino* n2r = std::get<1>(itn2 -> second); 

                particle_template* n1r_ = (particle_template*)n1r; 
                particle_template* n2r_ = (particle_template*)n2r; 

                particle_template* n1t = std::get<0>(itn1 -> second); 
                particle_template* n2t = std::get<0>(itn2 -> second); 

                double ch1 = this -> chi2(n1t, n1r_); 
                double ch2 = this -> chi2(n2t, n2r_); 

                double ch1s = this -> chi2(n1t, n2r_); 
                double ch2s = this -> chi2(n2t, n1r_); 

                if ((ch1 + ch2) > (ch1s + ch2s)){
                    n1r_ = (particle_template*)std::get<1>(itn2 -> second);  
                    n2r_ = (particle_template*)std::get<1>(itn1 -> second); 
                }


                if (sols_map[ch1 + ch2].count(top1) && sols_map[ch1 + ch2].count(top2)){continue;}
                this -> data.nu1.push_back(n1r); 
                this -> data.nu2.push_back(n2r); 
                this -> data.top_index_sols1.push_back(top1); 
                this -> data.top_index_sols2.push_back(top2); 

                this -> data.chi2_sols.push_back(ch1 + ch2);
                this -> data.agnR.push_back(this -> angle(n1r_, n1r -> lepton, n1r -> bquark, n2r_, n2r -> lepton, n2r -> bquark, det_, met_)); 
                this -> data.agnT.push_back(this -> angle(n1t, leptons[top1], quarks[top1], n2t, leptons[top2], quarks[top2], det_, met_)); 

                angle_t ag = this -> angle(n1t, leptons[top1], quarks[top1], n2t, leptons[top2], quarks[top2], det_, met_); 
                angle_t ar = this -> angle(n1r_, n1r -> lepton, n1r -> bquark, n2r_, n2r -> lepton, n2r -> bquark, det_, met_); 
                //if (!ag.has_null){
                //    std::cout << ag.dR_nu_lep1 << " " << ag.dR_nu_lep2 <<  " | "; 
                //    std::cout << ar.dR_nu_lep1 << " " << ar.dR_nu_lep2 <<  " | "; 
                //    std::cout << std::pow(ch1 + ch2, 0.5) << std::endl;
                //}
                sols_map[ag.dR_nu_lep1][top1] = {n1t, n1r};
                sols_map[ag.dR_nu_lep1][top2] = {n2t, n2r};
                std::vector<particle_template*> tp = {n1r_, n1r -> lepton, n1r -> bquark}; 
                this -> data.top_mass_sols1.push_back(this -> sum(&tp)); 
                tp = {n2r_, n2r -> lepton, n2r -> bquark}; 
                this -> data.top_mass_sols2.push_back(this -> sum(&tp)); 
            }
        }
    }
    return true; 
}

