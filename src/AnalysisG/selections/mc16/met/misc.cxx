#include <bsm_4tops/event.h>
#include <pyc/pyc.h>
#include "met.h"

std::vector<double> met::pmc(particle_template* p1){return {p1 -> px, p1 -> py, p1 -> pz, p1 -> e};}

std::vector<double> met::cross(std::vector<double> v1, std::vector<double> v2, bool norm){
    double x1 = v1[0]; double y1 = v1[1]; double z1 = v1[2]; 
    double x2 = v2[0]; double y2 = v2[1]; double z2 = v2[2]; 

    double xi = x1 * z2 - z1 * y2; 
    double yi = z1 * x2 - x1 * z2; 
    double zi = x1 * y2 - y1 * x2; 
    double di = std::pow(xi*xi + yi*yi + zi*zi, 0.5); 
//    if (norm){xi = xi / di; yi = yi / di; zi = zi / di;}
    return {xi, yi, zi, di}; 
}

double met::chi2(particle_template* nut, particle_template* nur){
    std::vector<double> pnut = this -> pmc(nut);  
    std::vector<double> pnur = this -> pmc(nur); 
    double out = 0; 
    for (size_t x(0); x < pnut.size(); ++x){out += std::pow((pnut[x] - pnur[x])*0.001, 2);}
    return out; 
}

std::tuple<std::vector<neutrino*>, std::vector<neutrino*>> met::reconstruction(std::vector<particle_template*> nodes, double met_, double phi_){
    std::vector<double> metv = std::vector<double>({met_}); 
    std::vector<double> phiv = std::vector<double>({phi_});
    std::vector<std::vector<particle_template*>> prt = {nodes}; 
    std::vector<std::pair<neutrino*, neutrino*>> nux = pyc::nusol::combinatorial(metv, phiv, prt, "cuda:0", this -> masstop, this -> massw, this -> distance, this -> perturb, this -> steps); 
    std::vector<neutrino*> v1, v2;
    for (size_t x(0); x < nux.size(); ++x){
        neutrino* nu1 = std::get<0>(nux[x]);
        neutrino* nu2 = std::get<1>(nux[x]); 
        this -> storage.push_back(nu1); 
        this -> storage.push_back(nu2); 
        std::vector<neutrino*> v1_, v2_;
        merge_data(&v1_, &nu1 -> alternatives); 
        merge_data(&v2_, &nu2 -> alternatives); 
        v1_.push_back(nu1); v2_.push_back(nu2); 
        for (size_t i(0); i < v1_.size(); ++i){
            neutrino* n1 = v1_[i]; neutrino* n2 = v2_[i]; 
            particle_template* l1 = n1 -> lepton;
            particle_template* b1 = n1 -> bquark; 
            particle_template* l2 = n2 -> lepton;
            particle_template* b2 = n2 -> bquark; 

            double l1b1 = l1 -> DeltaR(b1); 
            double l1b2 = l1 -> DeltaR(b2); 

            double l2b1 = l2 -> DeltaR(b1);
            double l2b2 = l2 -> DeltaR(b2); 
            if (l1b1 > l1b2 || l2b2 > l2b1){continue;}
            v1.push_back(n1); v2.push_back(n2); 
        }

    }
    return {v1, v2};     
}

std::vector<double> met::sum_cart(std::vector<particle_template*> ptx){
    std::vector<double> out = {0, 0, 0, 0}; 
    for (size_t x(0); x < ptx.size(); ++x){
        std::vector<double> pc = this -> pmc(ptx[x]);
        for (size_t y(0); y < out.size(); ++y){out[y] += pc[y];}
    }
    return out; 
}

double met::cart_px(std::vector<particle_template*> prt){return this -> sum_cart(prt)[0];}
double met::cart_py(std::vector<particle_template*> prt){return this -> sum_cart(prt)[1];}
double met::cart_pz(std::vector<particle_template*> prt){return this -> sum_cart(prt)[2];}

double met::cart_px(double pt, double phi){return pt*std::cos(phi);}
double met::cart_py(double pt, double phi){return pt*std::sin(phi);}


std::map<int, std::map<std::string, particle_template*>> met::match_tops(
        std::vector<particle_template*> tops, std::vector<particle_template*> dleps
){
    std::map<int, std::map<std::string, particle_template*>> out;  

    std::vector<top*> tops_; 
    this -> upcast(&tops, &tops_); 
    for (size_t x(0); x < tops_.size(); ++x){
        top* tpx = tops_[x]; 
        int top_index = tpx -> index; 

        std::vector<particle_template*> ch = this -> vectorize(&tpx -> children); 
        for (size_t c(0); c < ch.size(); ++c){
            if (!ch[c] -> is_nu){continue;}
            out[top_index][ch[c] -> hash] = ch[c]; 
            break;
        }
        std::vector<particle_template*> jets; 
        this -> upcast(&tpx -> Jets, &jets); 
        this -> add(&out[top_index], &jets); 

        particle_template* lepton = nullptr; 
        for (size_t c(0); c < dleps.size(); ++c){
            std::map<std::string, particle_template*> pr = dleps[c] -> parents; 
            for (size_t j(0); j < ch.size(); ++j){
                if (!ch[j] -> is_lep){continue;}
                if (!pr.count(ch[j] -> hash)){continue;}
                lepton = dleps[c]; break; 
            }
            if (!lepton){continue;}
            this -> data.num_leptons_reco += 1; 
            out[top_index][lepton -> hash] = lepton; 
            break; 
        }
        if (lepton){continue;}
        out[top_index].clear();
    }
    return out; 
}
