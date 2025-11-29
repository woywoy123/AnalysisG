#include <reconstruction/nusol.h>
#include <ellipse/ellipse.h>
#include <conuix/conuix.h>

nusol::neutrino::~neutrino(){}

nusol::neutrino::neutrino(nuelx* nu, mtx* mx) : 
    particle_template(mx -> _m[0][0], mx -> _m[0][1], mx -> _m[0][2]) 
{
    this -> type = "nunu"; 
    this -> register_parent(nu -> l -> lnk); 
    this -> register_parent(nu -> b -> lnk); 
    this -> score = mx -> _m[0][3]; 
}

void nusol::neutrino::merge(neutrino* nx){
    std::map<std::string, particle_template*> pr = nx -> parents; 
    std::map<std::string, particle_template*>::iterator ipr = pr.begin();
    for (; ipr != pr.end(); ++ipr){this -> register_parent(ipr -> second);}
}


particle_template* nusol::neutrino::release(){
    particle_template* o = new particle_template(this -> px, this -> py, this -> pz); 
    o -> type = "nunu"; 
    std::map<std::string, particle_template*> pr = this -> parents; 
    std::map<std::string, particle_template*>::iterator ipr = pr.begin();
    for (; ipr != pr.end(); ++ipr){o -> register_parent(ipr -> second);}
    return o; 
}


nusol::nusol(nusol_t* parameters){
    this -> params = parameters; 
    this -> prefix = "NuSol"; 

    this -> params -> met_x = std::cos(this -> params -> phi) * this -> params -> met;  
    this -> params -> met_y = std::sin(this -> params -> phi) * this -> params -> met;
}

nusol::~nusol(){
    if (this -> D_nunu){delete this -> D_nunu;}
    if (this -> M_nunu){delete this -> M_nunu;}
}

std::vector<particle_template*> nusol::solve(){
    auto ch2 =[](neutrino* nx1, neutrino* nx2) -> double {
        double dx = (nx1 -> px - nx2 -> px) / (std::fabs(nx1 -> px) + std::fabs(nx2 -> px)); 
        double dy = (nx1 -> py - nx2 -> py) / (std::fabs(nx1 -> py) + std::fabs(nx2 -> py)); 
        double dz = (nx1 -> pz - nx2 -> pz) / (std::fabs(nx1 -> pz) + std::fabs(nx2 -> pz)); 
        return dx * dx + dy * dy + dz * dz; 
    }; 

    this -> M_nunu = new conuix( this -> params);    
    this -> D_nunu = new ellipse(this -> params);
    this -> D_nunu -> prepare(this -> params -> mt, this -> params -> mw, this -> M_nunu -> cnx); 
    this -> D_nunu -> solve(); 
    std::map<double, nunu_t>* slc = &this -> D_nunu -> solvs; 
    std::map<double, std::map<std::string, particle_template*>> vio_map;  
    std::map<std::string, std::map<std::string, particle_template*>> lep_jet; 

    std::vector<neutrino*> nu_ptrs; 
    nu_ptrs.reserve(slc -> size() * 2); 

    std::map<double, nunu_t>::iterator itr = slc -> begin(); 
    for (; itr != slc -> end(); ++itr){
        const nunu_t* nx = &itr -> second; 
        if (itr -> first > this -> params -> limit){continue;}

        if (nx -> nux1){
            std::string hl1 = nx -> nux1 -> l -> lnk -> hash; 
            neutrino*   nu1 = new neutrino(nx -> nux1, nx -> nu1); 
            nu_ptrs.push_back(nu1); 

            lep_jet[hl1][nu1 -> hash] = nu1; 
            vio_map[nu1 -> score][hl1] = nx -> nux1 -> l -> lnk; 
        }

        if (nx -> nux2){
            std::string hl2 = nx -> nux2 -> l -> lnk -> hash; 
            neutrino*   nu2 = new neutrino(nx -> nux2, nx -> nu2); 
            nu_ptrs.push_back(nu2); 

            lep_jet[hl2][nu2 -> hash] = nu2; 
            vio_map[nu2 -> score][hl2] = nx -> nux2 -> l -> lnk; 
        }
    }

    std::map<std::string, neutrino*> collect; 
    std::map<std::string, particle_template*>::iterator itl; 
    std::map<double, std::map<std::string, particle_template*>>::iterator itf; 

    for (itf = vio_map.begin(); itf != vio_map.end(); ++itf){
        for (itl = itf -> second.begin(); itl != itf -> second.end(); ++itl){
            neutrino* nxn = nullptr; 
            for (size_t x(0); x < nu_ptrs.size(); ++x){ 
                neutrino* nxp = nu_ptrs[x]; 
                if (!nxp){continue;}

                if (!lep_jet[itl -> first].count(nxp -> hash)){continue;}
                if (!nxn){nxn = nxp; nu_ptrs[x] = nullptr; continue;}
                if (nxn -> score < nxp -> score){continue;}
                double sm = ch2(nu_ptrs[x], nxn);
                nu_ptrs[x] = nxn; nxn = nxp; 
                if (sm > 1){continue;}
                nxn -> merge(nu_ptrs[x]); 
                delete nu_ptrs[x]; 
                nu_ptrs[x] = nullptr; 
            }
            if (!nxn){continue;}
            collect[itl -> first] = nxn; 
            lep_jet[itl -> first] = {}; 
        }
    }

    std::vector<particle_template*> out; 
    std::map<std::string, neutrino*>::iterator itc = collect.begin(); 
    for (; itc != collect.end(); ++itc){
        out.push_back(itc -> second -> release());
        delete itc -> second; 
    }
        
    for (size_t x(0); x < nu_ptrs.size(); ++x){
        if (!nu_ptrs[x]){continue;}
        delete nu_ptrs[x]; nu_ptrs[x] = nullptr; 
    }
    return out; 
}

