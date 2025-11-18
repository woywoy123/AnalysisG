#include <conuix/conuix.h>
#include <conuix/conuic.h>

conuix::~conuix(){
    for (size_t x(0); x < this -> cnx -> size(); ++x){
        if (!this -> cnx -> at(x)){continue;}
        delete this -> cnx -> at(x); 
        (*this -> cnx)[x] = nullptr; 
    }
    delete this -> cnx; 
    this -> cnx = nullptr; 
}

conuix::conuix(nusol_t* parameters){
    this -> params = parameters; 
    this -> prefix = "Conuix"; 
    this -> params -> met_x = std::cos(this -> params -> phi) * this -> params -> met;  
    this -> params -> met_y = std::sin(this -> params -> phi) * this -> params -> met;
    size_t n = this -> params -> targets -> size();

    std::vector<particle_template*> jets = {}; 
    std::vector<particle_template*> leps = {};
    jets.reserve(n); 
    leps.reserve(n);

    for (size_t x(0); x < this -> params -> targets -> size(); ++x){
        particle_template* pn = this -> params -> targets -> at(x); 
        if (pn -> is_lep){leps.push_back(pn);}
        else {jets.push_back(pn);}
    }

    int idx = -1; 
    this -> cnx = new std::vector<conuic*>(jets.size() * leps.size(), nullptr);
    this -> params -> phys_pairs = new std::vector< std::pair<particle_template*, particle_template*> >(); 
    for (size_t i(0); i < jets.size(); ++i){
        for (size_t j(0); j < leps.size(); ++j){
            conuic* cx = new conuic(jets[i], leps[j]);
            bool        lx = cx -> converged; 
            long double lf = cx -> error; 
            if (lf > 10e-12 || !lx || !cx -> mass_line(this -> params -> mw, this -> params -> mt)){delete cx; continue;}
            (*this -> cnx)[++idx] = cx;  
            params -> phys_pairs -> push_back({jets[i], leps[j]}); 
        }
    }
    std::vector<conuic*>* cnv = new std::vector<conuic*>(idx+1, nullptr); idx = -1;  
    for (size_t x(0); x < this -> cnx -> size(); ++x){
        if (!this -> cnx -> at(x)){continue;}
        (*cnv)[++idx] = this -> cnx -> at(x); 
    }
    delete this -> cnx; this -> cnx = nullptr;
    this -> cnx = cnv; 
}

std::vector<particle_template*> conuix::nunu_make(){
    conuic* c = this -> cnx -> at(0); 
    matrix_t N = c -> Nmatrix(1, 1); 
    return {}; 
}



