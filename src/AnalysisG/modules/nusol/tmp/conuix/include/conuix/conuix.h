#ifndef H_CONICS
#define H_CONICS

#include <templates/particle_template.h>

// forward declare the main interface struct.
struct nusol_t; 

class conics
{
    public:
        conics(nusol_t* parameters); 
        ~conics(); 

    private: 
        nusol_t* params = nullptr; 
}; 


#endif


//void nunu_solver::prepare(double mt, double mw){
//// neutrinos::::: 
////1
////2
////3
////------------ leptons ------------- 
////hash of lepton: 0x63854275c39e50bf top-index: 3
////hash of lepton: 0xee7e6b73d1f840e5 top-index: 2
////hash of lepton: 0xff77ec7543171a72 top-index: 1
////------------- b jets ------------- 
////hash of b-jet: 0x5bcc2083f142879b top-index:0
////hash of b-jet: 0x5a058b7a2b3c3150 top-index:0
////hash of b-jet: 0xca321d17a69c05f2 top-index:0
////hash of b-jet: 0x5bcc2083f142879b top-index:1 << 0xff77ec7543171a72
////hash of b-jet: 0xd6392d4acc2883b3 top-index:2 << 0xee7e6b73d1f840e5
////hash of b-jet: 0xcc1cd25d860e7321 top-index:3 << 0x63854275c39e50bf
//
//
//
////hash of lepton: 0x1a312f109a2f07a7 top-index: 2
////hash of lepton: 0xd84f33c2c0f488c4 top-index: 3
////hash of lepton: 0x5664ddcca665e60d top-index: 1
////------------- b jets ------------- 
////hash of b-jet: 0x284baa74142be1e1 top-index:0
////hash of b-jet: 0xcb23070ab42b4a1d top-index:0
////hash of b-jet: 0x5b6f2362aae20d2c top-index:1 << 0x5664ddcca665e60d
////hash of b-jet: 0x5b6f2362aae20d2c top-index:2 << 0x1a312f109a2f07a7
////hash of b-jet: 0x7cb1967b4c4ebb3f top-index:3 << 0xd84f33c2c0f488c4
//
//    std::string hax = "0xff77ec7543171a72"; 
//    std::vector<std::string> pairs; 
//    for (int l(0); l < this -> n_lp; ++l){
//        for (int b(0); b < this -> n_bs; ++b){
//            nusol*     nl = new nusol(this -> bquarks[b], this -> leptons[l], mw, mt); 
//            multisol* nux = new multisol(this -> bquarks[b], this -> leptons[l]); 
//            
//            // check if combination is actually physically viable:
//            // validates that the characterstic equation has real roots for t = 0 and z = 1.
//            // why t = 0; sinh(0) = 0!
//            if (nux -> eigenvalues(0, 1)){delete nl; delete nux; continue;}
//
//            // check if given combination has a special case where:
//            // dP(lambda_c, t*)/dt = 0 and P(lambda_c, t*) = 0. 
//            //double t1 = nux -> dp_dt(); 
//            //vec3 rel, img; 
//            //nux -> eigenvalues(t1, 1, &rel, &img); 
//            //if (std::fabs(img.x) > 0){delete nl; delete nux; continue;}
//
//            this -> engines.push_back(nl); 
//            this -> nuxs.push_back(nux); 
//            if (hax == std::string(this -> leptons[l] -> hash)){hax = "";}
//
//            std::string hb = std::string(this -> bquarks[b] -> hash); 
//            std::string hl = std::string(this -> leptons[l] -> hash); 
//
//            std::string b1 = "0x5bcc2083f142879b";
//            std::string l1 = "0xff77ec7543171a72"; 
//
//            std::string b2 = "0xd6392d4acc2883b3";
//            std::string l2 = "0xee7e6b73d1f840e5"; 
//
//            std::string b3 = "0xcc1cd25d860e7321";
//            std::string l3 = "0x63854275c39e50bf"; 
//
//            
//            std::string k1 = "0x5b6f2362aae20d2c";
//            std::string f1 = "0x5664ddcca665e60d"; 
//            std::string k2 = "0x5b6f2362aae20d2c";
//            std::string f2 = "0x1a312f109a2f07a7"; 
//            std::string k3 = "0x7cb1967b4c4ebb3f";
//            std::string f3 = "0xd84f33c2c0f488c4"; 
//
//            hb += (hb == b1 && hl == l1) ? " (truth) " : ""; 
//            hb += (hb == b2 && hl == l2) ? " (truth) " : ""; 
//            hb += (hb == b3 && hl == l3) ? " (truth) " : ""; 
//
//
//            hb += (hb == k1 && hl == f1) ? " (truth) " : ""; 
//            hb += (hb == k2 && hl == f2) ? " (truth) " : ""; 
//            hb += (hb == k3 && hl == f3) ? " (truth) " : ""; 
//
//            pairs.push_back("l (hash): " + hl + " b (hash): " + hb); 
//        }
//    }
//  
//
//    vec3 met{this -> _metx, this -> _mety, 0}; 
//    odeRK* slx = new odeRK(&this -> nuxs, met, 10000000000, 0.0001); 
//    slx -> solve(); 
//
//    abort(); 
//
//
//
//
//
//
//    int lx = -1; 
//    size_t le = this -> engines.size();   
//    for (size_t x(0); x < le; ++x){
//        for (size_t y(0); y < le; ++y){
//            if (y == x){continue;}
//             
//            multisol* n1 = this -> nuxs[x];
//            multisol* n2 = this -> nuxs[y]; 
//            std::cout << "----------------------" << std::endl;
//            
//            std::cout << pairs[x] << " " << pairs[y] << std::endl; 
//            double t1 = n1 -> dp_dt(); 
//            double t2 = n2 -> dp_dt(); 
//
//            n1 -> H(t1, 1).print();
//            n2 -> H(t2, 1).print(); 
//            geo_t gx = n1 -> intersection(n2, t1, 1, t2, 1); 
//
//            //double d2 = gx.d.mag2(); 
//            //double dx = (vx - gx.r0).dot(gx.d); 
//            //double s  = dx / d2; 
//            //std::cout << s << std::endl;
//
//
//            std::tuple<nusol*, nusol*> px;
//            px = std::tuple<nusol*, nusol*>(this -> engines[x], this -> engines[y]); 
//            this -> pairings[++lx] = px; 
//        }
//    } 
//    if (hax.size()){return;}
//    abort(); 
//}


