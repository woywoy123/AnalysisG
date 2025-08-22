#include <nunu/nunu.h>


nunu_t::nunu_t(){}
nunu_t::~nunu_t(){
    if (this -> nu1){delete this -> nu1;}
    if (this -> nu2){delete this -> nu2;}
    if (this -> agl){delete this -> agl;}
}

nunu_solver::nunu_solver(std::vector<particle_template*>* targets, double met, double phi){
    this -> _metx = std::cos(phi) * met;  
    this -> _mety = std::sin(phi) * met; 
  
    for (size_t x(0); x < targets -> size(); ++x){
        if (!targets -> at(x) -> is_lep){continue;}
        this -> leptons.push_back(targets -> at(x));
    }

    for (size_t x(0); x < targets -> size(); ++x){
        if (targets -> at(x) -> is_lep){continue;}
        this -> bquarks.push_back(targets -> at(x));
    }

    this -> n_lp = this -> leptons.size();
    this -> n_bs = this -> bquarks.size();
    this -> engines.reserve(this -> n_lp*this -> n_bs); 
}

void nunu_solver::prepare(double mt, double mw){
// neutrinos::::: 
//1
//2
//3
//------------ leptons ------------- 
//hash of lepton: 0x63854275c39e50bf top-index: 3
//hash of lepton: 0xee7e6b73d1f840e5 top-index: 2
//hash of lepton: 0xff77ec7543171a72 top-index: 1
//------------- b jets ------------- 
//hash of b-jet: 0x5bcc2083f142879b top-index:0
//hash of b-jet: 0x5a058b7a2b3c3150 top-index:0
//hash of b-jet: 0xca321d17a69c05f2 top-index:0
//hash of b-jet: 0x5bcc2083f142879b top-index:1 << 0xff77ec7543171a72
//hash of b-jet: 0xd6392d4acc2883b3 top-index:2 << 0xee7e6b73d1f840e5
//hash of b-jet: 0xcc1cd25d860e7321 top-index:3 << 0x63854275c39e50bf



//hash of lepton: 0x1a312f109a2f07a7 top-index: 2
//hash of lepton: 0xd84f33c2c0f488c4 top-index: 3
//hash of lepton: 0x5664ddcca665e60d top-index: 1
//------------- b jets ------------- 
//hash of b-jet: 0x284baa74142be1e1 top-index:0
//hash of b-jet: 0xcb23070ab42b4a1d top-index:0
//hash of b-jet: 0x5b6f2362aae20d2c top-index:1 << 0x5664ddcca665e60d
//hash of b-jet: 0x5b6f2362aae20d2c top-index:2 << 0x1a312f109a2f07a7
//hash of b-jet: 0x7cb1967b4c4ebb3f top-index:3 << 0xd84f33c2c0f488c4

    std::string hax = "0xff77ec7543171a72"; 
    std::vector<std::string> pairs; 
    for (int l(0); l < this -> n_lp; ++l){
        for (int b(0); b < this -> n_bs; ++b){
            nusol*     nl = new nusol(this -> bquarks[b], this -> leptons[l], mw, mt); 
            multisol* nux = new multisol(this -> bquarks[b], this -> leptons[l]); 
            
            // check if combination is actually physically viable:
            // validates that the characterstic equation has real roots for t = 0 and z = 1.
            // why t = 0; sinh(0) = 0!
            if (nux -> eigenvalues(0, 1)){delete nl; delete nux; continue;}

            // check if given combination has a special case where:
            // dP(lambda_c, t*)/dt = 0 and P(lambda_c, t*) = 0. 
            //double t1 = nux -> dp_dt(); 
            //vec3 rel, img; 
            //nux -> eigenvalues(t1, 1, &rel, &img); 
            //if (std::fabs(img.x) > 0){delete nl; delete nux; continue;}

            this -> engines.push_back(nl); 
            this -> nuxs.push_back(nux); 
            if (hax == std::string(this -> leptons[l] -> hash)){hax = "";}

            std::string hb = std::string(this -> bquarks[b] -> hash); 
            std::string hl = std::string(this -> leptons[l] -> hash); 

            std::string b1 = "0x5bcc2083f142879b";
            std::string l1 = "0xff77ec7543171a72"; 

            std::string b2 = "0xd6392d4acc2883b3";
            std::string l2 = "0xee7e6b73d1f840e5"; 

            std::string b3 = "0xcc1cd25d860e7321";
            std::string l3 = "0x63854275c39e50bf"; 

            
            std::string k1 = "0x5b6f2362aae20d2c";
            std::string f1 = "0x5664ddcca665e60d"; 
            std::string k2 = "0x5b6f2362aae20d2c";
            std::string f2 = "0x1a312f109a2f07a7"; 
            std::string k3 = "0x7cb1967b4c4ebb3f";
            std::string f3 = "0xd84f33c2c0f488c4"; 

            hb += (hb == b1 && hl == l1) ? " (truth) " : ""; 
            hb += (hb == b2 && hl == l2) ? " (truth) " : ""; 
            hb += (hb == b3 && hl == l3) ? " (truth) " : ""; 


            hb += (hb == k1 && hl == f1) ? " (truth) " : ""; 
            hb += (hb == k2 && hl == f2) ? " (truth) " : ""; 
            hb += (hb == k3 && hl == f3) ? " (truth) " : ""; 

            pairs.push_back("l (hash): " + hl + " b (hash): " + hb); 
        }
    }
  

    vec3 met{this -> _metx, this -> _mety, 0}; 
    odeRK* slx = new odeRK(&this -> nuxs, met, 10000000000, 0.0001); 
    slx -> solve(); 

    abort(); 






    int lx = -1; 
    size_t le = this -> engines.size();   
    for (size_t x(0); x < le; ++x){
        for (size_t y(0); y < le; ++y){
            if (y == x){continue;}
             
            multisol* n1 = this -> nuxs[x];
            multisol* n2 = this -> nuxs[y]; 
            std::cout << "----------------------" << std::endl;
            
            std::cout << pairs[x] << " " << pairs[y] << std::endl; 
            double t1 = n1 -> dp_dt(); 
            double t2 = n2 -> dp_dt(); 

            n1 -> H(t1, 1).print();
            n2 -> H(t2, 1).print(); 
            geo_t gx = n1 -> intersection(n2, t1, 1, t2, 1); 

            //double d2 = gx.d.mag2(); 
            //double dx = (vx - gx.r0).dot(gx.d); 
            //double s  = dx / d2; 
            //std::cout << s << std::endl;


            std::tuple<nusol*, nusol*> px;
            px = std::tuple<nusol*, nusol*>(this -> engines[x], this -> engines[y]); 
            this -> pairings[++lx] = px; 
        }
    } 
    if (hax.size()){return;}
    abort(); 
}

void nunu_solver::solve(){
    auto lamb1 =[this](nusol* nux1, nusol* nux2, double mt_, double mw_, double* dz) -> bool{
        if (mt_ < 0 || mw_ < 0){return false;}
        double z = nux1 -> Z2(); 
        nux1 -> update(mt_, mw_); 
        this -> generate(nux1, nux2); 
        bool r = fabs(nux1 -> Z2() - z) < *dz; 
        *dz = (r || *dz < 0) ? fabs(nux1 -> Z2() - z) : *dz; 
        return r; 
    }; 

    auto lamb2 =[this](nusol* nux1, nusol* nux2, double mt_, double mw_, double* dz) -> bool{
        if (mt_ < 0 || mw_ < 0){return false;}
        double z = nux2 -> Z2(); 
        nux2 -> update(mt_, mw_); 
        this -> generate(nux1, nux2); 
        bool r = fabs(nux2 -> Z2() - z) < *dz; 
        *dz = (r || *dz < 0) ? fabs(nux2 -> Z2() - z) : *dz; 
        return r;
    }; 

    std::map<int, std::tuple<nusol*, nusol*>>::iterator itr = this -> pairings.begin(); 
    for (; itr != this -> pairings.end(); ++itr){
        std::tuple<nusol*, nusol*> px = itr -> second;
        nusol* nx1 = std::get<0>(px); 
        nusol* nx2 = std::get<1>(px);

        int x = 0; 
        double z1 = -1; 
        double z2 = -1; 
        double mt1(0), mt2(0), mw1(0), mw2(0); 
        while (true){
            size_t si = this -> solvs.size();
            double mw1_1, mw1_2, mt1_1, mt1_2, mt1_3, mt1_4; 
            nx1 -> Z_mW(&mw1_1, &mw1_2);
            nx1 -> Z_mT(&mt1_1, &mt1_2, mw1_1); 
            nx1 -> Z_mT(&mt1_3, &mt1_4, mw1_2); 

            double mw2_1, mw2_2, mt2_1, mt2_2, mt2_3, mt2_4; 
            nx2 -> Z_mW(&mw2_1, &mw2_2); 
            nx2 -> Z_mT(&mt2_1, &mt2_2, mw2_1); 
            nx2 -> Z_mT(&mt2_3, &mt2_4, mw2_2); 
           
            this -> generate(nx1, nx2); 
            if (lamb1(nx1, nx2, mw1_1, mt1_1, &z1)){mt1 = mt1_1; mw1 = mw1_1;}
            if (lamb1(nx1, nx2, mw1_2, mt1_1, &z1)){mt1 = mt1_1; mw1 = mw1_2;}
            if (lamb1(nx1, nx2, mw1_1, mt1_2, &z1)){mt1 = mt1_2; mw1 = mw1_1;} 
            if (lamb1(nx1, nx2, mw1_2, mt1_2, &z1)){mt1 = mt1_2; mw1 = mw1_2;}
            if (lamb1(nx1, nx2, mw1_1, mt1_3, &z1)){mt1 = mt1_3; mw1 = mw1_1;} 
            if (lamb1(nx1, nx2, mw1_2, mt1_3, &z1)){mt1 = mt1_3; mw1 = mw1_2;} 
            if (lamb1(nx1, nx2, mw1_1, mt1_4, &z1)){mt1 = mt1_4; mw1 = mw1_1;} 
            if (lamb1(nx1, nx2, mw1_2, mt1_4, &z1)){mt1 = mt1_4; mw1 = mw1_2;} 

            if (lamb2(nx1, nx2, mw2_1, mt2_1, &z2)){mt2 = mt2_1; mw2 = mw2_1;}
            if (lamb2(nx1, nx2, mw2_2, mt2_1, &z2)){mt2 = mt2_1; mw2 = mw2_2;}
            if (lamb2(nx1, nx2, mw2_1, mt2_2, &z2)){mt2 = mt2_2; mw2 = mw2_1;}
            if (lamb2(nx1, nx2, mw2_2, mt2_2, &z2)){mt2 = mt2_2; mw2 = mw2_2;}
            if (lamb2(nx1, nx2, mw2_1, mt2_3, &z2)){mt2 = mt2_3; mw2 = mw2_1;}
            if (lamb2(nx1, nx2, mw2_2, mt2_3, &z2)){mt2 = mt2_3; mw2 = mw2_2;}
            if (lamb2(nx1, nx2, mw2_1, mt2_4, &z2)){mt2 = mt2_4; mw2 = mw2_1;}
            if (lamb2(nx1, nx2, mw2_2, mt2_4, &z2)){mt2 = mt2_4; mw2 = mw2_2;}

            nx1 -> update(mt1, mw1);   
            nx2 -> update(mt2, mw2);   
            ++x; 
            if (si != this -> solvs.size() || x < 10){continue;}
            break;
        }
    }
}

void nunu_solver::nunu_make(particle_template** nu1, particle_template** nu2, double limit){
    if (!this -> solvs.size()){return;}
    std::map<double, nunu_t>::iterator itr = this -> solvs.begin(); 
//    if (itr -> first > 100){return;}
    std::cout << itr -> first << std::endl;
    nunu_t* nx = &itr -> second; 
    *nu1 = new particle_template(nx -> nu1 -> _m[0][0], nx -> nu1 -> _m[0][1], nx -> nu1 -> _m[0][2]); 
    *nu2 = new particle_template(nx -> nu2 -> _m[0][0], nx -> nu2 -> _m[0][1], nx -> nu2 -> _m[0][2]); 
    (*nu1) -> type = "nunu"; 
    (*nu2) -> type = "nunu"; 
    
    (*nu1) -> register_parent(nx -> nux1 -> l -> lnk); 
    (*nu2) -> register_parent(nx -> nux2 -> l -> lnk); 
    (*nu1) -> register_parent(nx -> nux1 -> b -> lnk); 
    (*nu2) -> register_parent(nx -> nux2 -> b -> lnk); 
}


nunu_solver::~nunu_solver(){
    for (size_t x(0); x < this -> engines.size(); ++x){delete this -> engines[x];}
    for (size_t x(0); x < this -> nuxs.size(); ++x){delete this -> nuxs[x];}
    this -> solvs.clear(); 
}

int nunu_solver::generate(nusol* nu1, nusol* nu2){
    this -> flush(); 
    this -> p_nu1 = nu1; this -> p_nu2 = nu2; 
    mtx* vi = nullptr; mtx* vi_ = nullptr;
    mtx* vr = nullptr; mtx* vr_ = nullptr;

    int n_pts = this -> intersection(&vi, &vi_); 
    int n_rts = this -> angle_cross( &vr, &vr_); 
    this -> m_agl = new mtx(3, n_pts + n_rts); 

    mtx* v1 = (vi  && vr ) ? vi  -> cat(vr ) : nullptr;
    mtx* v2 = (vi_ && vr_) ? vi_ -> cat(vr_) : nullptr; 
    
    if (vi && vr){
        delete  vi; delete vr; 
        delete vi_; delete vr_; 
        this -> make_neutrinos(v1, v2);
        delete v1; delete v2; 
        return this -> m_lx; 
    } 

    v1 = (!vi  && vr ) ? vr  : v1; 
    v2 = (!vi_ && vr_) ? vr_ : v2; 

    v1 = (vi  && !vr ) ? vi  : v1; 
    v2 = (vi_ && !vr_) ? vi_ : v2; 
    this -> make_neutrinos(v1, v2);
    delete v1; 
    delete v2; 
    return this -> m_lx; 
}


void nunu_solver::flush(){
    if (this -> m_nu1){delete this -> m_nu1;}
    if (this -> m_nu2){delete this -> m_nu2;} 
    if (this -> m_agl){delete this -> m_agl;}
    this -> m_nu1 = nullptr; 
    this -> m_nu2 = nullptr; 
    this -> m_agl = nullptr; 
    this -> m_lx   = 0; 
    this -> m_bst  = 0; 
}

int nunu_solver::intersection(mtx** v, mtx** v_){
    auto safe_del = [this](mtx** val) -> void{
        if (!(*val)){return;}
        delete (*val); (*val) = nullptr; 
    }; 

    auto make_nu = [this](mtx* S_, mtx* vx) -> mtx*{
        mtx vxt = S_ -> dot(vx -> T()).T();
        return new mtx(vxt);  
    };

    mtx* S  = smatx(this -> _metx, this -> _mety, this -> _metz);
    mtx  n_ = S -> T().dot(this -> p_nu2 -> N()).dot(S); 
    mtx  n  = S -> T().dot(this -> p_nu1 -> N()).dot(S); 

    mtx* sol1 = nullptr; 
    mtx* sol2 = nullptr; 
    mtx* lin1 = nullptr; 
    mtx* lin2 = nullptr; 
    mtx* vx   = nullptr; 
    mtx* vx_  = nullptr; 
    int n_pts = intersection_ellipses(this -> p_nu1 -> N(), &n_, &lin1, &vx , &sol1); 
    n_pts    += intersection_ellipses(this -> p_nu2 -> N(), &n , &lin2, &vx_, &sol2); 

    if (!vx && !vx_){safe_del(&S); return 0;}
    safe_del(&sol1); 
    safe_del(&sol2); 
    safe_del(&lin1);
    safe_del(&lin2);

    if (vx && vx_){
        mtx* _vn = make_nu(S, vx ); 
        mtx*  vn = make_nu(S, vx_); 
        (*v_) = vx_ -> cat(_vn); 
        (*v ) = vx  -> cat( vn); 
        safe_del(&_vn); safe_del(&vn); 
        safe_del(&vx_); safe_del(&vx); 
        
    }
    
    if (vx ){(*v_) = make_nu(S, vx ); (*v ) = vx ;}
    if (vx_){(*v ) = make_nu(S, vx_); (*v_) = vx_;}
    safe_del(&S);
    return n_pts; 
}

int nunu_solver::angle_cross(mtx** v, mtx** v_){
    mtx* p1 = this -> p_nu1 -> H_perp(); 
    mtx* p2 = this -> p_nu2 -> H_perp(); 
    mtx met(1, 3); 
    met._m[0][0] = this -> _metx; 
    met._m[0][1] = this -> _mety; 
    met._m[0][2] = this -> _metz; 
     
    int n_rts = 0;  
    mtx* agl = nullptr; //intersection_angle(p1, p2, &met, &n_rts);
    if (!n_rts){delete agl; return n_rts;}
    *v  = new mtx(n_rts, 3);
    *v_ = new mtx(n_rts, 3); 
    for (int i(0); i < n_rts; ++i){
        mtx v1 = make_ellipse(p1, agl -> _m[i][0]); 
        mtx v2 = make_ellipse(p2, agl -> _m[i][1]); 
        (*v ) -> copy(&v1, i, 0, 3); 
        (*v_) -> copy(&v2, i, 0, 3); 
    }
    delete agl; 
    return n_rts; 
}

void nunu_solver::make_neutrinos(mtx* v, mtx* v_){
    if (!v || !v_){return;}

    mtx invH1 = this -> p_nu1 -> H_perp() -> inv().dot(v  -> T()); 
    mtx invH2 = this -> p_nu2 -> H_perp() -> inv().dot(v_ -> T()); 
    mtx nux1  = this -> p_nu1 -> K() -> dot(v  -> T()).T(); 
    mtx nux2  = this -> p_nu2 -> K() -> dot(v_ -> T()).T(); 

    this -> m_nu1 = new mtx(invH1.dim_j, 3); 
    this -> m_nu2 = new mtx(invH2.dim_j, 3); 

    double xd = -1; 
    for (int x(0); x < invH1.dim_j; ++x){
        double a1 = std::atan2(invH1._m[1][x], invH1._m[0][x]); 
        double a2 = std::atan2(invH2._m[1][x], invH2._m[0][x]); 
        double dx = distance(this -> p_nu1 -> H(), a1, this -> p_nu2 -> H(), a2); 
        if (std::isnan(dx)){continue;}
        if (this -> m_agl -> unique(0, 2, a1, dx)){continue;}
        this -> m_agl -> assign(1, this -> m_lx, a2); 

        this -> m_nu1 -> copy(&nux1, this -> m_lx, x, 3); 
        this -> m_nu2 -> copy(&nux2, this -> m_lx, x, 3); 
        if (xd < 0 || xd > dx){this -> m_bst = this -> m_lx; xd = dx;}
        this -> m_lx++;  
    }
    if (xd < 0){return;} 

    std::map<double, nunu_t>::iterator itr = this -> solvs.begin(); 
//    if (this -> solvs.size() && xd > itr -> first){return;}
    nunu_t* nx = &this -> solvs[xd];
    if (nx -> nu1){delete nx -> nu1;}
    if (nx -> nu2){delete nx -> nu2;}
    nx -> nu1 = new mtx(1, 3);
    nx -> nu1 -> copy(this -> m_nu1, 0, this -> m_bst, 3); 
    nx -> nu2 = new mtx(1, 3);
    nx -> nu2 -> copy(this -> m_nu2, 0, this -> m_bst, 3); 
    nx -> nux1 = this -> p_nu1; 
    nx -> nux2 = this -> p_nu2; 
}

