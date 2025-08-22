#include <reconstruction/nusol.h>
#include <ellipse/ellipse.h>

nunu_t::nunu_t(){}
nunu_t::~nunu_t(){
    if (this -> nu1){delete this -> nu1;}
    if (this -> nu2){delete this -> nu2;}
    if (this -> agl){delete this -> agl;}
}

ellipse::ellipse(nusol_t* parameters){
    this -> _metx = parameters -> met_x;  
    this -> _mety = parameters -> met_y;
    this -> _metz = parameters -> met_z;  
  
    for (size_t x(0); x < parameters -> targets -> size(); ++x){
        if (!parameters -> targets -> at(x) -> is_lep){continue;}
        this -> leptons.push_back(parameters -> targets -> at(x));
    }

    for (size_t x(0); x < parameters -> targets -> size(); ++x){
        if (!parameters -> targets -> at(x) -> is_b){continue;}
        this -> bquarks.push_back(parameters -> targets -> at(x));
    }

    this -> n_lp = this -> leptons.size();
    this -> n_bs = this -> bquarks.size();
    this -> engines.reserve(this -> n_lp*this -> n_bs); 
    this -> params = parameters; 
}

ellipse::~ellipse(){
    for (size_t x(0); x < this -> engines.size(); ++x){
        delete this -> engines[x];
    }
    this -> solvs.clear(); 
}


void ellipse::prepare(double mt, double mw){
    for (int l(0); l < this -> n_lp; ++l){
        for (int b(0); b < this -> n_bs; ++b){
            nuelx* nl = new nuelx(this -> bquarks[b], this -> leptons[l], mw, mt); 
            this -> engines.push_back(nl); 
        }
    }
  
    int lx = -1; 
    size_t le = this -> engines.size();   
    for (size_t x(0); x < le; ++x){
        for (size_t y(0); y < le; ++y){
            if (y == x){continue;}
            this -> pairings[++lx] = std::tuple<nuelx*, nuelx*>(this -> engines[x], this -> engines[y]);  
        }
    } 
}

void ellipse::solve(){
    auto lamb1 =[this](nuelx* nux1, nuelx* nux2, double mt_, double mw_, double* dz) -> bool{
        if (mt_ < 0 || mw_ < 0){return false;}
        double z = nux1 -> Z2(); 
        nux1 -> update(mt_, mw_); 
        this -> generate(nux1, nux2); 
        bool r = fabs(nux1 -> Z2() - z) < *dz; 
        *dz = (r || *dz < 0) ? fabs(nux1 -> Z2() - z) : *dz; 
        return r; 
    }; 

    auto lamb2 =[this](nuelx* nux1, nuelx* nux2, double mt_, double mw_, double* dz) -> bool{
        if (mt_ < 0 || mw_ < 0){return false;}
        double z = nux2 -> Z2(); 
        nux2 -> update(mt_, mw_); 
        this -> generate(nux1, nux2); 
        bool r = fabs(nux2 -> Z2() - z) < *dz; 
        *dz = (r || *dz < 0) ? fabs(nux2 -> Z2() - z) : *dz; 
        return r;
    }; 

    std::map<int, std::tuple<nuelx*, nuelx*>>::iterator itr = this -> pairings.begin(); 
    for (; itr != this -> pairings.end(); ++itr){
        std::tuple<nuelx*, nuelx*> px = itr -> second;
        nuelx* nx1 = std::get<0>(px); 
        nuelx* nx2 = std::get<1>(px);

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

int ellipse::generate(nuelx* nu1, nuelx* nu2){
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

void ellipse::flush(){
    if (this -> m_nu1){delete this -> m_nu1;}
    if (this -> m_nu2){delete this -> m_nu2;} 
    if (this -> m_agl){delete this -> m_agl;}
    this -> m_nu1 = nullptr; 
    this -> m_nu2 = nullptr; 
    this -> m_agl = nullptr; 
    this -> m_lx   = 0; 
    this -> m_bst  = 0; 
}

int ellipse::intersection(mtx** v, mtx** v_){
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

int ellipse::angle_cross(mtx** v, mtx** v_){
    mtx* p1 = this -> p_nu1 -> H_perp(); 
    mtx* p2 = this -> p_nu2 -> H_perp(); 
    mtx met(1, 3); 
    met._m[0][0] = this -> _metx; 
    met._m[0][1] = this -> _mety; 
    met._m[0][2] = this -> _metz; 
     
    int n_rts = 0;  
    mtx* agl = intersection_angle(p1, p2, &met, &n_rts);
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

void ellipse::make_neutrinos(mtx* v, mtx* v_){
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
    if (this -> solvs.size() && xd > itr -> first){return;}
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

std::vector<particle_template*> ellipse::nunu_make(){
    std::vector<particle_template*> nus = {}; 

    if (!this -> solvs.size()){return nus;}
    std::map<double, nunu_t>::iterator itr = this -> solvs.begin(); 
    if (itr -> first > this -> params -> limit){return nus;}

    nunu_t* nx = &itr -> second; 

    particle_template* nu1 = new particle_template(nx -> nu1 -> _m[0][0], nx -> nu1 -> _m[0][1], nx -> nu1 -> _m[0][2]); 
    particle_template* nu2 = new particle_template(nx -> nu2 -> _m[0][0], nx -> nu2 -> _m[0][1], nx -> nu2 -> _m[0][2]); 
    nu1 -> type = "nunu"; 
    nu2 -> type = "nunu"; 
    
    nu1 -> register_parent(nx -> nux1 -> l -> lnk); 
    nu2 -> register_parent(nx -> nux2 -> l -> lnk); 
    nu1 -> register_parent(nx -> nux1 -> b -> lnk); 
    nu2 -> register_parent(nx -> nux2 -> b -> lnk); 

    nus.push_back(nu1); 
    nus.push_back(nu2); 
    return nus; 
}
