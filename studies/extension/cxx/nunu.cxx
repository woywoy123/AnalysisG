#include "nunu.h"


nunu::nunu(
        double b1_px, double b1_py, double b1_pz, double b1_e,
        double l1_px, double l1_py, double l1_pz, double l1_e, 
        double b2_px, double b2_py, double b2_pz, double b2_e,
        double l2_px, double l2_py, double l2_pz, double l2_e,
        double mt1  , double mt2  , double mw1  , double mw2
){
    particle* b1 = new particle(b1_px, b1_py, b1_pz, b1_e);
    particle* l1 = new particle(l1_px, l1_py, l1_pz, l1_e);
    particle* b2 = new particle(b2_px, b2_py, b2_pz, b2_e);
    particle* l2 = new particle(l2_px, l2_py, l2_pz, l2_e);
    this -> nu1 = new nusol(b1, l1, mw1, mt1, true);
    this -> nu2 = new nusol(b2, l2, mw2, mt2, true);
}


nunu::nunu(
        particle* b1, particle* b2, particle* l1, particle* l2, 
        double mt1, double mt2, double mw1, double mw2, bool delP
){
    this -> nu1 = new nusol(b1, l1, mw1, mt1, delP);
    this -> nu2 = new nusol(b2, l2, mw2, mt2, delP);
}

int nunu::intersection(mtx** v, mtx** v_, double metx, double mety, double metz){
    mtx* S  = smatx(metx, mety, metz);
    mtx  n_ = S -> T().dot(this -> nu2 -> N()).dot(S); 
    mtx* sol = nullptr; 
    mtx* lin = nullptr; 

    int n_pts = intersection_ellipses(this -> nu1 -> N(), &n_, &lin, v, &sol); 
    if (!sol){delete S;return 0;}
    mtx vn = (*v) -> T(); 
    mtx vl = S -> dot(vn).T();
    *v_ = new mtx(vl); 
    delete S; 
    delete sol; 
    delete lin; 
    return n_pts; 
}

int nunu::angle_cross(mtx** v, mtx** v_, double metx, double mety, double metz){
    mtx* p1 = this -> nu1 -> H_perp(); 
    mtx* p2 = this -> nu2 -> H_perp(); 
    mtx met(1, 3); 
    met._m[0][0] = metx; met._m[0][1] = mety; met._m[0][2] = metz; 
    
    int n_rts = 0;  
    mtx* agl = get_intersection_angle(p1, p2, &met, &n_rts);
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

void nunu::make_neutrinos(mtx* v, mtx* v_){
    if (!v || !v_){return;}

    mtx invH1 = this -> nu1 -> H_perp() -> inv().dot(v  -> T()); 
    mtx invH2 = this -> nu2 -> H_perp() -> inv().dot(v_ -> T()); 
    mtx nux1  = this -> nu1 -> K() -> dot(v  -> T()).T(); 
    mtx nux2  = this -> nu2 -> K() -> dot(v_ -> T()).T(); 

    this -> m_nu1 = new mtx(invH1.dim_j, 3); 
    this -> m_nu2 = new mtx(invH2.dim_j, 3); 

    double xd = -1; 
    for (int x(0); x < invH1.dim_j; ++x){
        double a1 = std::atan2(invH1._m[1][x], invH1._m[0][x]); 
        double a2 = std::atan2(invH2._m[1][x], invH2._m[0][x]); 
        if (this -> m_agl -> unique(0, 1, a1, a2)){continue;}
        double dx = distance(this -> nu1 -> H(), a1, this -> nu2 -> H(), a2); 
        this -> m_agl -> assign(2, this -> m_lx, dx); 

        this -> m_nu1 -> copy(&nux1, this -> m_lx, x, 3); 
        this -> m_nu2 -> copy(&nux2, this -> m_lx, x, 3); 
        if (xd < 0 || xd > dx){this -> m_bst = this -> m_lx; xd = dx;}
        this -> m_lx++;  
    }
}


particle** nunu::make_particle(double** v, double** d, int lx){
    particle** pxt = (particle**)malloc(lx*sizeof(particle*)); 
    for (int x(0); x < lx; ++x){
        pxt[x] = new particle(v[x]);
        pxt[x] -> d = d[x][0]; 
    }
    return pxt;  
}

void nunu::get_misc(){
    if (!this -> verbose){return;}
    this -> nu1 -> misc(); 
    this -> nu2 -> misc(); 
    std::cout << "============" << std::endl; 
}

int nunu::generate(double metx, double mety, double metz){
    mtx* vi = nullptr; mtx* vi_ = nullptr;
    int n_pts = this -> intersection(&vi, &vi_, metx, mety, metz); 
    
    mtx* vr = nullptr; mtx* vr_ = nullptr;
    int n_rts = this -> angle_cross(&vr, &vr_, metx, mety, metz); 
    // print(Rxyz(S, 0, 0, 0)); 

    int lx = n_pts + n_rts; 
    this -> m_agl = new mtx(3, lx); 

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

double** nunu::loss(int* lx){
    return nullptr; 
    //this -> flush(); 
    //this -> generate(this -> metx, this -> mety, 0); 
    //double** mx = matrix(1, 1); 
    //for (int x(0); x < 1; ++x){mx[x][0] = 1.0;}
    //for (int x(0); x < this -> m_lx; ++x){
    //    double dx = this -> m_d1_[x][0]; 
    //    if (!x){mx[0][0] = dx;}
    //    if (mx[0][0] < dx){continue;}
    //    mx[0][0] = dx; 
    //}
    //*lx = 1; 
    //return mx; 
}

double** nunu::jacobian(int* ix, int* jx){
    double** J = matrix(1, 4); 
    //int y_ = 0; 
    //double mxl = -1; 
    //for (int x(0); x < this -> m_lx; ++x){
    //    if (!x){mxl = this -> m_d1_[x][0]; y_ = x;}
    //    if (mxl < this -> m_d1_[x][0]){continue;}
    //    mxl = this -> m_d1_[x][0]; y_ = x; 
    //}

    //double a1(0), a2(0);  
    //if (mxl >= 0){
    //    a1 = this -> m_agl_[y_][0]; 
    //    a2 = this -> m_agl_[y_][1]; 
    //}

    //double** dH_mw1 = this -> nu1 -> dH_dmW();
    //double** dH_mw2 = this -> nu2 -> dH_dmW(); 

    //double** dH_mt1 = this -> nu1 -> dH_dmT();
    //double** dH_mt2 = this -> nu2 -> dH_dmT(); 
    //*ix = 1; *jx = 4;  

    //double** vmw1 = make_ellipse(dH_mw1, a1); 
    //double** vmw2 = make_ellipse(dH_mw2, a2); 
    //double** vmt1 = make_ellipse(dH_mt1, a1); 
    //double** vmt2 = make_ellipse(dH_mt2, a2); 

    //double** h1 = make_ellipse(this -> nu1 -> H(), a1); 
    //double** h2 = make_ellipse(this -> nu2 -> H(), a2); 

    //double** Hx = arith(h1, h2, -1.0, 1, 3);
    //J[0][0] = 2 * (Hx[0][0] * vmt1[0][0] + Hx[0][1] * vmt1[0][1] + Hx[0][2] * vmt1[0][2]); 
    //J[0][1] = 2 * (Hx[0][0] * vmw1[0][0] + Hx[0][1] * vmw1[0][1] + Hx[0][2] * vmw1[0][2]); 
    //J[0][2] = 2 * (Hx[0][0] * vmt2[0][0] + Hx[0][1] * vmt2[0][1] + Hx[0][2] * vmt2[0][2]); 
    //J[0][3] = 2 * (Hx[0][0] * vmw2[0][0] + Hx[0][1] * vmw2[0][1] + Hx[0][2] * vmw2[0][2]); 

    //clear(Hx, 1, 3); 
    //clear(  h1, 1, 3); clear(  h2, 1, 3); 
    //clear(vmw1, 1, 3); clear(vmw2, 1, 3); 
    //clear(vmt1, 1, 3); clear(vmt2, 1, 3); 
    return J; 
}

void nunu::get_nu(particle** nu1, particle** nu2, int l){
    *nu1 = this -> m_v1[l]; this -> m_v1[l] = 0; 
    *nu2 = this -> m_v2[l]; this -> m_v2[l] = 0; 
}

void nunu::_clear(){
    free(this -> m_v1);
    free(this -> m_v2);  
}

nunu::~nunu(){
    this -> flush(); 
    delete this -> nu1; 
    delete this -> nu2; 
}


void nunu::flush(){
    if (this -> m_nu1){delete this -> m_nu1;}
    if (this -> m_nu2){delete this -> m_nu2;} 
    if (this -> m_agl){delete this -> m_agl;}
    this -> m_nu1 = nullptr; 
    this -> m_nu2 = nullptr; 
    this -> m_agl = nullptr; 
    this -> m_lx   = 0; 
}

void nunu::update(double** params){
    this -> flush();
    if (!params){return;}

//    print_(params, 1, 4); 
//    double mw1 = 0; double mw2 = 0; 
//    double mw3 = 0; double mw4 = 0; 
//    this -> nu1 -> get_mw(&mw1, &mw2);
//    this -> nu2 -> get_mw(&mw3, &mw4); 
//
//    //double mt1 = 0; double mt2 = 0; 
//    //double mt3 = 0; double mt4 = 0; 
//    std::cout << mw2 << " " << mw4 << std::endl; 
//    this -> nu1 -> update(params[0][0], mw2);
//    this -> nu2 -> update(params[0][2], mw4); 
}


