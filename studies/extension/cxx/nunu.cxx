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
    this -> nu1 = new nusol(b1, l1, mt1, mt1, delP);
    this -> nu2 = new nusol(b2, l2, mt2, mt2, delP);
}

int nunu::intersection(double*** v, double*** v_, double metx, double mety, double metz){
    double** S  = smatx(metx, mety, metz); 
    double** n_ = dot(dot(T(S), this -> nu2 -> N(), true),  S, true); 

    double** sol = nullptr; 
    double** lin = nullptr;  
    int n_pts = intersection_ellipses(this -> nu1 -> N(), n_, &lin, v, &sol); 
    clear(n_, 3, 3); clear(sol, 1, n_pts); clear(lin, 2, 3); 

    double** vT = T(*v, n_pts, 3); 
    double** vl = dot(S, vT, true, 3, 3, 3, n_pts); 
    clear(vT , 3, n_pts); 
    *v_ = T(vl, 3, n_pts);
    clear(vl, 3, n_pts); 
    return n_pts; 
}

int nunu::angle_cross(double*** v, double*** v_, double metx, double mety, double metz){
    double** p1 = this -> nu1 -> H_perp(); 
    double** p2 = this -> nu2 -> H_perp(); 

    double** met = matrix(1, 3);
    met[0][0] = metx; met[0][1] = mety; met[0][2] = metz; 

    int n_rts = 0;  
    double** agl = get_intersection_angle(p1, p2, met, &n_rts);
    double** _v  = matrix(n_rts, 3);
    double** _v_ = matrix(n_rts, 3);
    for (int i(0); i < n_rts; ++i){
        double** v1 = make_ellipse(p1, agl[i][0]); 
        double** v2 = make_ellipse(p2, agl[i][1]); 
        _copy(_v [i], v1[0], 3); 
        _copy(_v_[i], v2[0], 3); 
        clear(v1, 1, 3); clear(v2, 1, 3); 
    }
    clear(agl, n_rts, 2); 
    clear(met, 1, 3); 
    *v = _v; *v_ = _v_; 
    return n_rts; 
}

void nunu::make_neutrinos(double** v, double** v_, double* d_, double* agl){
    double** lT1 = T(v , 1, 3); 
    double** lT2 = T(v_, 1, 3);
    double** invH1 = dot(inv(this -> nu1 -> H()), lT1, true, 3, 3, 3, 1); 
    double** invH2 = dot(inv(this -> nu2 -> H()), lT2, true, 3, 3, 3, 1); 

    double a1 = std::atan2(invH1[1][0], invH1[0][0]); 
    double a2 = std::atan2(invH2[1][0], invH2[0][0]); 
    clear(invH1, 3, 1); clear(invH2, 3, 1); 

    *d_ = distance(this -> nu1 -> H(), a1, this -> nu2 -> H(), a2); 
    agl[0] = a1; agl[1] = a2;

    double** l1 = dot(this -> nu1 -> K(), lT1, false, 3, 3, 3, 1); 
    double** l2 = dot(this -> nu2 -> K(), lT2, false, 3, 3, 3, 1);  
    clear(lT1, 3, 1); clear(lT2, 3, 1); 
    lT1 = T(l1, 3, 1); lT2 = T(l2, 3, 1); 
    _copy( v, lT1, 1, 3); 
    _copy(v_, lT2, 1, 3); 
    clear(lT1, 1, 3); clear(lT2, 1, 3); 
    clear(l1, 3, 1);  clear(l2, 3, 1);
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
    double _mw1, _mw2, _mw3, _mw4; 
    this -> nu1 -> get_mw(&_mw1, &_mw2);
    this -> nu2 -> get_mw(&_mw3, &_mw4); 

    std::cout << _mw1 << " " << _mw2 << std::endl;
    std::cout << _mw3 << " " << _mw4 << std::endl;

    double _mt1, _mt2, _mt3, _mt4;
    this -> nu1 -> get_mt(&_mt1, &_mt2);
    this -> nu2 -> get_mt(&_mt3, &_mt4); 
    std::cout << _mt1 << " " << _mt2 << std::endl;
    std::cout << _mt3 << " " << _mt4 << std::endl;

}

int nunu::generate(double metx, double mety, double metz){
    double** vi = nullptr; double** vi_ = nullptr; double** di = nullptr;
    int n_pts = this -> intersection(&vi, &vi_, metx, mety, metz); 
    di = matrix(n_pts, 1);
   // print(Rxyz(S, 0, 0, 0)); 

    double** vr = nullptr; double** vr_ = nullptr; double** dr = nullptr; 
    int n_rts = this -> angle_cross(&vr, &vr_, metx, mety, metz); 
    dr = matrix(n_rts, 1); 

    int lx = n_pts + n_rts; 
    this -> m_agl_ = matrix(lx, 2);
    for (int i(0); i < n_pts; ++i){
        this -> make_neutrinos(&vi[i], &vi_[i], di[i], this -> m_agl_[i]);
    }
    for (int i(0); i < n_rts; ++i){
        this -> make_neutrinos(&vr[i], &vr_[i], dr[i], this -> m_agl_[n_rts + i]);
    }

    double** v  = matrix(lx, 3); 
    double** v_ = matrix(lx, 3); 
    double** d_ = matrix(lx, 1); 
    for (int x(0); x < n_pts; ++x){
        _copy( v[x],  vi[x], 3); 
        _copy(v_[x], vi_[x], 3); 
        d_[x][0] = di[x][0]; 
    }

    for (int x(0); x < n_rts; ++x){
        _copy( v[n_pts + x],  vr[x], 3); 
        _copy(v_[n_pts + x], vr_[x], 3); 
        d_[n_pts + x][0] = dr[x][0]; 
    }

    clear(vi , n_pts, 3); clear(vr , n_rts, 3); 
    clear(vi_, n_pts, 3); clear(vr_, n_rts, 3); 
    clear(di , n_pts, 1); clear(dr , n_rts, 1); 
    if (!this -> m_nu1_){this -> m_nu1_ = v;}
    if (!this -> m_nu2_){this -> m_nu2_ = v_;} 
    if (!this -> m_d1_ ){this -> m_d1_  = d_;} 
    if (!this -> m_lx  ){this -> m_lx   = lx;} 

    //this -> m_v1 = this -> make_particle(v , d_, lx);  
    //this -> m_v2 = this -> make_particle(v_, d_, lx);  
    //clear(v_, lx, 3); clear(v, lx, 3); clear(d_, lx, 1); 
    return lx; 
}

double** nunu::loss(int* lx){
    this -> generate(this -> metx, this -> mety, 0); 
    double** mx = matrix(3, 1); 
    for (int x(0); x < 3; ++x){mx[x][0] = 1.0;}
    for (int x(0); x < 3; ++x){
        if (x > this -> m_lx){continue;}
        mx[x][0] = this -> m_d1_[x][0]; 
    }
    *lx = 3; 
    return mx; 
}

double** nunu::jacobian(int* ix, int* jx){
    double** J = matrix(3, 4); 
    double** dH_mw1 = this -> nu1 -> dH_dmW();
    double** dH_mw2 = this -> nu2 -> dH_dmW(); 

    double** dH_mt1 = this -> nu1 -> dH_dmT();
    double** dH_mt2 = this -> nu2 -> dH_dmT(); 
    *ix = 3; *jx = 4;  


    int y_ = 0; 
    double mxl = -1; 
    for (int x(0); x < this -> m_lx; ++x){
        if (!x){mxl = this -> m_d1_[x][0]; y_ = x;}
        if (mxl < this -> m_d1_[x][0]){continue;}
        mxl = this -> m_d1_[x][0]; y_ = x; 
    }

    if (mxl >= 0){
        double** vmw1 = make_ellipse(dH_mw1, this -> m_agl_[y_][0]); 
        double** vmw2 = make_ellipse(dH_mw2, this -> m_agl_[y_][1]); 
        double** vmt1 = make_ellipse(dH_mt1, this -> m_agl_[y_][0]); 
        double** vmt2 = make_ellipse(dH_mt2, this -> m_agl_[y_][1]); 
        for (int y(0); y < 3; ++y){ 
            if (this -> m_lx < y){break;}
            J[y][0] = vmt1[0][y]; J[y][1] = vmw1[0][y];  
            J[y][2] = vmt2[0][y]; J[y][3] = vmw2[0][y];  
        }

        clear(vmw1, 1, 3); clear(vmw2, 1, 3); 
        clear(vmt1, 1, 3); clear(vmt2, 1, 3); 
    }
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
    delete this -> nu1; 
    delete this -> nu2; 
}


void nunu::flush(){
    this -> nu1 -> flush(); 
    this -> nu2 -> flush(); 

    if (this -> m_nu1_){clear(this -> m_nu1_, this -> m_lx, 3);}
    if (this -> m_nu2_){clear(this -> m_nu2_, this -> m_lx, 3);} 
    if (this -> m_d1_ ){clear(this -> m_d1_ , this -> m_lx, 1);} 
    if (this -> m_agl_){clear(this -> m_agl_, this -> m_lx, 2);}
    this -> m_nu1_ = nullptr; 
    this -> m_nu2_ = nullptr; 
    this -> m_d1_  = nullptr; 
    this -> m_agl_ = nullptr; 
    this -> m_lx   = 0; 
}

void nunu::update(double** params){
    double mw1; double mw2; 
    this -> nu1 -> get_mw(&params[0][1], &mw1);
    this -> nu2 -> get_mw(&params[0][3], &mw2); 
    std::cout << mw1 << " " << mw2 << std::endl;
    print_(params, 1, 4); 

    this -> nu1 -> update(params[0][0], params[0][1]);
    this -> nu2 -> update(params[0][2], params[0][3]); 
}


