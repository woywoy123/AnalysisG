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

    this -> engines.reserve(parameters -> phys_pairs -> size()); 
    this -> params = parameters; 
}

ellipse::~ellipse(){
    for (size_t x(0); x < this -> engines.size(); ++x){delete this -> engines[x];}
    this -> solvs.clear(); 
}


void ellipse::prepare(double mt, double mw, std::vector<conuic*>* nxc){
    for (size_t x(0); x < this -> params -> phys_pairs -> size(); ++x){
        std::pair<particle_template*, particle_template*> p = this -> params -> phys_pairs -> at(x); 
        nuelx* nx = new nuelx(p.first, p.second, mw, mt); 
        if (nxc){nx -> cnx = nxc -> at(x);}
        this -> engines.push_back(nx); 
    }

    int lx = -1; 
    size_t le = this -> engines.size();   
    for (size_t x(0); x < le; ++x){
        for (size_t y(0); y < le; ++y){
            //if (this -> engines[x] -> l -> lnk -> hash == this -> engines[y] -> l -> lnk -> hash){continue;}
            //if (this -> engines[x] -> b -> lnk -> hash == this -> engines[y] -> b -> lnk -> hash){continue;}
            this -> pairings[++lx] = std::tuple<nuelx*, nuelx*>(this -> engines[x], this -> engines[y]);  
        }
    } 
    delete this -> params -> phys_pairs; 
    this -> params -> phys_pairs = nullptr; 
}

void ellipse::solve(){
    auto lamb1 =[this](nuelx* nux1, nuelx* nux2, double mt_, double mw_) -> bool{
        if (mt_ < mw_){return false;}
        nux1 -> update(mt_, mw_); 
        return this -> generate(nux1, nux2); 
    }; 

    auto lamb2 =[this](nuelx* nux1, nuelx* nux2, double mt_, double mw_) -> bool{
        if (mt_ < mw_){return false;}
        nux2 -> update(mt_, mw_); 
        return this -> generate(nux1, nux2); 
    }; 

    std::map<int, std::tuple<nuelx*, nuelx*>>::iterator itr = this -> pairings.begin(); 
    for (; itr != this -> pairings.end(); ++itr){
        std::tuple<nuelx*, nuelx*> px = itr -> second;
        nuelx* nx1 = std::get<0>(px); 
        nuelx* nx2 = std::get<1>(px);

        int x = 0; 
        double mt1 = nx1 -> mt;
        double mw1 = nx1 -> mw;
        double mt2 = nx2 -> mt;
        double mw2 = nx2 -> mw; 
        while (true){
            double mw1_1, mw1_2, mt1_1, mt1_2; 
            nx1 -> Z_mW(&mw1_1, &mw1_2);
            nx1 -> Z_mT(&mt1_1, &mt1_2, mw1_1); 

            double mw2_1, mw2_2, mt2_1, mt2_2; 
            nx2 -> Z_mW(&mw2_1, &mw2_2); 
            nx2 -> Z_mT(&mt2_1, &mt2_2, mw2_1); 
            if (!x){this -> generate(nx1, nx2);}
            if (lamb1(nx1, nx2, mt1_1, mw1_1)){mt1 = mt1_1; mw1 = mw1_1;}
            if (lamb1(nx1, nx2, mt1_1, mw1_2)){mt1 = mt1_1; mw1 = mw1_2;}
            if (lamb2(nx1, nx2, mt2_1, mw2_1)){mt2 = mt2_1; mw2 = mw2_1;}
            if (lamb2(nx1, nx2, mt2_1, mw2_2)){mt2 = mt2_1; mw2 = mw2_1;}
            nx1 -> update(mt1, mw1);   
            nx2 -> update(mt2, mw2);   
            ++x;
            
            if (x < this -> params -> iterations){continue;}
            break;
        }
    }
}

int ellipse::generate(nuelx* nu1, nuelx* nu2){
    this -> flush(); 
    this -> p_nu1 = nu1; this -> p_nu2 = nu2; 
    mtx* vi = nullptr; mtx* vi_ = nullptr;
    mtx* vr = nullptr; mtx* vr_ = nullptr;

    this -> intersection(&vi, &vi_); 
    this -> angle_cross( &vr, &vr_); 

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
    this -> clear(&this -> m_nu1); 
    this -> clear(&this -> m_nu2);  
    this -> m_lx   = 0; 
    this -> m_bst  = 0; 
}

int ellipse::intersection(mtx** v, mtx** v_){
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

    if (!vx && !vx_){this -> clear(&S); return 0;}
    this -> clear(&sol1); 
    this -> clear(&sol2); 
    this -> clear(&lin1);
    this -> clear(&lin2);

    if (vx && vx_){
        mtx* _vn = make_nu(S, vx ); 
        mtx*  vn = make_nu(S, vx_); 
        (*v_) = vx_ -> cat(_vn); 
        (*v ) = vx  -> cat( vn); 
        this -> clear(&_vn); this -> clear(&vn); 
        this -> clear(&vx_); this -> clear(&vx); 
        
    }
    
    if (vx ){(*v_) = make_nu(S, vx ); (*v ) = vx ;}
    if (vx_){(*v ) = make_nu(S, vx_); (*v_) = vx_;}
    this -> clear(&S);
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

int ellipse::violation_test(mtx* v, nuelx* pnu, mtx** out){
    mtx invH = pnu -> H_perp() -> inv().dot(v -> T()); 
    mtx nux  = pnu -> K() -> dot(v -> T()).T(); 
    mtx* ox = new mtx(invH.dim_j, 4); 
    (*out) = ox; 

    int mlx  = 0; 
    int mbst = -1;
    long double xd = -1; 
    for (int x(0); x < invH.dim_j; ++x){
        particle_template t_ = particle_template(nux._m[x][0], nux._m[x][1], nux._m[x][2]); 
        t_.iadd(pnu -> l -> lnk);
        t_.iadd(pnu -> b -> lnk);
        double mt_ = t_.mass; 
        if (std::isnan(std::abs(mt_))){continue;}

        particle_template w_ = particle_template(nux._m[x][0], nux._m[x][1], nux._m[x][2]); 
        w_.iadd(pnu -> l -> lnk);
        double mw_ = w_.mass; 

        long double lx = pnu -> cnx -> dPl0(pnu -> cnx -> cache -> Mobius.tstar); 
        if (std::abs(lx) > this -> params -> violation){continue;}

        long double t1, z1; 
        pnu -> cnx -> get_TauZ(pnu -> Sx(), pnu -> Sy(), &z1, &t1); 
        if (std::fabs(t1) >= 1){continue;}
        z1 = std::abs(z1); 

        long double l0 = pnu -> cnx -> dPdtL0(t1, z1); 
        if (std::fabs(pnu -> cnx -> dPdt(l0, t1, z1)) > this -> params -> violation){continue;}

        long double tp_; 
        std::complex<long double> z_; 
        pnu -> cnx -> mass_line(mw_, mt_, &z_, &tp_); 
        z1 = z_.real(); 
        t1 = tp_; 

        l0 = pnu -> cnx -> dPdtL0(t1, z1); 
        if (std::fabs(pnu -> cnx -> dPdt(l0, t1, z1)) > this -> params -> violation){continue;}
        l0 = pnu -> cnx -> dPl0(t1);

        long double dx = std::abs(l0);
        if (xd > -1 && xd < dx){continue;}
        ox -> copy(&nux, mlx, x, 3); 
        ox -> _m[mlx][3] = double(dx); 
        mbst = mlx; xd = dx; ++mlx;  
    }
    return mbst;
}

void ellipse::make_neutrinos(mtx* v, mtx* v_){
    if (!v || !v_){return;}
    int mbx1 = this -> violation_test(v , this -> p_nu1, &this -> m_nu1); 
    int mbx2 = this -> violation_test(v_, this -> p_nu2, &this -> m_nu2); 
    if (mbx1 < 0 && mbx2 < 0){return;}

    if (mbx1 > -1){
        double sx = this -> m_nu1 -> _m[mbx1][3];    
        nunu_t* nx = &this -> solvs[sx]; 
        if (nx -> nu1){delete nx -> nu1;}
        nx -> nu1 = new mtx(1, 4);
        nx -> nu1 -> copy(this -> m_nu1, 0, mbx1, 4); 
        nx -> nux1 = this -> p_nu1; 
    }

    if (mbx2 > -1){
        double sx = this -> m_nu2 -> _m[mbx2][3];    
        nunu_t* nx = &this -> solvs[sx]; 
        if (nx -> nu2){delete nx -> nu2;}
        nx -> nu2 = new mtx(1, 4);
        nx -> nu2 -> copy(this -> m_nu2, 0, mbx2, 4); 
        nx -> nux2 = this -> p_nu2; 
    }
}
