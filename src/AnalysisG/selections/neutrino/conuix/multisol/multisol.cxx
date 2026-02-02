#include "multisol/multisol.h"
#include <algorithm>
#include <vector>

void get_trgt(
        stats_t* metric, params_t* prm, 
        std::vector<particle_template*>* jts, 
        std::vector<particle_template*>* lps
){
    size_t lpn = 0; size_t jpn = 0; 
    std::vector<particle_template*>* det = prm -> targets; 
    for (size_t x(0); x < det -> size(); ++x){
        particle_template* ptr = det -> at(x); 
        if (ptr -> is_lep){lps -> push_back(ptr); ++lpn;}
        else {jts -> push_back(ptr); ++jpn;}
    }
    metric -> num_lep = lpn; 
    metric -> num_jet = jpn; 
    metric -> num_cmb = jpn * lpn; 
} 

void multisol::vio_t::print(int prc){
    std::cout << "------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(prc); 
    std::cout << "mw          : " << this -> mw           << "\n"; 
    std::cout << "mt          : " << this -> mt           << "\n"; 
    std::cout << "tau         : " << this -> tau          << "\n"; 
    std::cout << "Z           : " << this -> Z            << "\n"; 
    std::cout << "l0vio       : " << this -> l0vio        << "\n"; 
    std::cout << "dpdtl0      : " << this -> dpdtl0       << "\n"; 
    std::cout << "pl          : " << this -> pl           << "\n"; 
    std::cout << "dpdt        : " << this -> dpdt         << "\n"; 
    std::cout << "tau_pole (1): " << this -> tau_pole[0]  << " (2): " << tau_pole[1] << " \n"; 
    std::cout << "dpdz     (1): " << this -> dpdz[0]      << " (2): " << dpdz[1]     << " \n"; 
    std::cout << "------------------------" << std::endl;
}


multisol::multisol(params_t* prm){
    this -> param = prm;
    this -> param -> metx = std::cos(this -> param -> phi) * this -> param -> met;  
    this -> param -> mety = std::sin(this -> param -> phi) * this -> param -> met;

    std::vector<particle_template*> leps = {};
    std::vector<particle_template*> jets = {};
    get_trgt(&this -> metric, prm, &jets, &leps); 

    int p = 0; 
    this -> engines.reserve(this -> metric.num_cmb); 
    for (size_t x(0); x < this -> metric.num_lep; ++x){
        particle_template* lp = leps[x]; 
        for (size_t y(0); y < this -> metric.num_jet; ++y){
            particle_template* jt = jets[y]; 
            conuic* cnx = new conuic(jt, lp); 

            bool conv       = cnx -> converged; 
            long double err = std::fabs(cnx -> taustar[1]); 
            if (!conv || err > param -> tstar_lim){
                delete cnx; cnx = nullptr;
                continue; 
            }

            this -> engines[p] = cnx; ++p; 
            if (this -> metric.low_mob > 0 && this -> metric.low_mob < err){continue;}
            this -> metric.low_mob = err;
        }
    }
    this -> metric.num_apt = p;
    std::vector<conuic*> candidate(p, nullptr); 
    for (size_t x(0); x < p; ++x){candidate[x] = this -> engines[x];}
    this -> engines.clear(); this -> engines = candidate; 
}

multisol::~multisol(){
    this -> dSafe(&this -> engines); 
}


multisol::vio_t multisol::test(
        conuic* tr, matrix_t* nux, long double mt, long double mw, 
        long double t , long double z
){
    auto lmb =[](std::complex<long double> val) -> bool {return std::abs(val.imag()) > 0;};
    multisol::vio_t out; 
    out.tau_pole[0] = (std::abs(tr -> poles[0]) < 1) ? std::atanh(tr -> poles[0]) : -1; 
    out.tau_pole[1] = (std::abs(tr -> poles[0]) < 1) ? std::atanh(tr -> poles[1]) : -1; 
    
    coef_t r; 
    if (t != -1.0 && z != -1 && !nux){
        tr -> hyper(t);
        r = tr -> get_tauZ(
                tr -> Sx(tr -> _ct, tr -> _st, z), 
                tr -> Sy(tr -> _ct, tr -> _st, z)
        ); 
        out.tau = r.b; out.Z = std::abs(r.c); 
        r = tr -> masses(out.Z, out.tau); 
        out.mw = (!std::abs(r.a_cplx.imag())) ? std::abs(r.a_cplx.real()) : std::abs(r.a_cplx.imag()); 
        out.mt = (!std::abs(r.b_cplx.imag())) ? std::abs(r.b_cplx.real()) : std::abs(r.b_cplx.imag()); 
    }
    else {
        matrix_t Wv = tr -> make_w(nux);   
        matrix_t tv = tr -> make_top(nux); 

        std::complex<long double> rmw = tr -> get_mass(&Wv); 
        std::complex<long double> rmt = tr -> get_mass(&tv); 
        out.mw = std::abs(rmw.real()) * ( 0.5 - std::abs(rmw.imag()) > 0 ) * 2.0; 
        out.mt = std::abs(rmt.real()) * ( 0.5 - std::abs(rmt.imag()) > 0 ) * 2.0; 

        r = tr -> get_tauZ(
                tr -> Sx(out.mw, out.mt), 
                tr -> Sy(out.mw, out.mt)
        ); 
        out.tau = r.b; out.Z = std::abs(r.c); 
        tr -> hyper(out.tau); 
    }

    out.l0vio  = tr -> dPl0(); 
    out.dpdtl0 = tr -> dPdtL0(out.Z); 
    out.pl     = tr -> PL(out.Z, out.dpdtl0); 
    out.dpdt   = tr -> dPdt(out.Z, out.dpdtl0); 
    
    r = tr -> dPdZ0(out.Z); 
    out.dpdz[0] = std::abs(r.a_cplx.real()) * ( std::abs(r.a_cplx.imag()) == 0); 
    out.dpdz[1] = std::abs(r.b_cplx.real()) * ( std::abs(r.b_cplx.imag()) == 0); 
    return out;
}


int multisol::violation_test(conuic* v, matrix_t* sols){

    int mlx  = 0; 
    int mbst = -1;
    long double xd = -1; 
  
    for (int x(0); x < sols -> r; ++x){
        matrix_t nu = sols -> at(x); 
        if (std::abs(v -> taustar[2]) > 1){continue;}
        multisol::vio_t vio = this -> test(v, &nu, -1, -1, v -> taustar[2], 1.0); 
        if (std::abs(vio.l0vio) > this -> param -> violation){continue;}
        if (std::fabs(vio.tau) >= 1){continue;}
        long double dx = std::abs(vio.l0vio);
        if (xd > -1 && xd < dx){continue;}
        xd = dx; mbst = mlx;  mlx++;
    }

    return mbst;
}

void multisol::mass_sample(conuic* nux, long double mw, long double mt){
    coef_t ms = nux -> mass_line(mw, mt); 
    multisol::vio_t ev = this -> test(nux, nullptr, -1, -1, ms.b_cplx.real(), ms.a_cplx.real()); 
    if (std::isnan(ev.mw) || std::isnan(ev.mt)){return;}
    this -> test_points[nux -> hash][std::abs(ev.l0vio)] = ev; 
    ev.print(12);
}

void multisol::prescan(){
    auto lamb =[](std::complex<long double> val, long double thr) -> long double{
        long double vr = val.real(); 
        long double vi = val.imag(); 
        if (std::abs(thr - vi) < std::abs(thr - vr)){return vi;}
        return vr; 
    }; 



    abort(); 
    long double metx = this -> param -> metx; 
    long double mety = this -> param -> mety; 

    for (size_t x(0); x < this -> engines.size(); ++x){

        for (size_t y(0); y < 10; ++y){
            conuic* cnx = this -> engines[x]; 
            this -> mass_sample(cnx, this -> param -> mass_w, this -> param -> mass_t); 
            std::map<long double, multisol::vio_t>::iterator itr; 
            itr = this -> test_points[cnx -> hash].begin(); 
            multisol::vio_t vio = itr -> second; 
            coef_t r = cnx -> masses(vio.Z, cnx -> taustar[2]); 
            std::cout << r.a_cplx << " " << r.b_cplx << std::endl;
            this -> mass_sample(cnx, vio.mw, vio.mt); 

        }

//        coef_t ms1 = cnx -> mass_line(this -> param -> mass_w, this -> param -> mass_t); 
        //long double z1 = std::fabs(ms1.a_cplx.real());
        //long double t1 = std::fabs(ms1.b_cplx.real()); 

        //for (size_t y(0); y < this -> engines.size(); ++y){
        //    coef_t ms2 = engines[y] -> mass_line(this -> param -> mass_w, this -> param -> mass_t); 
        //    long double z2 = std::fabs(ms2.a_cplx.real());
        //    long double t2 = std::fabs(ms2.b_cplx.real()); 
    
        //    cnx -> intersection(this -> engines[y], metx, mety, t1, z1, t2, z2); 
        //}
    }
    abort(); 
}







