#include "tools.h"
#include "matrix.h"
#include "mtx.h"

#include <complex.h>

double f_sqrt(double x) {
    if (x <= 0.0){return 0.0;}
    union {double d; int64_t i;} u;

    u.d = x;
    const int64_t magic = 0x5fe6eb50c7b537a9;
    u.i = magic - (u.i >> 1); 
    double y = u.d;
    return y * (1.5 - 0.5 * x * y * y); 
}

double f_crt(double x) {
    if (x == 0.0){return 0.0;}
    int sign = 1;
    if (x < 0) {sign = -1; x = -x;}
    union {double d; int64_t i;} u;

    u.d = x;
    const int64_t magic = 0x2a9f84fe36e9aa00;
    u.i = u.i / 3 + magic; // Initial guess
    double y = u.d;
    return sign * (2.0 * y + x / (y * y)) * (1.0/3.0);
}

double costheta(particle* p1, particle* p2){
    double pxx  = p1 -> px * p2 -> px + p1 -> py * p2 -> py + p1 -> pz * p2 -> pz; 
    return pxx / pow(p1 -> p2() * p2 -> p2(), 0.5); 
}

double sintheta(particle* p1, particle* p2){return pow(1 - pow(costheta(p1, p2), 2), 0.5);}




mtx* unit(){
    mtx* m = new mtx(3, 3); 
    m -> _m[0][0] = 1; 
    m -> _m[1][1] = 1; 
    m -> _m[2][2] = -1; 
    return m; 
}

mtx* smatx(double px, double py, double pz){
    mtx* o = new mtx(3, 3); 
    o -> _m[0][0] = -1; 
    o -> _m[0][2] = px; 
    o -> _m[1][1] = -1;
    o -> _m[1][2] = py; 
    o -> _m[2][2] = pz+1; 
    return o; 
}


mtx* get_intersection_angle(mtx* H1, mtx* H2, mtx* MET, int* n_sols){
    auto lamb = [](
        mtx* h1, mtx* h2, mtx* met, mtx* solx, double k1, double k2, 
        double lr, double v, double det, double tol, int ix) -> int
    {
        double x_ = k1 - (h2 -> _m[0][0] * lr + h2 -> _m[0][1] * v); 
        double y_ = k2 - (h2 -> _m[1][0] * lr + h2 -> _m[1][1] * v); 
        
        double ct = ( h1 -> _m[1][1] * x_ - h1 -> _m[0][1] * y_)/det; 
        double st = (-h1 -> _m[1][0] * x_ + h1 -> _m[0][0] * y_)/det; 

        double q1 = -met -> _m[0][0]; 
        q1 += h1 -> _m[0][0] * ct + h1 -> _m[0][1] * st + h1 -> _m[0][2]; 
        q1 += h2 -> _m[0][0] * lr + h2 -> _m[0][1] * v  + h2 -> _m[0][2]; 

        double q2 = - met -> _m[0][1]; 
        q2 += h1 -> _m[1][0] * ct + h1 -> _m[1][1] * st + h1 -> _m[1][2]; 
        q2 += h2 -> _m[1][0] * lr + h2 -> _m[1][1] * v  + h2 -> _m[1][2]; 

        if (fabs(q1) >= tol || fabs(q2) >= tol){return ix;}
        solx -> assign(ix, 0, std::atan2(st, ct)); 
        solx -> assign(ix, 1, std::atan2( v, lr)); 
        return ++ix; 
    }; 

    double k1 = MET -> _m[0][0] - H1 -> _m[0][2] - H2 -> _m[0][2];
    double k2 = MET -> _m[0][1] - H1 -> _m[1][2] - H2 -> _m[1][2];
    double _det = H1 -> m_22(); 

    const double tol = 1e-12; 
    if (fabs(_det) < tol){*n_sols = 0; return nullptr;}
    double p1 =  H1 -> _m[1][1] * k1 - H1 -> _m[0][1] * k2;
    double p2 = -H1 -> _m[1][0] * k1 + H1 -> _m[0][0] * k2;
    double q1 =  H1 -> _m[1][1] * H2 -> _m[0][0] - H1 -> _m[0][1] * H2 -> _m[1][0];
    double q2 = -H1 -> _m[1][0] * H2 -> _m[0][0] + H1 -> _m[0][0] * H2 -> _m[1][0];
    double r1 =  H1 -> _m[1][1] * H2 -> _m[0][1] - H1 -> _m[0][1] * H2 -> _m[1][1];
    double r2 = -H1 -> _m[1][0] * H2 -> _m[0][1] + H1 -> _m[0][0] * H2 -> _m[1][1];

    double a = q1 * q1 + q2 * q2;
    double b = r1 * r1 + r2 * r2;
    double c = q1 * r1 + q2 * r2;
    double d = -2 * (p1 * q1 + p2 * q2);
    double e = -2 * (p1 * r1 + p2 * r2);

    double A = a - b;
    double C = b + p1 * p1 + p2 * p2 - _det * _det;

    double a4 = A * A + 4.0 * c * c;
    double a3 = 2.0 * (A * d + 2.0 * c * e);
    double a2 = 2.0 *  A * C + d * d + e * e - 4.0 * c * c;
    double a1 = 2.0 * (d * C - 2.0 * c * e);
    double a0 = C * C - e * e;
    int sl = 0; 

    mtx* roots = find_roots(a4, a3, a2, a1, a0);
    int sx = roots -> dim_j; 

    mtx* solx = new mtx(sx, 2); 
    for (int i(0); i < sx; ++i){
        if (!roots -> valid(0, i)){continue;}
        double lr = roots -> _m[0][i]; 
        double dn = 2 * c * lr + e; 
        if (fabs(dn) < tol){
            double v = 1 - lr * lr;   
            if (v < 0){continue;}
            v = pow(v, 0.5); 
            sl = lamb(H1, H2, MET, solx, k1, k2, lr,  v, _det, tol, sl); 
            sl = lamb(H1, H2, MET, solx, k1, k2, lr, -v, _det, tol, sl); 
            if (sl < sx){continue;}
            break;
        }
        double v = -(A*lr*lr + d * lr + C) / dn; 
        sl = lamb(H1, H2, MET, solx, k1, k2, lr,  v, _det, tol, sl); 
        if (sl < sx){continue;}
        break;
    }
    *n_sols = sl;
    delete roots; 
    roots = new mtx(sl, 2); 
    for (int x(0); x < sl; ++x){roots -> copy(solx, x, 2);}
    delete solx; 
    return roots; 
}

mtx make_ellipse(mtx* H, double angle){
    mtx vt(3, 1); 
    vt.assign(0, 0, std::cos(angle)); 
    vt.assign(1, 0, std::sin(angle)); 
    vt.assign(2, 0, 1.0); 
    return H -> dot(vt).T(); 
}

double distance(mtx* H1, double a1, mtx* H2, double a2){
    mtx dx = make_ellipse(H1, a1) - make_ellipse(H2, a2); 
    dx = dx*dx; 
    double d = 0.0; 
    for (int i(0); i < 3; ++i){d += dx._m[0][i];}
    return pow(d, 0.5); 
}
