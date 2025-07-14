#include "tools.h"
#include "matrix.h"
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

double** unit(){
    double** m = matrix(3, 3);
    m[0][0] = 1; m[1][1] = 1; m[2][2] = -1;
    return m; 
}

double** smatx(double px, double py, double pz){
    double** o = matrix(3, 3);
    o[0][0] = -1; o[1][1] = -1;
    o[0][2] = px; o[1][2] = py; 
    o[2][2] = pz+1; 
    return o; 
}

void multisqrt(double y, double roots[2], int *count){
    *count = 0;
    if (y < 0) return;
    if (!fabs(y)){roots[0] = 0; *count = 1; return;}
    double r = pow(y, 0.5);
    roots[0] = -r; roots[1] = r;
    *count = 2;
}

void swap_index(double** v, int idx){
    double tmp = v[idx][0];  
    v[idx][0] = v[idx][1]; 
    v[idx][1] = tmp; 
}

void print(double** mx, int prec, int w){
    std::cout << std::fixed << std::setprecision(prec); 
    std::cout << std::setw(w) << mx[0][0] << " " << std::setw(w) << mx[0][1] << " " << std::setw(w) << mx[0][2] << "\n";
    std::cout << std::setw(w) << mx[1][0] << " " << std::setw(w) << mx[1][1] << " " << std::setw(w) << mx[1][2] << "\n";
    std::cout << std::setw(w) << mx[2][0] << " " << std::setw(w) << mx[2][1] << " " << std::setw(w) << mx[2][2] << "\n";
    std::cout << std::endl;
}


void print_(double** mx, int row, int col, int prec, int w){
    std::cout << std::fixed << std::setprecision(prec); 
    for (int x(0); x < row; ++x){
        for (int y(0); y < col; ++y){std::cout << std::setw(w) << mx[x][y] << " ";}
        std::cout << "\n"; 
    }
    std::cout << std::endl;
}

int fill_real(int idx, std::complex<double>* v, double** m){
if (!v){return idx;}
    double r = v -> real(); 
    double i = fabs(v -> imag()) > 1e-9; 
    for (int x(0); x < idx; ++x){
        if (m[0][x] != r && m[0][x] != 66){continue;}
        return idx;
    }
    m[0][idx] = r; m[1][idx] = i; 
    return ++idx; 
}

double** fill_real(int* sx, 
        std::complex<double>* v1 = nullptr, std::complex<double>* v2 = nullptr, 
        std::complex<double>* v3 = nullptr, std::complex<double>* v4 = nullptr
){
    int s = (v1 != nullptr) + (v2 != nullptr) + (v3 != nullptr) + (v4 != nullptr); 
    double** md = matrix(2, s); 
    for (int i(0); i < s; ++i){md[0][i] = 66; md[1][i] = 1;}
    int idx = fill_real(0, v1, md); 
    idx = fill_real(idx, v2, md); 
    idx = fill_real(idx, v3, md); 
    idx = fill_real(idx, v4, md); 
    *sx = s; 
    return md;
}

double** find_roots(double a, double b, double c, int* sx){
    double dsx = b * b - 4 * a * c;
    double** sol = matrix(2, 2); 
    sol[1][0] = 1; sol[1][2] = 1; *sx = 2; 
    if (dsx < -1e-12){return sol;}
    if (fabs(dsx) < 1e-12) {
        sol[1][0] = 0; sol[0][0] = -b / (2 * a);
        return sol; 
    }
    double sdsx = pow(dsx, 0.5);
    sol[1][0] = 0; sol[1][1] = 0; 
    sol[0][0] = (-b + sdsx)/(2*a); 
    sol[0][1] = (-b - sdsx)/(2*a); 
    return sol; 
}

double** find_roots(double a, double b, double c, double d, int* sx){
    if (!fabs(a)){return find_roots(b, c, d, sx);}
    b = b/a; c = c/a; d = d/a; a = 1.0; 
    double p     = (3.0 * a * c - b * b) / (3.0 * a * a);
    double q     = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    double delt  = (q * q / 4.0) + (p * p * p / 27.0);
    const double shift = b / (3.0 * a);
    const double cb    = 1.0 / 3.0; 

    std::complex<double> root1, root2, root3;
    std::complex<double> ui = std::complex<double>(0.0, 1.0); 

    if (delt >= 0) {
        delt = (delt > 0) ? pow(delt, 0.5) : 0.0; 
        std::complex<double> u = pow(std::complex<double>(-q / 2.0 + delt), cb);
        std::complex<double> v = pow(std::complex<double>(-q / 2.0 - delt), cb);
        root1 = u + v;
        u = u - v; 

        const double rt3   = pow(3.0, 0.5) / 2.0; 
        root2 = -0.5 * root1 + std::complex<double>(ui * rt3) * u;
        root3 = -0.5 * root1 - std::complex<double>(ui * rt3) * u;
    } 
    else { 
        std::complex<double> r   = pow(-(p * p * p) / 27.0, 0.5);
        std::complex<double> phi = acos(-q / (2.0 * r));
        r = 2.0*pow(r, cb); 
        root1 = r * cos( phi * cb);
        root2 = r * cos((phi + 2.0 * M_PI) * cb);
        root3 = r * cos((phi + 4.0 * M_PI) * cb);
    }
    root1 = root1 - shift; root2 = root2 - shift; root3 = root3 - shift; 
    return fill_real(sx, &root1, &root2, &root3); 
}

double** find_roots(double a, double b, double c, double d, double e, int* sx){
    if (!fabs(a)){return find_roots(b, c, d, e, sx);}
    a = 1.0 / a; b = b*a; c = c*a; d = d*a; e = e*a; a = 1.0; 

    double r = e - b * d / 4 + b * b * c / 16 - 3 * b * b * b * b / 256;
    double q = d + b * b * b / 8 - b * c / 2;
    double p = c - 3 * b * b / 8;
    double s = b / 4.0; 
 
    // row 0: real, row 1: imag 
    double** sol = nullptr; 
    if (fabs(q) < 1e-12){
        std::complex<double> disc = p * p - 4 * r; 
        std::complex<double> z0 = pow((-p + pow(disc, 0.5)) / 2.0, 0.5);
        std::complex<double> z1 = pow((-p - pow(disc, 0.5)) / 2.0, 0.5); 
        std::complex<double> s1 = ( z0 - s); 
        std::complex<double> s2 = (-z0 - s); 
        std::complex<double> s3 = ( z1 - s); 
        std::complex<double> s4 = (-z1 - s); 
        sol = fill_real(sx, &s1, &s2, &s3, &s4); 
        return sol;
    }

    double** sl = find_roots(1.0, 2.0*p, p*p-4.0*r, -q*q, sx); 
    std::complex<double> w0 = std::complex(sl[0][0], sl[1][0]); 
    std::complex<double> r0 = pow(w0, 0.5); 
    std::complex<double> d1 = pow(r0*r0 - 2.0*(p + w0 - q / r0), 0.5); 
    std::complex<double> d2 = pow(r0*r0 - 2.0*(p + w0 + q / r0), 0.5); 

    std::complex<double> s1 = (-r0 + d1)/2.0 - s; 
    std::complex<double> s2 = (-r0 - d1)/2.0 - s; 
    std::complex<double> s3 = ( r0 + d2)/2.0 - s; 
    std::complex<double> s4 = ( r0 - d2)/2.0 - s; 
    sol = fill_real(sx, &s1, &s2, &s3, &s4); 
    clear(sl, 2, *sx); 
    return sol; 
}

double** get_intersection_angle(double** H1, double** H2, double** MET, int* n_sols){
    auto lamb = [](
        double** h1, double** h2, double** met, double** solx, double k1, double k2, 
        double lr, double v, double det, double tol, int ix) -> int
    {
        double x_ = k1 - (h2[0][0] * lr + h2[0][1] * v); 
        double y_ = k2 - (h2[1][0] * lr + h2[1][1] * v); 
        
        double ct = ( h1[1][1] * x_ - h1[0][1] * y_)/det; 
        double st = (-h1[1][0] * x_ + h1[0][0] * y_)/det; 

        double q1 = - met[0][0]; 
        q1 += h1[0][0] * ct + h1[0][1] * st + h1[0][2]; 
        q1 += h2[0][0] * lr + h2[0][1] * v  + h2[0][2]; 

        double q2 = - met[0][1]; 
        q2 += h1[1][0] * ct + h1[1][1] * st + h1[1][2]; 
        q2 += h2[1][0] * lr + h2[1][1] * v  + h2[1][2]; 

        if (fabs(q1) >= tol || fabs(q2) >= tol){return ix;}
        solx[ix][0] = std::atan2(st, ct); solx[ix][1] = std::atan2(v, lr); 
        return ++ix; 
    }; 

    double k1 = MET[0][0] - H1[0][2] - H2[0][2];
    double k2 = MET[0][1] - H1[1][2] - H2[1][2];
    double _det = m_22(H1); 

    const double tol = 1e-12; 
    if (fabs(_det) < tol){*n_sols = 0; return nullptr;}
    
    double p1 =  H1[1][1] * k1 - H1[0][1] * k2;
    double p2 = -H1[1][0] * k1 + H1[0][0] * k2;

    double q1 =  H1[1][1] * H2[0][0] - H1[0][1] * H2[1][0];
    double q2 = -H1[1][0] * H2[0][0] + H1[0][0] * H2[1][0];

    double r1 =  H1[1][1] * H2[0][1] - H1[0][1] * H2[1][1];
    double r2 = -H1[1][0] * H2[0][1] + H1[0][0] * H2[1][1];


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
   
    int sx = 0; 
    double** roots = find_roots(a4, a3, a2, a1, a0, &sx);
    double** solx = matrix(sx, 2);  

    int sl = 0; 
    for (int i(0); i < sx; ++i){
        if (fabs(roots[0][i]) == 66){continue;}
        double lr = roots[0][i]; 
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
    clear(roots, 2, sx);
    return _copy(matrix(sl, 2), solx, sl, 2, true);
}

double** make_ellipse(double** H, double angle){
    double** v = matrix(1, 3); 
    v[0][0] = H[0][0]*std::cos(angle) + H[0][1]*std::sin(angle) + H[0][2]; 
    v[0][1] = H[1][0]*std::cos(angle) + H[1][1]*std::sin(angle) + H[1][2]; 
    v[0][2] = H[2][0]*std::cos(angle) + H[2][1]*std::sin(angle) + H[2][2]; 
    return v; 
}

double distance(double** H1, double a1, double** H2, double a2){
    double** d1 = make_ellipse(H1, a1); 
    double** d2 = make_ellipse(H2, a2); 
    double d = 0.0; 
    for (int i(0); i < 3; ++i){d += pow(d1[0][i] - d2[0][i], 2);}
    clear(d1, 1, 3); clear(d2, 1, 3); 
    return pow(d, 0.5); 
}
