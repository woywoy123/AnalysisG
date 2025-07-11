#include "matrix.h"
#include <complex.h>



// Fast square root approximation using inverse square root method
static inline double f_sqrt(double x) {
    if (x <= 0.0){return 0.0;}
    union {double d; int64_t i;} u;

    u.d = x;
    const int64_t magic = 0x5fe6eb50c7b537a9;
    u.i = magic - (u.i >> 1); 
    double y = u.d;
    return y * (1.5 - 0.5 * x * y * y); 
}

// Fast cube root approximation using integer division and Newton-Raphson
static inline double f_crt(double x) {
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

double** matrix(int row, int col){
    double** mx = (double**)malloc(row*sizeof(double*));
    for (int x(0); x < row; ++x){mx[x] = (double*)malloc(col*sizeof(double));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = 0;}}
    return mx;  
}

double** dot(double** v1, double** v2, int r1, int c1, int r2, int c2){
    double** vo = matrix(r1, c2); 
    for (int x(0); x < r1; ++x){
        for (int j(0); j < c2; ++j){
            double sm = 0; 
            for (int y(0); y < r2; ++y){sm += v1[x][y] * v2[y][j];}
            vo[x][j] = sm; 
        }
    }
    return vo; 
}

double** T(double** v1, int r, int c){
    double** vo = matrix(r, c); 
    for (int x(0); x < r; ++x){for (int y(0); y < c; ++y){vo[y][x] = v1[x][y];}}
    return vo; 
}


double** scale(double** v, double s){
    double** o = matrix(3, 3); 
    for (int x(0); x < 3; ++x){for (int y(0); y < 3; ++y){o[x][y] = v[x][y]*s;}}
    return o; 
}


double** arith(double** v1, double** v2, double s){
    double** o = matrix(3, 3); 
    for (int x(0); x < 3; ++x){for (int y(0); y < 3; ++y){o[x][y] = v1[x][y] + s*v2[x][y];}}
    return o; 
}


double abcd(double a, double b, double c, double d){return a*d - b*c;}
double m_00(double** M){return M[1][1] * M[2][2] - M[1][2] * M[2][1];}
double m_01(double** M){return M[1][0] * M[2][2] - M[1][2] * M[2][0];}
double m_02(double** M){return M[1][0] * M[2][1] - M[1][1] * M[2][0];}
double m_10(double** M){return M[0][1] * M[2][2] - M[0][2] * M[2][1];}
double m_11(double** M){return M[0][0] * M[2][2] - M[0][2] * M[2][0];}
double m_12(double** M){return M[0][0] * M[2][1] - M[0][1] * M[2][0];}
double m_20(double** M){return M[0][1] * M[1][2] - M[0][2] * M[1][1];}
double m_21(double** M){return M[0][0] * M[1][2] - M[0][2] * M[1][0];}
double m_22(double** M){return M[0][0] * M[1][1] - M[0][1] * M[1][0];}

double** cof(double** v){
    double** ov = matrix(3, 3); 
    ov[0][0] =  m_00(v); ov[1][0] = -m_10(v); ov[2][0] =  m_20(v);
    ov[0][1] = -m_01(v); ov[1][1] =  m_11(v); ov[2][1] = -m_21(v);
    ov[0][2] =  m_02(v); ov[1][2] = -m_12(v); ov[2][2] =  m_22(v);
    return ov; 
}

double det(double** v){
    return v[0][0] * m_00(v) -v[0][1] * m_01(v) + v[0][2] * m_02(v); 
}

double** inv(double** v){
    double a =  v[0][0] * abcd(v[1][1], v[1][2], v[2][1], v[2][2]); 
    double b = -v[0][1] * abcd(v[1][0], v[1][2], v[2][0], v[2][2]); 
    double c =  v[0][2] * abcd(v[1][0], v[1][1], v[2][0], v[2][1]); 
    double det_ = det(v); //a + b + c; 
    double** o = matrix(3, 3); 
    if (det_ == 0){return o;}
    det_ = 1.0/det_; 

    double** co = cof(v); 
    double** ct = T(co, 3, 3); 
    for (int x(0); x < 3; ++x){
        for (int y(0); y < 3; ++y){o[x][y] = ct[x][y]*det_;}
    }
    clear(co, 3, 3); clear(ct, 3, 3); 
    return o; 
}

double** unit(){
    double** m = matrix(3, 3);
    m[0][0] =  1; m[1][1] =  1; m[2][2] = -1;
    return m; 
}

double** smatx(double px, double py, double pz){
    double** o = matrix(3, 3);
    o[0][0] = -1; o[1][1] = -1;
    o[0][2] = px; o[1][2] = py; 
    o[2][2] = pz+1; 
    return o; 
}


void clear(double** mx, int row, int col){
    for (int x(0); x < row; ++x){free(mx[x]);}
    free(mx); 
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


double** solve_cubic(double a, double b, double c, double d){
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
    } else { 
        std::complex<double> r   = pow(-(p * p * p) / 27.0, 0.5);
        std::complex<double> phi = acos(-q / (2.0 * r));
        r = 2.0*pow(r, cb); 
        root1 = r * cos( phi * cb);
        root2 = r * cos((phi + 2.0 * M_PI) * cb);
        root3 = r * cos((phi + 4.0 * M_PI) * cb);
    }
    root1 = root1 - shift; root2 = root2 - shift; root3 = root3 - shift; 
    double** eg = matrix(2, 3); 
    eg[0][0] = root1.real(); eg[0][1] = root2.real(); eg[0][2] = root3.real();
    eg[1][0] = root1.imag(); eg[1][1] = root2.imag(); eg[1][2] = root3.imag();
    return eg; 
}


double** find_roots(double a, double b, double c){
    double s = a / 3.0; 
    double p = b - a*a / 3.0; 
    double q = c - a*b / 3.0 + (2 * a * a * a) / 27.0; 
    double** sol = matrix(2, 3);
    if (fabs(p) < 1e-12 && fabs(q) < 1e-12){
        sol[0][0] = -s; sol[0][1] = -s; sol[0][2] = -s; 
        return sol; 
    }

    if (fabs(p) < 1e-12){
        std::complex<double> w0 = -q; 
        std::complex<double> om = std::complex<double>(-0.5,  0.5 * pow(3, 0.5)); 
        std::complex<double> o2 = std::complex<double>(-0.5, -0.5 * pow(3, 0.5)); 
        w0 = pow(w0, 1.0/3.0);

        std::complex<double> s1 = w0 - s; 
        std::complex<double> s2 = w0 * om - s; 
        std::complex<double> s3 = w0 * o2 - s; 
        sol[0][0] = s1.real(); sol[0][1] = s2.real(); sol[0][2] = s3.real(); 
        sol[1][0] = s1.imag(); sol[1][1] = s2.imag(); sol[1][2] = s3.imag(); 
        return sol; 
    }
   

    std::complex<double> ls = q*q / 4.0 + p*p*p / 27.0; 
    std::complex<double> u = -q / 2.0 + pow(ls, 0.5); 
    std::complex<double> f = pow(u, 1.0/3.0); 
    std::complex<double> w = f - p / (3.0*f); 
    std::complex<double> ds = w*w - 4.0*(w*w + p); 
    std::complex<double> w1 = (-w + pow(ds, 0.5))/2.0 - s; 
    std::complex<double> w2 = (-w - pow(ds, 0.5))/2.0 - s;
    w = w - s;  
    sol[0][0] = w.real(); sol[0][1] = w1.real(); sol[0][2] = w2.real(); 
    sol[1][0] = w.imag(); sol[1][1] = w1.imag(); sol[1][2] = w2.imag(); 
    return sol; 
} 


double** find_roots(double a, double b, double c, double d, double e){
    a = 1.0/a; b = b*a; c = c*a; d = d*a; e = e*a; a = 1.0; 
    
    double p = c - (3.0 * b*b) / 8.0; 
    double q = d - (b * c) / 2.0 + pow(b, 3) / 8.0; 
    double r = e - (3.0 * b*b*b*b) / 256.0 + (b * b * c) / 16.0 - (b * d)/4.0; 
    double s = b / 4.0; 
   
    // row 0: real, row 1: imag 
    double** sol = matrix(2, 4);  
    if (fabs(q) < 1e-12){
        std::complex<double> disc = p * p - 4 * r; 
        std::complex<double> z0 = pow((-p + pow(disc, 0.5)) / 2.0, 0.5);
        std::complex<double> z1 = pow((-p - pow(disc, 0.5)) / 2.0, 0.5); 
        std::complex<double> s1 = ( z0 - s); 
        std::complex<double> s2 = (-z0 - s); 
        std::complex<double> s3 = ( z1 - s); 
        std::complex<double> s4 = (-z1 - s); 
        sol[0][0] = s1.real(); sol[0][1] = s2.real(); sol[0][2] = s3.real(); sol[0][3] = s4.real(); 
        sol[1][0] = s1.imag(); sol[1][1] = s2.imag(); sol[1][2] = s3.imag(); sol[1][3] = s4.imag(); 
        return sol; 
    }

    double** sl = find_roots(2.0*p, p*p-4.0*r, -q*q); 
    std::complex<double> w0 = std::complex(sl[0][0], sl[1][0]); 
    std::complex<double> r0 = pow(w0, 0.5); 
    std::complex<double> d1 = pow(r0*r0 - 2.0*(p + w0 - q / r0), 0.5); 
    std::complex<double> d2 = pow(r0*r0 - 2.0*(p + w0 + q / r0), 0.5); 

    std::complex<double> s1 = (-r0 + d1)/2.0 - s; 
    std::complex<double> s2 = (-r0 - d1)/2.0 - s; 
    std::complex<double> s3 = ( r0 + d2)/2.0 - s; 
    std::complex<double> s4 = ( r0 - d2)/2.0 - s; 

    sol[0][0] = s1.real(); sol[0][1] = s2.real(); sol[0][2] = s3.real(); sol[0][3] = s4.real(); 
    sol[1][0] = s1.imag(); sol[1][1] = s2.imag(); sol[1][2] = s3.imag(); sol[1][3] = s4.imag(); 
    return sol; 
}

double** get_intersection_angle(double** H1, double** H2){
    auto solx =[](
            double r, double i, 
            double** r0, double** r1, double** r2, double** r3,
            double w00, double w01, double w10, double w11, 
            double h10, double h11, double h12, 
            double h20, double h21, double h22,
            double r00, double r11
    ) -> bool {
        if (-1 > r || r > 1 || i){return false;}
        double x = 1 - r*r;
        double vp = pow(x, 0.5); 
        double vn = -vp; 

        double x1p = w00*r + w01*vp + r00; 
        double y1p = w10*r + w11*vp + r11; 

        double x1n = w00*r + w01*vn + r00; 
        double y1n = w10*r + w11*vn + r11; 
        
        bool p = fabs(x1p*x1p + y1p*y1p - 1) < 1e-6;
        bool n = fabs(x1n*x1n + y1n*y1n - 1) < 1e-6;
        p *= fabs(h10*x1p + h11*y1p + h12 - (h20*r + h21*vp + h22)) < 1e-6; 
        n *= fabs(h10*x1n + h11*y1n + h12 - (h20*r + h21*vn + h22)) < 1e-6; 
        if (!p && !n){return false;}
        double a1 = std::atan2(y1p, x1p); 
        double a3 = std::atan2(y1n, x1n); 
        double a2 = std::atan2(vp, r); 
        double a4 = std::atan2(vn, r);  
        if (p){*r0 = new double(a1); *r1 = new double(a2);}
        if (n){*r2 = new double(a3); *r3 = new double(a4);} 
        return true; 
    }; 



    double d1 = H2[0][2] - H1[0][2]; 
    double d2 = H2[1][2] - H1[1][2]; 

    double x1 = H1[0][0]*H1[1][1] - H1[0][1]*H1[1][0]; 
    double x2 = H2[0][0]*H2[1][1] - H2[0][1]*H2[1][0]; 
    if (fabs(x1) < 1e-6 || fabs(x2) < 1e-6){return nullptr;}
    x1 = 1.0/x1; x2 = 1.0/x2; 
    double h1_00 =  H1[1][1]*x1; double h1_01 = -H1[0][1]*x1; 
    double h1_10 = -H1[1][0]*x1; double h1_11 =  H1[0][0]*x1; 
    double w00 = h1_00*H2[0][0] + h1_01 * H2[1][0]; double w01 = h1_00*H2[0][1] + h1_01 * H2[1][1]; 
    double w10 = h1_10*H2[0][0] + h1_11 * H2[1][0]; double w11 = h1_10*H2[0][1] + h1_11 * H2[1][1]; 
    double r00 = h1_00*d1 + h1_01*d2;               double r11 = h1_10*d1 + h1_11*d2; 
    double p  = w01*w01 + w11*w11 - w00*w00 - w10*w10; 
    double tv = r00*r00 + r11*r11 - 1 + w00*w00 + w10*w10; 

    double q  = 2*(w00*w01 + w10*w11); 
    double rv = 2*(w00*r00 + w10*r11); 
    double sv = 2*(w01*r00 + w11*r11); 

    double a = p*p + q*q;
    double b = 2*(q*sv - p*rv); 
    double c = sv*sv - 2*p*(p+tv) + rv*rv - q*q; 
    double d = 2*(rv * (p+tv)-q*sv); 
    double e = pow(p+tv, 2) - sv*sv;
    double** roots = find_roots(a, b, c, d, e); 
    
    double** vou = matrix(8, 2); 
    for (int x(0); x < 4; ++x){
        double* r1 = nullptr; double* r2 = nullptr; 
        double* r3 = nullptr; double* r4 = nullptr; 
        bool p = solx(
            roots[0][x], roots[1][x], &r1, &r2, &r3, &r4, w00, w01, w10, w11, 
            H1[2][0], H1[2][1], H1[2][2], H2[2][0], H2[2][1], H2[2][2], r00, r11
        ); 
        if (r1 && r2){vou[x  ][0] = *r1; vou[x  ][1] = *r2;}
        if (r3 && r4){vou[x+4][0] = *r3; vou[x+4][1] = *r4;}
        if (r1){delete r1;}
        if (r2){delete r2;}
        if (r3){delete r3;}
        if (r4){delete r4;}
    }
    clear(roots, 2, 4);
    return vou; 
}


double find_real_eigenvalue(double** M, double* rx){
    double a = -(M[0][0] + M[1][1] + M[2][2]);
    double b = M[0][0]*M[1][1] + M[0][0]*M[2][2] + M[1][1]*M[2][2] - M[0][1]*M[1][0] - M[0][2]*M[2][0] - M[1][2]*M[2][1];
    double c = -det(M);
    double** r = find_roots(a, b, c);
    for (int i(0); i < 3; ++i){
        if (fabs(r[1][i])){continue;} 
        *rx = r[0][i];
        clear(r, 2, 3); 
        return true;
    }
    clear(r, 2, 3);
    return false;
}


void multisqrt(double y, double roots[2], int *count) {
    *count = 0;
    if (y < 0) return;
    if (fabs(y) < 0) {roots[0] = 0; *count = 1; return;}
    double r = pow(y, 0.5);
    roots[0] = -r; roots[1] = r;
    *count = 2;
}

void swap_index(double** v, int idx){
    double tmp = v[idx][0];  
    v[idx][0] = v[idx][1]; 
    v[idx][1] = tmp; 
}

void factor_degenerate(double** G, double** lines, int* lc, double* q0) {
    if (fabs(G[0][0]) == 0 && fabs(G[1][1]) == 0) {
        lines[0][0] = G[0][1]; lines[0][1] = 0;       lines[0][2] = G[1][2];
        lines[1][0] = 0;       lines[1][1] = G[0][1]; lines[1][2] = G[0][2] - G[1][2];
        *lc = 2; *q0 = 0;
        return;
    }
    int swapxy = (fabs(G[0][0]) > fabs(G[1][1]));
    double** Q = scale(G, 1); 
    for (int i(0); i < 3*swapxy; i++){
        double tmp = Q[0][i];
        Q[0][i] = Q[1][i];
        Q[1][i] = tmp;
    }
    for (int j(0); j < 3*swapxy; j++){swap_index(Q, j);}
    double** Q_ = scale(Q, 1.0/Q[1][1]);
    double** D_ = cof(Q_);
    clear(Q, 3, 3);
    double q22 = -D_[2][2]; 

    int r_count;
    double r[2];
    if (q22 < 0){
        multisqrt(-D_[0][0], r, &r_count);
        for (int i(0); i < r_count; ++i, ++(*lc)) {
            lines[*lc][0] = Q_[0][1]; 
            lines[*lc][1] = Q_[1][1];
            lines[*lc][2] = Q_[1][2] + r[i];
            if (!swapxy){continue;}
            swap_index(lines, *lc); 
        }
    } 
    else {
        double x0 = D_[0][2] / -q22; double y0 = D_[1][2] / -q22;
        multisqrt(q22, r, &r_count);
        for (int i(0); i < r_count; ++i, ++(*lc)) {
            lines[*lc][0] =  Q_[0][1] + r[i];
            lines[*lc][1] =  Q_[1][1], 
            lines[*lc][2] = -Q_[1][1]*y0 - (Q_[0][1] + r[i])*x0;
            if (!swapxy){continue;}
            swap_index(lines, *lc); 
        }
    }
    clear(D_, 3, 3); 
    clear(Q_, 3, 3);  
    *q0 = q22;
}


double trace(double** A){return A[0][0] + A[1][1] + A[2][2];}


double** get_eigen(double** A){
    double tr  = -trace(A); 
    double m11 = A[1][1] * A[2][2] - A[1][2] * A[2][1];
    double m22 = A[0][0] * A[2][2] - A[0][2] * A[2][0];
    double m33 = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    double q   = m11 + m22 + m33; 
    double delta = -A[0][0] * m11 + A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) - A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    double** eig = solve_cubic(1.0, tr, q, delta); 
    double** eiv = matrix(3, 4); 
    for (int i(0); i < 3; ++i){
        if (fabs(eig[1][i]) >= 1e-9){continue;}
        eiv[i][3] = 1; 

        double B[3][3] = {{0}}; 
        for (int j(0); j < 3; ++j){ for (int k(0); k < 3; ++k){B[j][k] = A[j][k];}}
        for (int j(0); j < 3; ++j){ B[j][j] -= eig[0][i];}
        double r1[3] = {B[0][0], B[0][1], B[0][2]};
        double r2[3] = {B[1][0], B[1][1], B[1][2]};

        eiv[i][0] = r1[1] * r2[2] - r1[2] * r2[1];
        eiv[i][1] = r1[2] * r2[0] - r1[0] * r2[2];
        eiv[i][2] = r1[0] * r2[1] - r1[1] * r2[0];

        double mag = pow(eiv[i][0]*eiv[i][0] + eiv[i][1]*eiv[i][1] + eiv[i][2]*eiv[i][2], 0.5);
        if (mag < 1e-9){
            double r3[3] = {B[2][0], B[2][1], B[2][2]};
            eiv[i][0] = r1[1] * r3[2] - r1[2] * r3[1];
            eiv[i][1] = r1[2] * r3[0] - r1[0] * r3[2];
            eiv[i][2] = r1[0] * r3[1] - r1[1] * r3[0];
        }
        mag = pow(eiv[i][0]*eiv[i][0] + eiv[i][1]*eiv[i][1] + eiv[i][2]*eiv[i][2], 0.5);
        if (mag > 1e-9){eiv[i][0] = eiv[i][0]/mag; eiv[i][1] = eiv[i][1]/mag; eiv[i][2] = eiv[i][2]/mag;}
    }
    clear(eig, 2, 3); 
    return eiv; 
}

int intersections_ellipse_line(double** ellipse, double* line, double** pts){
    double** C = matrix(3, 3);
    for (int i(0); i < 3; ++i){
        C[0][i] = line[1] * ellipse[2][i] - line[2] * ellipse[1][i];
        C[1][i] = line[2] * ellipse[0][i] - line[0] * ellipse[2][i];
        C[2][i] = line[0] * ellipse[1][i] - line[1] * ellipse[0][i];
    }
    double** eign = get_eigen(C);

    int pt = 0;
    clear(C, 3, 3);
    for (int i(0); i < 3; ++i){
        if (!eign[i][3] || !eign[i][2]){continue;}
        double** v = matrix(3, 1); 
        for (int j(0); j < 3; ++j){v[j][0] = eign[i][j];}
        double** l1 = dot(&line, v, 1, 3, 3, 1); 
        double** l2 = dot(ellipse, v, 3, 3, 3, 1);
        double** lt = T(l2, 3, 1);
        double** l3 = dot(lt, v, 1, 3, 3, 1); 
        for (int j(0); j < 3; ++j){pts[pt][j] = eign[i][j]/eign[i][2];}
        pts[pt][3] = (std::log10(pow(l3[0][0], 2) + pow(l1[0][0], 2)));  
        clear(l1, 1, 1); clear(l2, 3, 1); clear(lt, 1, 3); clear(l3, 1, 1);
        pts[pt][4] = 1.0; ++pt; 
    }
    clear(eign, 2, 4); 
    return pt;
}


void intersection_ellipses(double** A, double** B, double eps){
    double** A_ = nullptr; double** B_ = nullptr; 
    bool swp = fabs(det(B)) > fabs(det(A)); 
    if (swp){A_ = B; B_ = A;}
    else {A_ = A; B_ = B;}
    
    double** AT = inv(A_);
    double** t  = dot(AT, B_); 
    clear(AT, 3, 3);
    double e = 0; 
    if (!find_real_eigenvalue(t, &e)){return;}
    clear(t, 3, 3);   
 
    int lc = 0; double q0;  
    double** line = matrix(2, 3); 
    double** G = arith(B_, A_, -e); 
    factor_degenerate(G, line, &lc, &q0); 
    clear(G, 3, 3); 

    eps = std::log10(eps);  
    double** all_points = matrix(4, 3); 
    for (int i(0); i < lc; ++i){
        int lx = 0; 
        double   sml = 0; 
        double** pts = matrix(3, 5);
        std::cout << "+++++++++++++" << std::endl;
        for (int j(0); j < intersections_ellipse_line(A_, line[i], pts); ++j){
            if (!pts[j][4] || pts[j][3] > eps){continue;}
            //bool lf = (!sml || pts[j][3] < sml); 
            //for (int k(0); k < 3*lf; ++k){all_points[lx][k] = pts[k][k];}
            //sml = (lf) ? pts[j][3] : sml; lx += lf; 
        }
        print_(pts, 3, 5); 
    }
}



