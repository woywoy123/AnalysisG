#include <templates/solvers.h>
#include <templates/mtx.h>
#include <iostream>
#include <complex>
#include <cmath>

mtx* find_roots(double a, double b, double c){
    mtx* sol = new mtx(2, 2); 
    double disc = b * b - 4 * a * c;
    if (disc < -sol -> tol){return sol;}
    if (fabs(disc) < sol -> tol) {
        sol -> assign(0, 0, -b / (2 * a));
        return sol;
    }
    disc = std::pow(disc, 0.5);
    sol -> assign(0, 0, (-b + disc) / (2 * a));
    sol -> assign(0, 1, (-b - disc) / (2 * a));
    return sol; 
}

mtx* find_roots(double a, double b, double c, double d){
    mtx* sol = new mtx(2, 3); 
    if (fabs(a) < sol -> tol){
        if (fabs(b) >= sol -> tol){delete sol; return find_roots(b, c, d);}
        if (fabs(c) < sol -> tol ){return sol;}
        sol -> assign(0, 0, -d / c); 
        return sol; 
    }
    
    double b2 = b * b;
    double p = (3 * a * c - b2) / (3 * a * a);
    double q = (2 * b2 * b - 9 * a * b * c + 27 * a * a * d) / (27 * a * a * a);
    double p3 = p * p * p;
    double disc = q * q / 4 + p3 / 27;
    
    if (disc > sol -> tol) {
        disc = std::pow(disc, 0.5);
        double u = std::cbrt(-q / 2 + disc);
        double v = std::cbrt(-q / 2 - disc);
        sol -> assign(0, 0, u + v - b / (3 * a));
        return sol; 
    } 
    if (disc < -sol -> tol) {
        double om = std::acos(3 * q * std::pow(-3 / p3, 0.5) / (2 * p));
        double r = 2 * std::pow(-p / 3, 0.5);
        sol -> assign(0, 0, r * std::cos(om / 3) - b / (3 * a)); 
        sol -> assign(0, 1, r * std::cos((om + 2 * M_PI) / 3) - b / (3 * a)); 
        sol -> assign(0, 2, r * std::cos((om + 4 * M_PI) / 3) - b / (3 * a)); 
        return sol; 
    } 
    double u = std::cbrt(-q / 2);
    sol -> assign(0, 0, 2 * u - b / (3 * a)); 
    sol -> assign(0, 1,    -u - b / (3 * a)); 
    return sol; 
}

mtx* find_roots(double a, double b, double c, double d, double e){
    mtx* sol = new mtx(2, 4); 
    if (fabs(a) < sol -> tol){delete sol; return find_roots(b, c, d, e);}
    b = b / a; c = c / a; d = d / a; e = e / a;
    
    double p = c - 3 * b * b / 8;
    double q = d + b * b * b / 8 - b * c / 2;
    double r = e - b * d / 4 + b * b * c / 16 - 3 * b * b * b * b / 256;
  
    int n = -1;  
    if (fabs(q) < sol -> tol){
        mtx* sqx = find_roots(1.0, p, r);
        for (int i(0); i < 2; ++i){
            if (!sqx -> valid(0, i)){continue;}
            if (sqx -> _m[0][i] < 0){continue;}
            double sq = pow(sqx -> _m[0][i], 0.5);
            sol -> assign(0, ++n, sq - b / 4.0); 
            if (sq <= sol -> tol){continue;}
            sol -> assign(0, ++n, -sq - b / 4.0);
        }
        delete sqx; 
        return sol;
    }
    mtx* solc = find_roots(1.0, 2*p, p*p - 4*r, -q*q);
    
    for (int i(0); i < 3; ++i){
        if (!solc -> valid(0, i)){continue;}
        double z = solc -> _m[0][i];
        if (z < -solc -> tol){continue;}
        z = (z < 0) ? 0 : z; 
        double g = pow(z, 0.5);
        if (fabs(2 * g) < solc -> tol){continue;}
        double h = (z + p - q / g) / 2;
        double k = (z + p + q / g) / 2;
        mtx* sol_ = find_roots(1.0, g, h);
        for (int x(0); x < 2; ++x){
            if (!sol_ -> valid(0, x)){continue;}
            sol -> assign(0, ++n, sol_ -> _m[0][x] - b/4.0);
        }
        delete sol_; 

        sol_ = find_roots(1.0, -g, k);
        for (int x(0); x < 2; ++x){
            if (!sol_ -> valid(0, x)){continue;}
            sol -> assign(0, ++n, sol_ -> _m[0][x] - b/4.0);
        }
        delete sol_; 
    }
    delete solc; 
    return sol; 
}

mtx* solve_cubic(double a, double b, double c, double d) {
    std::complex<double> p = (3.0 * a * c - b * b) / (3.0 * a * a);
    std::complex<double> q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    std::complex<double> root1, root2, root3;
    std::complex<double> delt = (q * q / 4.0) + (p * p * p / 27.0);
    double rt3 = std::pow(3.0, 0.5) / 2.0; 
    std::complex<double> ui = std::complex<double>(0.0, 1.0); 
    const double cb = 1.0/3.0; 

    if (delt.real() >= 0) {
        std::complex<double> u_val = std::pow(-q / 2.0 + pow(delt, 0.5), cb);
        std::complex<double> v_val = std::pow(-q / 2.0 - pow(delt, 0.5), cb);
        root1 = u_val + v_val;
        root2 = -0.5 * (u_val + v_val) + std::complex<double>(ui * rt3) * (u_val - v_val);
        root3 = -0.5 * (u_val + v_val) - std::complex<double>(ui * rt3) * (u_val - v_val);
    } else { 
        std::complex<double> r   = std::pow(-(p * p * p) / 27.0, 0.5);
        std::complex<double> phi = std::acos(-q / (2.0 * r));
        root1 = 2.0 * std::pow(r, cb) * std::cos( phi / 3.0);
        root2 = 2.0 * std::pow(r, cb) * std::cos((phi + 2.0 * M_PI) * cb);
        root3 = 2.0 * std::pow(r, cb) * std::cos((phi + 4.0 * M_PI) * cb);
    }

    double shift = b / (3.0 * a);
    root1 = root1 - shift; root2 = root2 - shift; root3 = root3 - shift; 
    mtx* rx_ = new mtx(2, 3); 
    rx_ -> unique(0, 1, root1.real(), root1.imag()); 
    rx_ -> unique(0, 1, root2.real(), root2.imag()); 
    rx_ -> unique(0, 1, root3.real(), root3.imag()); 
    return rx_; 
}

mtx* Rz(double angle){
    mtx* out = new mtx(3, 3); 
    out -> _m[0][0] =  std::cos(angle); 
    out -> _m[0][1] = -std::sin(angle); 
    out -> _m[1][0] =  std::sin(angle); 
    out -> _m[1][1] =  std::cos(angle); 
    out -> _m[2][2] = 1.0;
    return out; 
}

mtx* Ry(double angle){
    mtx* out = new mtx(3, 3); 
    out -> _m[0][0] = std::cos(angle); 
    out -> _m[0][2] = std::sin(angle); 
    out -> _m[1][1] = 1.0; 
    out -> _m[2][0] = -std::sin(angle); 
    out -> _m[2][2] =  std::cos(angle);
    return out; 
}

mtx* Rx(double angle){
    mtx* out = new mtx(3, 3); 
    out -> _m[0][0] = 1.0; 
    out -> _m[1][1] =  std::cos(angle); 
    out -> _m[1][2] = -std::sin(angle); 
    out -> _m[2][1] =  std::sin(angle);
    out -> _m[2][2] =  std::cos(angle); 
    return out; 
}


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

mtx* intersection_angle(mtx* H1, mtx* H2, mtx* MET, int* n_sols){
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

    const double tol = 1e-9; 
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


void swap_index(double** v, int idx){
    double tmp = v[idx][0];  
    v[idx][0] = v[idx][1]; 
    v[idx][1] = tmp; 
}

void multisqrt(double y, double roots[2], int *count){
    *count = 0;
    if (y < 0) return;
    if (!fabs(y)){roots[0] = 0; *count = 1; return;}
    double r = pow(y, 0.5);
    roots[0] = -r; roots[1] = r;
    *count = 2;
}

void factor_degenerate(mtx G, mtx* lines, int* lc, double* q0) {
    if (fabs(G._m[0][0]) == 0 && fabs(G._m[1][1]) == 0) {
        lines -> assign(0, 0, G._m[0][1]); 
        lines -> assign(0, 1, 0);       
        lines -> assign(0, 2, G._m[1][2]);
        lines -> assign(1, 0, 0);       
        lines -> assign(1, 1, G._m[0][1]); 
        lines -> assign(1, 2, G._m[0][2] - G._m[1][2]);
        *lc = 2; *q0 = 0;
        return;
    }
    mtx Q = G.copy(); 
    int swapxy = (fabs(G._m[0][0]) > fabs(G._m[1][1]));
    for (int i(0); i < 3*swapxy; i++){
        double tmp = Q._m[0][i];
        Q.assign(0, i, Q._m[1][i]);
        Q.assign(1, i, tmp);
    }

    for (int j(0); j < 3*swapxy; j++){swap_index(Q._m, j);}
    mtx Q_ = Q * (1.0/Q._m[1][1]);
    mtx D_ = Q_.cof();
    double  q22 = D_._m[2][2]; 
    *q0 = q22;

    int r_count;
    double r[2];
    if (-q22 <= 0){
        multisqrt(-D_._m[0][0], r, &r_count);
        for (int i(0); i < r_count; ++i) {
            lines -> assign(i, 0, Q_._m[0][1]); 
            lines -> assign(i, 1, Q_._m[1][1]);
            lines -> assign(i, 2, Q_._m[1][2] + r[i]);
            if (!swapxy){continue;}
            swap_index(lines -> _m, i); 
        }
        *lc = r_count; 
        return; 
    } 
    double x0 = D_._m[0][2] / q22; 
    double y0 = D_._m[1][2] / q22;
    multisqrt(-q22, r, &r_count);
    for (int i(0); i < r_count; ++i) {
        lines -> assign(i, 0,  Q_._m[0][1] + r[i]); 
        lines -> assign(i, 1,  Q_._m[1][1]); 
        lines -> assign(i, 2, -Q_._m[1][1]*y0 - (Q_._m[0][1] + r[i])*x0);
        if (!swapxy){continue;}
        swap_index(lines -> _m, i); 
    }
    *lc = r_count; 
}

int intersections_ellipse_line(mtx* ellipse, mtx* line, mtx* pts){
    mtx* eign = ellipse -> cross(line).eigenvector(); 
    int pt = 0;
    for (int i(0); i < 3; ++i){
        if (!eign -> valid(i, 1) || !eign -> _m[i][2]){continue;}
        mtx* sl = eign -> slice(i); 
        double z = 1.0/sl -> _m[0][2]; 
        pts -> assign(pt, 0, sl -> _m[0][0]*z); 
        pts -> assign(pt, 1, sl -> _m[0][1]*z);
        pts -> assign(pt, 2, sl -> _m[0][2]*z); 

        mtx v  = sl -> T(); 
        mtx l1 = line -> dot(v);
        mtx l2 = sl -> dot(ellipse).dot(v);
        double vl = 2*std::log10(fabs(l1._m[0][0])) + 2*std::log10(fabs(l2._m[0][0])); 
        pts -> assign(pt, 3, vl); 
        pts -> assign(pt, 4, 1.0); 
        delete sl; ++pt; 
    }
    delete eign; 
    delete line; 
    return pt;
}


int intersection_ellipses(mtx* A, mtx* B, mtx** lines, mtx** pts, mtx** sols){
    bool swp = fabs(B -> det()) > fabs(A -> det()); 
    int lx = -1; double q0 = -1;  
    mtx A_ = (swp) ? B -> copy() : A -> copy(); 
    mtx B_ = (swp) ? A -> copy() : B -> copy(); 
    mtx t  = A_.inv().dot(B_); 

    mtx* ln = nullptr; 
    mtx sol_ = mtx(1, 18); 
    mtx all_ = mtx(18, 3); 
    mtx* ex = t.eigenvalues(); 
    for (int x(0); x < ex -> dim_j; ++x){
        if (!ex -> valid(0, x)){continue;}
        mtx G    = B_ - ex -> _m[0][x]*A_;
        int lc = 0; 
        mtx line = mtx(2, 3); 
        factor_degenerate(G, &line, &lc, &q0); 
        for (int i(0); i < lc; ++i){
            mtx _pts(3, 5); 
            for (int j(0); j < intersections_ellipse_line(&A_, line.slice(i), &_pts); ++j){
                if (!_pts._m[j][4]){continue;}
                if (sol_.unique(0, 0, _pts._m[j][3], _pts._m[j][3])){continue;}
                all_.copy(&_pts, ++lx, j, 3); 
            }
        }
        if (!ln){ln = new mtx(line); continue;}
        mtx* kx = ln -> cat(&line); 
        delete ln; ln = kx; 
    }
    delete ex; 
    if (lx < 0){return 0;}
    *lines = ln; 
    *sols  = new mtx(1, lx); 
    *pts   = new mtx(lx, 3); 
    (*sols) -> copy(&sol_, 0, 0, lx); 
    for (int x(0); x < lx; ++x){(*pts) -> copy(&all_, x);}
    return lx; 
}



double** _matrix(int row, int col){
    double** mx = (double**)malloc(row*sizeof(double*));
    for (int x(0); x < row; ++x){mx[x] = (double*)malloc(col*sizeof(double));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = 0;}}
    return mx;  
}

void _arith(double** o, double** v2, double s, int idx, int idy){
    if (!v2 || !o){return;}
    for (int x(0); x < idx; ++x){
        for (int y(0); y < idy; ++y){o[x][y] = o[x][y] + s*v2[x][y];}
    }
}

void _scale(double** v, double** f, int idx, int idy, double s){
    if (!v || !f){return;}
    for (int x(0); x < idx; ++x){
        for (int y(0); y < idy; ++y){v[x][y] = f[x][y]*s;}
    }
}

bool** _mask(int row, int col){
    bool** mx = (bool**)malloc(row*sizeof(bool*));
    for (int x(0); x < row; ++x){mx[x] = (bool*)malloc(col*sizeof(bool));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = false;}}
    return mx;  
}

void _copy(double* dst, double* src, int lx){
    for (int x(0); x < lx; ++x){dst[x] = src[x];}
}

void _copy(bool* dst, bool* src, int lx){
    for (int x(0); x < lx; ++x){dst[x] = src[x];}
}

void _copy(double** dst, double** src, int lx, int ly){
    for (int x(0); x < lx; ++x){_copy(dst[x], src[x], ly);}
}

void _copy(bool** dst, bool** src, int lx, int ly){
    for (int x(0); x < lx; ++x){_copy(dst[x], src[x], ly);}
}


double _trace(double** A){return A[0][0] + A[1][1] + A[2][2];}
double _m_00(double**  M){return M[1][1] * M[2][2] - M[1][2] * M[2][1];}
double _m_01(double**  M){return M[1][0] * M[2][2] - M[1][2] * M[2][0];}
double _m_02(double**  M){return M[1][0] * M[2][1] - M[1][1] * M[2][0];}
double _m_10(double**  M){return M[0][1] * M[2][2] - M[0][2] * M[2][1];}
double _m_11(double**  M){return M[0][0] * M[2][2] - M[0][2] * M[2][0];}
double _m_12(double**  M){return M[0][0] * M[2][1] - M[0][1] * M[2][0];}
double _m_20(double**  M){return M[0][1] * M[1][2] - M[0][2] * M[1][1];}
double _m_21(double**  M){return M[0][0] * M[1][2] - M[0][2] * M[1][0];}
double _m_22(double**  M){return M[0][0] * M[1][1] - M[0][1] * M[1][0];}
double _det( double**  v){return v[0][0] * _m_00(v) - v[0][1] * _m_01(v) + v[0][2] * _m_02(v);}

mtx operator+(const mtx& o1, const mtx& o2){
    mtx out(o1.dim_i, o1.dim_j); 
    _arith(out._m, o1._m, 1, o1.dim_i, o1.dim_j); 
    _arith(out._m, o2._m, 1, o2.dim_i, o2.dim_j); 
    return out; 
}

mtx operator-(const mtx& o1, const mtx& o2){
    mtx out(o1.dim_i, o1.dim_j);
    _arith(out._m, o1._m,  1, o1.dim_i, o1.dim_j); 
    _arith(out._m, o2._m, -1, o2.dim_i, o2.dim_j); 
    return out; 
}


mtx operator*(const mtx& other, double scale){
    mtx out(other.dim_i, other.dim_j); 
    _scale(out._m, other._m, other.dim_i, other.dim_j, scale); 
    return out; 
}

mtx operator*(double scale, const mtx& other){
    mtx out(other.dim_i, other.dim_j); 
    _scale(out._m, other._m, other.dim_i, other.dim_j, scale); 
    return out; 
}

mtx operator*(const mtx& o1, const mtx& o2){
    mtx out(o1.dim_i, o1.dim_j); 
    for (int x(0); x < o1.dim_i; ++x){
        for (int y(0); y < o1.dim_j; ++y){out.assign(x, y, o1._m[x][y]*o2._m[x][y]);}
    } 
    return out; 
}
