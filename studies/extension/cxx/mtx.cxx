#include "mtx.h"
#include <iostream>
#include <iomanip>
#include <complex>

double mtx::trace(double** A){return A[0][0] + A[1][1] + A[2][2];}
double mtx::m_00(double**  M){return M[1][1] * M[2][2] - M[1][2] * M[2][1];}
double mtx::m_01(double**  M){return M[1][0] * M[2][2] - M[1][2] * M[2][0];}
double mtx::m_02(double**  M){return M[1][0] * M[2][1] - M[1][1] * M[2][0];}
double mtx::m_10(double**  M){return M[0][1] * M[2][2] - M[0][2] * M[2][1];}
double mtx::m_11(double**  M){return M[0][0] * M[2][2] - M[0][2] * M[2][0];}
double mtx::m_12(double**  M){return M[0][0] * M[2][1] - M[0][1] * M[2][0];}
double mtx::m_20(double**  M){return M[0][1] * M[1][2] - M[0][2] * M[1][1];}
double mtx::m_21(double**  M){return M[0][0] * M[1][2] - M[0][2] * M[1][0];}
double mtx::m_22(double**  M){return M[0][0] * M[1][1] - M[0][1] * M[1][0];}
double mtx::m_00(){return m_00(this -> _m);}
double mtx::m_01(){return m_01(this -> _m);}
double mtx::m_02(){return m_02(this -> _m);}
double mtx::m_10(){return m_10(this -> _m);}
double mtx::m_11(){return m_11(this -> _m);}
double mtx::m_12(){return m_12(this -> _m);}
double mtx::m_20(){return m_20(this -> _m);}
double mtx::m_21(){return m_21(this -> _m);}
double mtx::m_22(){return m_22(this -> _m);}
double mtx::det(double**   v){return v[0][0] * this -> m_00(v) - v[0][1] * this -> m_01(v) + v[0][2] * this -> m_02(v);}

double** mtx::_matrix(int row, int col){
    double** mx = (double**)malloc(row*sizeof(double*));
    for (int x(0); x < row; ++x){mx[x] = (double*)malloc(col*sizeof(double));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = 0;}}
    return mx;  
}

static void _arith(double** o, double** v2, double s, int idx, int idy){
    if (!v2 || !o){return;}
    for (int x(0); x < idx; ++x){
        for (int y(0); y < idy; ++y){o[x][y] = o[x][y] + s*v2[x][y];}
    }
}

static void _scale(double** v, double** f, int idx, int idy, double s){
    if (!v || !f){return;}
    for (int x(0); x < idx; ++x){
        for (int y(0); y < idy; ++y){v[x][y] = f[x][y]*s;}
    }
}

void mtx::scale(double** v, int idx, int idy, double s){
    for (int x(0); x < idx; ++x){
        for (int y(0); y < idy; ++y){v[x][y] = this -> _m[x][y]*s;}
    }
}

bool** mtx::_mask(int row, int col){
    bool** mx = (bool**)malloc(row*sizeof(bool*));
    for (int x(0); x < row; ++x){mx[x] = (bool*)malloc(col*sizeof(bool));}
    for (int x(0); x < row; ++x){for (int y(0); y < col; ++y){mx[x][y] = false;}}
    return mx;  
}

void mtx::_copy(double* dst, double* src, int lx){
    for (int x(0); x < lx; ++x){dst[x] = src[x];}
}

void mtx::_copy(double** src, int lx, int ly){
    if (!this -> _m){this -> _m = this -> _matrix(lx, ly);}
    for (int x(0); x < lx; ++x){_copy(this -> _m[x], src[x], ly);}
}

void mtx::_copy(bool* dst, bool* src, int lx){
    for (int x(0); x < lx; ++x){dst[x] = src[x];}
}

void mtx::_copy(bool** src, int lx, int ly){
    if (!this -> _m){this -> _m = this -> _matrix(lx, ly);}
    for (int x(0); x < lx; ++x){_copy(this -> _b[x], src[x], ly);}
}

mtx mtx::Rz(double angle){
    mtx out(3, 3); 
    out._m[0][0] =  std::cos(angle); 
    out._m[0][1] = -std::sin(angle); 
    out._m[1][0] =  std::sin(angle); 
    out._m[1][1] =  std::cos(angle); 
    out._m[2][2] = 1.0;
    return out; 
}

mtx mtx::Ry(double angle){
    mtx out(3, 3); 
    out._m[0][0] = std::cos(angle); 
    out._m[0][2] = std::sin(angle); 
    out._m[1][1] = 1.0; 
    out._m[2][0] = -std::sin(angle); 
    out._m[2][2] =  std::cos(angle);
    return out; 
}

mtx mtx::Rx(double angle){
    mtx out(3, 3); 
    out._m[0][0] = 1.0; 
    out._m[1][1] =  std::cos(angle); 
    out._m[1][2] = -std::sin(angle); 
    out._m[2][1] =  std::sin(angle);
    out._m[2][2] =  std::cos(angle); 
    return out; 
}

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

mtx::mtx(){}
mtx::mtx(int idx, int idy){
    this -> _m = this -> _matrix(idx, idy);
    this -> _b = this -> _mask(idx, idy); 
    this -> dim_i = idx; this -> dim_j = idy; 
}

mtx::mtx(mtx* in){
    this -> _m = in -> _m; 
    this -> _b = in -> _b; 
    this -> dim_i = in -> dim_i;
    this -> dim_j = in -> dim_j; 
    in -> _m = nullptr;     
    in -> _b = nullptr; 
}

mtx::mtx(mtx& in){
    this -> _m = in._m; 
    this -> _b = in._b;
    in._m = nullptr;
    in._b = nullptr; 
    this -> dim_i = in.dim_i;
    this -> dim_j = in.dim_j; 
}

mtx::~mtx(){
    if (this -> _m){
        for (int x(0); x < this -> dim_i; ++x){free(this -> _m[x]);}
        free(this -> _m); 
        this -> _m = nullptr; 
    }
    if (this -> _b){
        for (int x(0); x < this -> dim_i; ++x){free(this -> _b[x]);}
        free(this -> _b); 
        this -> _b = nullptr; 
    }
}

double mtx::trace(){return this -> trace(this -> _m);}

double mtx::det(){
    double o = this -> _m[0][0] * this -> m_00(this-> _m); 
    o -= this -> _m[0][1] * this -> m_01(this-> _m); 
    o += this -> _m[0][2] * this -> m_02(this-> _m); 
    return o;
}

mtx mtx::copy(){
    mtx out(this -> dim_i, this -> dim_j); 
    for (int x(0); x < this -> dim_i; ++x){
        for (int y(0); y < this -> dim_j; ++y){
            out._m[x][y] = this -> _m[x][y]; 
            out._b[x][y] = this -> _b[x][y]; 
        }
    }
    return out; 
}

void mtx::copy(const mtx* ipt, int idx, int idy){
    if (idy < 0){idy = ipt -> dim_j;}
    for (int x(0); x < idy; ++x){
        this -> _m[idx][x] = ipt -> _m[idx][x];
        this -> _b[idx][x] = ipt -> _b[idx][x]; 
    }
}


void mtx::copy(const mtx* ipt, int idx, int jdx, int idy){
    if (idy < 0){idy = ipt -> dim_j;}
    for (int x(0); x < idy; ++x){
        this -> _m[idx][x] = ipt -> _m[jdx][x];
        this -> _b[idx][x] = ipt -> _b[jdx][x]; 
    }
}




void mtx::print(int prec, int width){
    if (!this -> _m){std::cout << "null pointer error" << std::endl; return;}
    std::cout << std::fixed << std::setprecision(prec); 
    for (int x(0); x < this -> dim_i; ++x){
        for (int y(0); y < this -> dim_j; ++y){
            std::cout << std::setw(width) << this -> _m[x][y] << " ";
        }
        std::cout << "\n"; 
    }
    std::cout << std::endl;
}

mtx mtx::cof(){
    mtx out(this -> dim_i, this -> dim_j); 
    out._m[0][0] =  this -> m_00(this -> _m); 
    out._m[1][0] = -this -> m_10(this -> _m); 
    out._m[2][0] =  this -> m_20(this -> _m);
    out._m[0][1] = -this -> m_01(this -> _m); 
    out._m[1][1] =  this -> m_11(this -> _m); 
    out._m[2][1] = -this -> m_21(this -> _m);
    out._m[0][2] =  this -> m_02(this -> _m); 
    out._m[1][2] = -this -> m_12(this -> _m); 
    out._m[2][2] =  this -> m_22(this -> _m);
    return out; 
}

mtx mtx::T(){
    mtx out(this -> dim_j, this -> dim_i); 
    double** vo = out._m; 
    bool**   vb = out._b; 
    for (int x(0); x < this -> dim_j; ++x){
        for (int y(0); y < this -> dim_i; ++y){
            vo[x][y] = this -> _m[y][x];
            vb[x][y] = this -> _b[y][x]; 
        }
    }
    return out; 
}

mtx mtx::dot(const mtx& other){
    mtx out = mtx(this -> dim_i, other.dim_j); 
    for (int x(0); x < this -> dim_i; ++x){
        for (int j(0); j < other.dim_j; ++j){
            double sm = 0; 
            for (int y(0); y < other.dim_i; ++y){sm += this -> _m[x][y] * other._m[y][j];}
            out._m[x][j] = sm; 
        }
    }
    return out;  
}

mtx mtx::dot(const mtx* other){
    return this -> dot(*other);
}

mtx& mtx::operator=(const mtx& o){
    if (this != &o){
        this -> dim_i = o.dim_i; 
        this -> dim_j = o.dim_j; 
        this -> _copy(o._m, o.dim_i, o.dim_j); 
        this -> _copy(o._b, o.dim_i, o.dim_j); 
    }
    return *this; 
}


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


mtx mtx::inv(){
    auto inv3x3 =[this]() -> mtx{
        double det_ = this -> det();
        det_ = (!det_) ? 0.0 : 1.0/det_; 
        return det_*this -> cof().T();
    }; 
    if (this -> dim_i == 3 && this -> dim_j == 3){return inv3x3();}
    return mtx(this -> dim_i, this -> dim_j); 
}

bool mtx::valid(int idx, int idy){
    if (idx >= this -> dim_i || idy >= this -> dim_j){return false;}
    return this -> _b[idx][idy]; 
}


bool mtx::assign(int idx, int idy, double val, bool valid){
    if (this -> dim_i < idx || this -> dim_j < idy){return false;}
    this -> _m[idx][idy] = val;
    this -> _b[idx][idy] = valid; 
    return true;
}

bool mtx::unique(int id1, int id2, double v1, double v2){
    int idx = 0;  
    for (int x(0); x < this -> dim_j; ++x){
        bool sm = fabs(this -> _m[id1][x] - v1) < this -> tol;
        sm *= fabs(this -> _m[id2][x] - v2) < this -> tol; 
        sm *= this -> _b[id1][x]; 
        if (sm){return true;}
        idx += _b[id1][x]; 
        if (this -> _b[id1][x]){continue;}
        break; 
    }
    this -> _m[id1][idx] = v1; this -> _b[id1][idx] = true;
    this -> _m[id2][idx] = v2; this -> _b[id2][idx] = true; 
    return false;
}

mtx* mtx::slice(int idx){
    mtx* ox = new mtx(1, this -> dim_j); 
    this -> _copy(ox -> _m[0], this -> _m[idx], this -> dim_j); 
    return ox; 
}

mtx* mtx::cat(const mtx* v2){
    int lx = this -> dim_i + v2 -> dim_i; 

    int i(0); 
    mtx* vo = new mtx(lx, this -> dim_j); 
    for (int x(0); x < lx; ++x, ++i){
        const mtx* vx = (x < this -> dim_i) ? this : v2; 
        i = (i < this -> dim_i) ? i : 0; 
        vo -> copy(vx, x, i , this -> dim_j); 
    }
    return vo; 
}




mtx mtx::diag(){
    mtx o(this -> dim_i, this -> dim_i); 
    for (int x(0); x < this -> dim_i; ++x){o.assign(x, x, 1);}
    return o; 
}


mtx* mtx::eigenvalues(){
    double a = - this -> trace(); 
    double b =   this -> m_00(this -> _m) + this -> m_11(this -> _m) + this -> m_22(this -> _m); 
    double c = - this -> det(); 
    return solve_cubic(1.0, a, b, c);  
}

mtx mtx::cross(mtx* r1){
    mtx vXc(3, 3); 
    for (int i(0); i < this -> dim_j; ++i){
        vXc.assign(0, i, r1 -> _m[0][1] * this -> _m[2][i] - r1 -> _m[0][2] * this -> _m[1][i]);
        vXc.assign(1, i, r1 -> _m[0][2] * this -> _m[0][i] - r1 -> _m[0][0] * this -> _m[2][i]);
        vXc.assign(2, i, r1 -> _m[0][0] * this -> _m[1][i] - r1 -> _m[0][1] * this -> _m[0][i]);
    }
    return vXc; 
}

mtx mtx::cross(mtx* r1, mtx* r2){
    mtx vx(1, 3); 
    vx.assign(0, 0, r1 -> _m[0][1] * r2 -> _m[0][2] - r1 -> _m[0][2] * r2 -> _m[0][1]);
    vx.assign(0, 1, r1 -> _m[0][2] * r2 -> _m[0][0] - r1 -> _m[0][0] * r2 -> _m[0][2]);
    vx.assign(0, 2, r1 -> _m[0][0] * r2 -> _m[0][1] - r1 -> _m[0][1] * r2 -> _m[0][0]);
    return vx; 
}

mtx* mtx::eigenvector(){
    auto mag =[this](mtx* r) -> double{
        double m = 0; 
        for (int i(0); i < 3; ++i){m += r -> _m[0][i]*r -> _m[0][i];}
        return std::pow(m, 0.5);
    }; 
    double q = this -> m_00(this -> _m) + this -> m_11(this -> _m) + this -> m_22(this -> _m); 
    mtx* eig = solve_cubic(1.0, -this -> trace(), q, -this -> det()); 
    mtx* eiv = new mtx(3, 3); 
    for (int i(0); i < 3; ++i){
        if (fabs(eig -> _m[1][i]) >= eig -> tol){continue;}
        double e = eig -> _m[0][i]; 
        mtx B = this -> copy() - this -> diag()*e; 

        mtx r1(1, 3);
        r1.assign(0, 0, B._m[0][0]); 
        r1.assign(0, 1, B._m[0][1]); 
        r1.assign(0, 2, B._m[0][2]); 

        mtx r2(1, 3);
        r2.assign(0, 0, B._m[1][0]); 
        r2.assign(0, 1, B._m[1][1]); 
        r2.assign(0, 2, B._m[1][2]); 
        mtx cx = this -> cross(&r1, &r2); 
        if (mag(&cx) < eig -> tol){
            mtx r3(1, 3);
            r3.assign(0, 0, B._m[2][0]); 
            r3.assign(0, 1, B._m[2][1]); 
            r3.assign(0, 2, B._m[2][2]); 
            cx = this -> cross(&r1, &r3); 
        }
        double mag_ = mag(&cx);
        if (mag_ > eig -> tol){cx = cx * (1.0/mag_);}
        eiv -> assign(i, 0, cx._m[0][0]); 
        eiv -> assign(i, 1, cx._m[0][1]); 
        eiv -> assign(i, 2, cx._m[0][2]); 
    }
    delete eig; 
    return eiv; 
}







