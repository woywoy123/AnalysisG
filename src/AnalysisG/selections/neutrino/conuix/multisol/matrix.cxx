#include "multisol/memory.h"
#include "multisol/matrix.h"
#include <iostream>
#include <iomanip>
#include <vector>

matrix_t::matrix_t(int _r, int _c){
    this -> r = _r; 
    this -> c = _c; 
    this -> data = alloc(this -> r, this -> c); 
}

matrix_t::matrix_t(const matrix_t& other){
    this -> r = other.r; 
    this -> c = other.c; 
    this -> data = alloc(this -> r, this -> c); 
    copy(other.data, this -> data, this -> r, this -> c); 
}


matrix_t::~matrix_t(){
    this -> data = flush(this -> data, this -> r); 
}

matrix_t matrix_t::cat(const matrix_t& o){
    matrix_t m(this -> r + o.r, this -> c); 
    int s = 0; 
    for (int x(0); x < this -> r; ++x, ++s){copy_ij(this -> data, m.data, s, x, this -> c);} 
    for (int x(0); x <       o.r; ++x, ++s){copy_ij(      o.data, m.data, s, x,       o.c);} 
    return m; 
}

matrix_t matrix_t::dot(const matrix_t& o){
    matrix_t mx(this -> r, o.c); 
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < o.c; ++j){
            for (int k(0); k < this -> c; ++k){
                mx.data[i][j] += this -> data[i][k] * o.data[k][j];
            }
        }
    }
    return mx; 
}

matrix_t matrix_t::at(int _r){
    matrix_t o(1, this -> c); 
    copy_ij(this -> data, o.data, 0, _r, this -> c); 
    return o; 
}



long double& matrix_t::at(int _r, int _c){
    if (this -> r < _r || this -> c < _c){
        std::cout << "Invalid memory access: \n"; 
        std::cout << "r_matrix " << _r << " -> " << "r_in" << _r << "\n"; 
        std::cout << "c_matrix " << _r << " -> " << "c_in" << _r << "\n"; 
        std::cout << std::endl;
        abort(); 
    }
    return this -> data[_r][_c]; 
}

const long double& matrix_t::at(int _r, int _c) const {
    if (this -> r < _r || this -> c < _c){
        std::cout << "Invalid memory access: \n"; 
        std::cout << "r_matrix " << _r << " -> " << "r_in" << _r << "\n"; 
        std::cout << "c_matrix " << _r << " -> " << "c_in" << _r << "\n"; 
        std::cout << std::endl;
        abort(); 
    }
    return this -> data[_r][_c]; 
}

long double matrix_t::det(){
    if (r == 1){return this -> data[0][0];}
    if (r == 2){return det2(this -> data);} 
    if (r == 3){return det3(this -> data);} 
    int s = 1;
    long double _dt = 0;
    for (int i(0); i < this -> c; ++i) {
        _dt += s * this -> data[0][i] * this -> minor(0, i).det();
        s = -s;
    }
    return _dt;
}

matrix_t matrix_t::minor(int row, int col){
    int mi = 0, mj = 0;
    matrix_t minor(this -> r-1, this -> c-1);
    for (int i(0); i < this -> r; ++i) {
        if (i == row){continue;}
        mj = 0;
        for (int j(0); j < this -> c; ++j) {
            if (j == col) continue;
            minor.at(mi, mj) = this -> data[i][j];
            ++mj;
        }
        ++mi;
    }
    return minor;
}

matrix_t matrix_t::minor(){
    matrix_t minr(this -> r, this -> c);
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < this -> c; ++j){minr.at(i, j) = this -> minor(i, j).det();}
    }
    return minr;
}

matrix_t matrix_t::cofactor(){
    matrix_t coef(this -> r, this -> c);
    matrix_t minr = this -> minor();
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < this -> c; ++j) {
            int sign = ((i + j) % 2 == 0) ? 1 : -1;
            coef.at(i, j) = sign * minr.at(i, j);
        }
    }
    return coef;
}

matrix_t matrix_t::adj(){
    return this -> cofactor().T();
}

matrix_t matrix_t::inv(){
    long double _det = this -> det();
    matrix_t adj = this -> adj();
    matrix_t inverse(this -> r, this -> c);
    opm(inverse.data, adj.data, this -> r, this -> c, 1.0L / _det); 
    return inverse;
}

long double matrix_t::trace(){
    long double sm = 0; 
    for (int x(0); x < this -> r; ++x){sm += this -> data[x][x];}
    return sm; 
}

matrix_t matrix_t::eigenvector(long double l, long double epsilon){
    matrix_t a_li(this -> r, this -> c);
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < this -> c; ++j){a_li.data[i][j] = this -> data[i][j] - (i == j) * l;}
    }
    return a_li.nullspace(epsilon);
}

roots_t matrix_t::eigenvalues(){
    if (this -> r == 1){return {this -> data[0][0]};}
    if (this -> r == 2){return find_roots(1.0l, - this -> trace(), this -> det(), 1e-12);} 
    if (this -> r != 3){return roots_t();}
    double a = - this -> trace(); 
    double b =   _m_00(this -> data) + _m_11(this -> data) + _m_22(this -> data); 
    double c = - _det(this -> data); 
    return find_roots(1.0L, a, b, c, 1e-12);
}


matrix_t matrix_t::cross(const matrix_t* r1){
    matrix_t vXc(3, 3); 
    for (int i(0); i < this -> c; ++i){
        vXc.data[0][i] = r1 -> data[0][1] * this -> data[2][i] - r1 -> data[0][2] * this -> data[1][i];
        vXc.data[1][i] = r1 -> data[0][2] * this -> data[0][i] - r1 -> data[0][0] * this -> data[2][i];
        vXc.data[2][i] = r1 -> data[0][0] * this -> data[1][i] - r1 -> data[0][1] * this -> data[0][i];
    }
    return vXc; 
}

matrix_t matrix_t::cross(matrix_t* r1, matrix_t* r2){
    matrix_t vx(1, 3); 
    vx.data[0][0] = r1 -> data[0][1] * r2 -> data[0][2] - r1 -> data[0][2] * r2 -> data[0][1];
    vx.data[0][1] = r1 -> data[0][2] * r2 -> data[0][0] - r1 -> data[0][0] * r2 -> data[0][2];
    vx.data[0][2] = r1 -> data[0][0] * r2 -> data[0][1] - r1 -> data[0][1] * r2 -> data[0][0];
    return vx; 
}

matrix_t matrix_t::eigenvector(){
    double a = - this -> trace(); 
    double b =   _m_00(this -> data) + _m_11(this -> data) + _m_22(this -> data); 
    double c = - _det(this -> data); 
    matrix_t egv = matrix_t(3, 3); 
    matrix_t eig = find_roots(1.0L, a, b, c, 1e-12).vec();
    for (int x(0); x < eig.r; ++x){
        long double ls = eig.at(x, 0); 
        matrix_t exv = this -> eigenvector(ls); 
        copy_ij(exv.data, egv.data, x, 0, 3);
    }
    return eig;
}


matrix_t matrix_t::nullspace(long double epsilon){
    matrix_t A = this -> clone(); 
    int m = this -> r;
    int n = this -> c;
    
    int rk = 0;
    std::vector<int> rp(m);
    std::vector<int> cp(n);
    for (int i(0); i < m; ++i){rp[i] = i;}
    for (int j(0); j < n; ++j){cp[j] = j;}
    
    for (int k = 0; k < std::min(m, n); ++k) {
        int max_i = k, max_j = k;
        long double max_v = 0.0L;
        
        for (int i = k; i < m; ++i) {
            for (int j = k; j < n; ++j) {
                long double abs_v = std::fabsl(A.at(rp[i], cp[j]));
                if (abs_v <= max_v){continue;}
                max_v = abs_v;
                max_i = i; max_j = j;
            }
        }
        if (max_v < epsilon) break;
        std::swap(rp[k], rp[max_i]);
        std::swap(cp[k], cp[max_j]);
        long double pv = 1.0L / A.at(rp[k], cp[k]);
        for (int j(k); j < n; ++j){A.at(rp[k], cp[j]) = pv;}
        for (int i(k + 1); i < m; ++i) {
            long double f = A.at(rp[i], cp[k]);
            for (int j(k); j < n; ++j){A.at(rp[i], cp[j]) -= f * A.at(rp[k], cp[j]);}
        }  
    }

    int nl = n - rk;
    matrix_t nb(n, nl);
    for (int j(rk); j < n; ++j) {
        for (int i(0); i < n; ++i){nb.at(i, j - rk) = (i == j) * 1.0L;}
        for (int i(rk - 1); i >= 0; --i) {
            long double sum = 0.0L;
            for (int k(i + 1); k < n; ++k){sum += A.at(rp[i], cp[k]) * nb.at(k, j - rk);}
            nb.at(i, j - rk) = -sum / A.at(rp[i], cp[i]);
        }
    }
    
    matrix_t result(n, nl);
    for (int i(0); i < n; ++i) {
        for (int j(0); j < nl; ++j){result.at(cp[i], j) = nb.at(i, j);}
    }
    return result;
}


matrix_t matrix_t::T() const {
    matrix_t mx(this -> c, this -> r); 
    opt(mx.data, this -> data, this -> r, this -> c); 
    return mx;
}

matrix_t matrix_t::clone(){
    matrix_t mx(this -> r, this -> c); 
    copy(this -> data, mx.data, this -> r, this -> c); 
    return mx;
}

matrix_t matrix_t::clone() const {
    matrix_t mx(this -> r, this -> c); 
    copy(this -> data, mx.data, this -> r, this -> c); 
    return mx;
}



matrix_t& matrix_t::operator=(const matrix_t& o){
    if (this == &o){return *this;}
    flush(this -> data, this -> r);
    this -> r = o.r; this -> c = o.c; 
    this -> data = alloc(this -> r, this -> c); 
    copy(o.data, this -> data, this -> r, this -> c); 
    return *this; 
}

matrix_t matrix_t::operator+(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, 1); 
    return mx;
}

matrix_t matrix_t::operator+(const matrix_t& o){
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, 1); 
    return mx;
}

matrix_t matrix_t::operator-(const matrix_t& o) const {
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, -1); 
    return mx;
}

matrix_t matrix_t::operator-(const matrix_t& o){
    matrix_t mx(this -> r, this -> c); 
    ops(mx.data, this -> data, o.data, this -> r, this -> c, -1); 
    return mx;
}


matrix_t matrix_t::operator*(long double s) const {
    matrix_t mx(this -> r, this -> c); 
    opm(mx.data, this -> data, this -> r, this -> c, s); 
    return mx;
}

matrix_t matrix_t::operator*(long double s){
    matrix_t mx(this -> r, this -> c); 
    opm(mx.data, this -> data, this -> r, this -> c, s); 
    return mx;
}

void matrix_t::print(int p){
    std::cout << "------------------------" << std::endl;
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < this -> c; ++j){
            std::cout << std::fixed << std::setprecision(p) << this -> data[i][j] << "\t"; 
        }
        std::cout << std::endl; 
    }
    std::cout << "------------------------" << std::endl;
}

void matrix_t::print(int p) const {
    std::cout << "------------------------" << std::endl;
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < this -> c; ++j){
            std::cout << std::fixed << std::setprecision(p) << this -> data[i][j] << "\t"; 
        }
        std::cout << std::endl; 
    }
    std::cout << "------------------------" << std::endl;
}








