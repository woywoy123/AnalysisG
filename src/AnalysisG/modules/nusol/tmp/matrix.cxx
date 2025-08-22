#include <reconstruction/solvers.h>
#include <reconstruction/matrix.h>
#include <reconstruction/mtx.h>

#include <iostream>
#include <iomanip>
#include <math.h>


// ------------------- vector 3 dims ------------- //
vec3 vec3::operator-(const vec3& o) const {
    return {this -> x - o.x, this -> y - o.y, this -> z - o.z};
}

vec3 vec3::operator+(const vec3& o) const {
    return {this -> x + o.x, this -> y + o.y, this -> z + o.z};
}

vec3 vec3::operator*(double s) const {
    return {this -> x*s, this -> y*s, this -> z*s};
}

double vec3::mag()  const {
    return std::pow(this -> mag2(), 0.5);
}

double vec3::mag2() const {
    return this -> x*this -> x + this -> y*this -> y + this -> z*this -> z;
}

double vec3::dot(const vec3& o) const {
    return this -> x*o.x + this -> y*o.y + this -> z*o.z;
}

vec3 vec3::cross(const vec3& o) const {
    return {this -> y*o.z - this -> z*o.y, this -> z*o.x - this -> x*o.z, this -> x*o.y - this -> y*o.x}; 
}

void vec3::print() const {
    std::cout << "(" << this -> x << ", " << this -> y << ", " << this -> z << ")" << std::endl;
}


// -------------------- vector 4 dims ------------------- //
vec4 vec4::operator-(const vec4& o) const {
    return {this -> x - o.x, this -> y - o.y, this -> z - o.z, this -> w - o.w};
}

vec4 vec4::operator+(const vec4& o) const {
    return {this -> x + o.x, this -> y + o.y, this -> z + o.z, this -> w + o.w};
}

vec4 vec4::operator*(double s) const {
    return {this -> x*s, this -> y*s, this -> z*s, this -> w*s};
}

double vec4::dot(const vec4& o) const {
    return this -> x*o.x + this -> y*o.y + this -> z*o.z + this -> w*o.w;
}

void vec4::print() {
    std::cout << "(" << this -> x << ", " << this -> y << ", " << this -> z << ", " << this -> w << ")\n";
}


// ------------------- matrix ----------------------- //
matrix::matrix(int r, int c){
    this -> _c = c; this -> _r = r; 
    this -> data = new double*[r];
    for (int i(0); i < r; ++i){this -> data[i] = new double[c]();}
}

matrix::matrix(const matrix& other){
    this -> _r = other._r; this -> _c = other._c; 
    this -> data = new double*[this -> _r];
    for (int i(0); i < this -> _r; ++i){
        this -> data[i] = new double[this -> _c];
        for (int j(0); j < this -> _c; ++j){this -> data[i][j] = other.data[i][j];}
    }
}

matrix::~matrix(){
    for (int i(0); i < this -> _r; ++i){ delete [] this -> data[i]; }
    delete [] this -> data; 
}


int matrix::rows() const { 
    return this -> _r; 
}

int matrix::cols() const { 
    return this -> _c; 
}

double& matrix::at(int r, int c) {
    return this -> data[r][c];
}

const double& matrix::at(int r, int c) const {
    return this -> data[r][c];
}

void matrix::print(int p){
    std::cout << "--------------" << std::endl; 
    for (int i(0); i < this -> _r; ++i) {
        for (int j(0); j < this -> _c; ++j){
            std::cout << std::fixed << std::setprecision(p) << this -> data[i][j] << " \t";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------" << std::endl; 
}

double matrix::mag() const {
    if (this -> _c != 1) { return 0; }
    double sum_sq = 0;
    for (int i = 0; i < this -> _r; ++i){
        sum_sq += this -> data[i][0] * this -> data[i][0];
    }
    return std::pow(sum_sq, 0.5);
}

matrix matrix::I(int size){
    matrix mx(size, size);
    for (int k(0); k < size; ++k){ mx.data[k][k] = 1.0; }
    return mx; 
}

matrix matrix::T() const{
    matrix mx(this -> _c, this -> _r);
    for (int i(0); i < this -> _r; ++i) {
        for (int j(0); j < this -> _c; ++j){ mx.data[j][i] = this -> data[i][j]; }
    }
    return mx;
}


double matrix::det(){
    double c00 = this -> data[0][0] * (this -> data[1][1] * this -> data[2][2] - this -> data[1][2] * this -> data[2][1]); 
    double c11 = this -> data[0][1] * (this -> data[1][0] * this -> data[2][2] - this -> data[1][2] * this -> data[2][0]); 
    double c22 = this -> data[0][2] * (this -> data[1][0] * this -> data[2][1] - this -> data[1][1] * this -> data[2][0]); 
    return c00 - c11 + c22; 
}

matrix matrix::inverse(){
    double det_ = this -> det(); 
    if (std::abs(det_) < 1e-12){ return matrix::I(3); }
    double inv_det = 1.0 / det_;

    matrix mx(3, 3);
    mx.data[0][0] = (this -> data[1][1]*this -> data[2][2] - this -> data[1][2]*this -> data[2][1]) * inv_det;
    mx.data[0][1] = (this -> data[0][2]*this -> data[2][1] - this -> data[0][1]*this -> data[2][2]) * inv_det;
    mx.data[0][2] = (this -> data[0][1]*this -> data[1][2] - this -> data[0][2]*this -> data[1][1]) * inv_det;
    mx.data[1][0] = (this -> data[1][2]*this -> data[2][0] - this -> data[1][0]*this -> data[2][2]) * inv_det;
    mx.data[1][1] = (this -> data[0][0]*this -> data[2][2] - this -> data[0][2]*this -> data[2][0]) * inv_det;
    mx.data[1][2] = (this -> data[0][2]*this -> data[1][0] - this -> data[0][0]*this -> data[1][2]) * inv_det;
    mx.data[2][0] = (this -> data[1][0]*this -> data[2][1] - this -> data[1][1]*this -> data[2][0]) * inv_det;
    mx.data[2][1] = (this -> data[0][1]*this -> data[2][0] - this -> data[0][0]*this -> data[2][1]) * inv_det;
    mx.data[2][2] = (this -> data[0][0]*this -> data[1][1] - this -> data[0][1]*this -> data[1][0]) * inv_det;
    return mx; 
}

matrix matrix::dot(const matrix& o){
    matrix mx(this -> _r, o._c); 
    for (int i(0); i < this -> _r; ++i) {
        for (int j(0); j < o._c; ++j){
            for (int k(0); k < this -> _c; ++k){ mx.data[i][j] += this -> data[i][k] * o.data[k][j]; }
        }
    }
    return mx; 
}

void matrix::eigenvalues(vec3* real, vec3* imag){
    if (this -> _c != 3 || this -> _r != 3){return;}
    double a00 = this -> data[0][0], a01 = this -> data[0][1], a02 = this -> data[0][2]; 
    double a10 = this -> data[1][0], a11 = this -> data[1][1], a12 = this -> data[1][2]; 
    double a20 = this -> data[2][0], a21 = this -> data[2][1], a22 = this -> data[2][2]; 

    double a = 1; 
    double b = -(a00 + a11 + a22); 
    double c =  (a00 * a11 - a01 * a10) + (a00 * a22 - a02 * a20) + (a11 * a22 - a12 * a21); 
    double d = -this -> det(); 
    mtx* solx = solve_cubic(a, b, c, d);

    real -> x = solx -> _m[0][0]; 
    real -> y = solx -> _m[0][1]; 
    real -> z = solx -> _m[0][2]; 
    
    imag -> x = solx -> _m[1][0]; 
    imag -> y = solx -> _m[1][1]; 
    imag -> z = solx -> _m[1][2]; 
    delete solx; 
}

matrix& matrix::operator=(const matrix& o){
    if (this == &o){ return *this; }
    for (int i(0); i < this -> _r; ++i){ delete [] this -> data[i]; }
    delete [] this -> data;
    
    this -> _r = o._r; this -> _c = o._c;
    this -> data = new double*[this -> _r];
    for (int i(0); i < this -> _r; ++i){
        this -> data[i] = new double[this -> _c];
        for (int j(0); j < this -> _c; ++j){ this -> data[i][j] = o.data[i][j]; }
    }
    return *this; 
}

matrix matrix::operator+(const matrix& o) const {
    matrix mx(this -> _r, this -> _c);
    for (int i(0); i < this -> _r; ++i) {
        for (int j(0); j < this -> _c; ++j){ mx.data[i][j] = this -> data[i][j] + o.data[i][j]; }
    }
    return mx;
}

matrix matrix::operator-(const matrix& o) const {
    matrix mx(this -> _r, this -> _c);
    for (int i(0); i < this -> _r; ++i){
        for (int j(0); j < this -> _c; ++j){ mx.data[i][j] = this -> data[i][j] - o.data[i][j]; }
    }
    return mx;
}

matrix matrix::operator*(const matrix& o) const {
    matrix mx(this -> _r, o._c); 
    for (int i(0); i < this -> _r; ++i) {
        for (int j(0); j < o._c; ++j){
            for (int k(0); k < this -> _c; ++k){ mx.data[i][j] += this -> data[i][k] * o.data[k][j]; }
        }
    }
    return mx; 
}

matrix matrix::operator*(double scalar) const {
    matrix mx(this -> _r, this -> _c);
    for (int i(0); i < this -> _r; ++i){
        for (int j(0); j < this -> _c; ++j){ mx.data[i][j] = this -> data[i][j] * scalar; }
    }
    return mx;
}

vec4 matrix::operator*(const vec4& v) const {
    return {
        this -> data[0][0]*v.x + this -> data[0][1]*v.y + this -> data[0][2]*v.z + this -> data[0][3]*v.w,
        this -> data[1][0]*v.x + this -> data[1][1]*v.y + this -> data[1][2]*v.z + this -> data[1][3]*v.w,
        this -> data[2][0]*v.x + this -> data[2][1]*v.y + this -> data[2][2]*v.z + this -> data[2][3]*v.w,
        this -> data[3][0]*v.x + this -> data[3][1]*v.y + this -> data[3][2]*v.z + this -> data[3][3]*v.w
    };
}

vec3 matrix::operator*(const vec3& v) const {
    return {
        this -> data[0][0]*v.x + this -> data[0][1]*v.y + this -> data[0][2]*v.z,
        this -> data[1][0]*v.x + this -> data[1][1]*v.y + this -> data[1][2]*v.z,
        this -> data[2][0]*v.x + this -> data[2][1]*v.y + this -> data[2][2]*v.z
    };
}


// ------------------- auxiliary functions ----------------- //

bool solve_linear(const matrix& A_in, const matrix& b_in, matrix& x_out){
    const int size = A_in.rows();
    matrix aug(size, size + 1);

    for (int i(0); i < size; ++i){
        for (int j(0); j < size; ++j){aug.at(i, j) = A_in.at(i, j);}
        aug.at(i, size) = b_in.at(i, 0);
    }

    for (int k(0); k < size; ++k){
        int pr = k;
        double max_val = std::abs(aug.at(k, k));
        for (int row(k + 1); row < size; ++row){
            double vx = std::abs(aug.at(row, k)); 
            if (vx <= max_val){ continue; }
            max_val = vx; pr = row;
        }

        if (max_val < 1e-15){ return false; }
        for (int col(0); col < (size + 1)*(pr != k); ++col){
            std::swap(aug.at(k, col), aug.at(pr, col));
        }

        for (int col(k); col <= size; ++col){
            aug.at(k, col) /= aug.at(k, k);
        }

        for (int row(0); row < size; ++row){
            if (row == k){ continue; }
            double f = aug.at(row, k);
            for (int col(k); col <= size; ++col){
                aug.at(row, col) -= f * aug.at(k, col);
            }
        }
        aug.print(); 
    }
    x_out = matrix(size, 1);
    for (int i(0); i < size; ++i){x_out.at(i, 0) = aug.at(i, size);}
    return true;
}

bool invert_matrix(const matrix& M, matrix& M_inv){
    const int n = M.rows();
    M_inv = matrix(n, n);
    matrix ic(n, 1), sc(n, 1);
    for (int j(0); j < n; ++j){
        ic.at(j, 0) = 1.0;
        if (!solve_linear(M, ic, sc)){
            matrix M_regularized = M;
            for (int i(0); i < n; ++i){ M_regularized.at(i, i) += 1e-8; }
            if (!solve_linear(M_regularized, ic, sc)){return false;}
        }
        for (int i(0); i < n; ++i){M_inv.at(i, j) = sc.at(i, 0);}
        ic.at(j, 0) = 0.0;
    }
    return true;
}

void print(double v, int p){
    std::cout << std::fixed << std::setprecision(p) << v << "\n";
}

