#include "structs.h"
#include "particle.h"
#include <iostream>
#include <iomanip>

void print(double v, int p){
    std::cout << std::fixed << std::setprecision(p) << v << "\n";
}

double cos_theta(const particle* b, const particle* mu){
    double _d = b -> px * mu -> px;
    _d += b -> py * mu -> py; 
    _d += b -> pz * mu -> pz; 
    return _d / (b -> p * mu -> p);
}

vec3 vec3::operator-(const vec3& o) const {return {this -> x - o.x, this -> y - o.y, this -> z - o.z};}
vec4 vec4::operator-(const vec4& o) const {return {this -> x - o.x, this -> y - o.y, this -> z - o.z, this -> w - o.w};}

matrix matrix::operator-(const matrix& o) const {
    matrix mx(this -> r, this -> c);
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < this -> c; ++j){ mx.data[i][j] = this -> data[i][j] - o.data[i][j]; }
    }
    return mx;
}

vec3 vec3::operator+(const vec3& o) const {return {this -> x + o.x, this -> y + o.y, this -> z + o.z};}
vec4 vec4::operator+(const vec4& o) const {return {this -> x + o.x, this -> y + o.y, this -> z + o.z, w + o.w};}

matrix matrix::operator+(const matrix& o) const {
    matrix mx(this -> r, this -> c);
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < this -> c; ++j){ mx.data[i][j] = this -> data[i][j] + o.data[i][j]; }
    }
    return mx;
}

vec3 vec3::operator*(double s) const {return {this -> x*s, this -> y*s, this -> z*s};}
vec4 vec4::operator*(double s) const {return {x*s, y*s, z*s, w*s};}

vec4 matrix::operator*(const vec4& v) const {
    return {
        this -> data[0][0]*v.x + this -> data[0][1]*v.y + this -> data[0][2]*v.z + this -> data[0][3]*v.w,
        this -> data[1][0]*v.x + this -> data[1][1]*v.y + this -> data[1][2]*v.z + this -> data[1][3]*v.w,
        this -> data[2][0]*v.x + this -> data[2][1]*v.y + this -> data[2][2]*v.z + this -> data[2][3]*v.w,
        this -> data[3][0]*v.x + this -> data[3][1]*v.y + this -> data[3][2]*v.z + this -> data[3][3]*v.w
    };
}

matrix matrix::operator*(const matrix& o) const {
    matrix mx(this -> r, o.c); 
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < o.c; ++j){
            for (int k(0); k < this -> c; ++k){ mx.data[i][j] += this -> data[i][k] * o.data[k][j]; }
        }
    }
    return mx; 
}

vec3 matrix::operator*(const vec3& v) const {
    return {
        this -> data[0][0]*v.x + this -> data[0][1]*v.y + this -> data[0][2]*v.z,
        this -> data[1][0]*v.x + this -> data[1][1]*v.y + this -> data[1][2]*v.z,
        this -> data[2][0]*v.x + this -> data[2][1]*v.y + this -> data[2][2]*v.z
    };
}

matrix matrix::operator*(double scalar) const {
    matrix mx(this -> r, this -> c);
    for (int i(0); i < this -> r; ++i){
        for (int j(0); j < this -> c; ++j){ mx.data[i][j] = this -> data[i][j] * scalar; }
    }
    return mx;
}


matrix& matrix::operator=(const matrix& o){
    if (this == &o){ return *this; }
    for (int i(0); i < this -> r; ++i){ delete [] this -> data[i]; }
    delete [] this -> data;
    
    r = o.r; c = o.c;
    this -> data = new double*[r];
    for (int i(0); i < this -> r; ++i){
        this -> data[i] = new double[c];
        for (int j(0); j < c; ++j){ this -> data[i][j] = o.data[i][j]; }
    }
    return *this; 
}


double vec3::mag()  const {return std::sqrt(this -> mag2());}
double vec3::mag2() const {return this -> x*this -> x + this -> y*this -> y + this -> z*this -> z;}

double matrix::mag() const {
    if (this -> c != 1) { return 0; } // Only for column vectors
    double sum_sq = 0;
    for (int i = 0; i < this->r; ++i){sum_sq += this->data[i][0] * this->data[i][0];}
    return std::sqrt(sum_sq);
}

double vec4::dot(const vec4& o) const {return x*o.x + y*o.y + z*o.z + w*o.w;}
double vec3::dot(const vec3& o) const {return this -> x*o.x + this -> y*o.y + this -> z*o.z;}

vec3 vec3::cross(const vec3& o) const {
    return {this -> y*o.z - this -> z*o.y, this -> z*o.x - this -> x*o.z, this -> x*o.y - this -> y*o.x}; 
}

void vec4::print() {std::cout << "(" << x << ", " << y << ", " << z << ", " << w << ")\n";}
void vec3::print() const {std::cout << "(" << this -> x << ", " << this -> y << ", " << this -> z << ")" << std::endl;}

void matrix::print(int p){
    std::cout << "--------------" << std::endl; 
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < this -> c; ++j){
            std::cout << std::fixed << std::setprecision(p) << this -> data[i][j] << " \t";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------" << std::endl; 
}

double&       matrix::at(int r, int c) {return this -> data[r][c];}
const double& matrix::at(int r, int c) const {return this -> data[r][c];}

matrix::matrix(int r, int c){
    this -> c = c; this -> r = r; 
    this -> data = new double*[r];
    for (int i(0); i < r; ++i){this -> data[i] = new double[c]();}
}

matrix::matrix(const matrix& other){
    this -> r = other.r; this -> c = other.c; 
    this -> data = new double*[r];
    for (int i(0); i < this -> r; ++i){
        this -> data[i] = new double[c];
        for (int j(0); j < this -> c; ++j){this -> data[i][j] = other.data[i][j];}
    }
}

matrix::~matrix(){
    for (int i(0); i < this -> r; ++i){ delete [] this -> data[i]; }
    delete [] this -> data; 
}


matrix matrix::I(int size){
    matrix mx(size, size);
    for (int k(0); k < size; ++k){ mx.data[k][k] = 1.0; }
    return mx; 
}

matrix matrix::T() const{
    matrix mx(this -> c, this -> r);
    for (int i(0); i < this -> r; ++i) {
        for (int j(0); j < this -> c; ++j){ mx.data[j][i] = this -> data[i][j]; }
    }
    return mx;
}

matrix matrix::inverse(){
    double det = this -> data[0][0] * (this -> data[1][1] * this -> data[2][2] - this -> data[1][2] * this -> data[2][1]) 
               - this -> data[0][1] * (this -> data[1][0] * this -> data[2][2] - this -> data[1][2] * this -> data[2][0]) 
               + this -> data[0][2] * (this -> data[1][0] * this -> data[2][1] - this -> data[1][1] * this -> data[2][0]);

    if (std::abs(det) < 1e-12){ return matrix::I(3); }
    double inv_det = 1.0 / det;

    matrix mx(3, 3);
    mx.data[0][0] = (this -> data[1][1]*this -> data[2][2] - this -> data[1][2]*this -> data[2][1])*inv_det;
    mx.data[0][1] = (this -> data[0][2]*this -> data[2][1] - this -> data[0][1]*this -> data[2][2])*inv_det;
    mx.data[0][2] = (this -> data[0][1]*this -> data[1][2] - this -> data[0][2]*this -> data[1][1])*inv_det;
    mx.data[1][0] = (this -> data[1][2]*this -> data[2][0] - this -> data[1][0]*this -> data[2][2])*inv_det;
    mx.data[1][1] = (this -> data[0][0]*this -> data[2][2] - this -> data[0][2]*this -> data[2][0])*inv_det;
    mx.data[1][2] = (this -> data[0][2]*this -> data[1][0] - this -> data[0][0]*this -> data[1][2])*inv_det;
    mx.data[2][0] = (this -> data[1][0]*this -> data[2][1] - this -> data[1][1]*this -> data[2][0])*inv_det;
    mx.data[2][1] = (this -> data[0][1]*this -> data[2][0] - this -> data[0][0]*this -> data[2][1])*inv_det;
    mx.data[2][2] = (this -> data[0][0]*this -> data[1][1] - this -> data[0][1]*this -> data[1][0])*inv_det;
    return mx; 
}


