#ifndef MULTISOL_MATRIX_H
#define MULTISOL_MATRIX_H

template <typename T>
void safe(T** o){
    if (!*o){return;}
    delete *o; *o = nullptr;
}; 

struct vec3 {
    double x = 0, y = 0, z = 0;
    vec3 operator+(const vec3& o) const; 
    vec3 operator-(const vec3& o) const;
    vec3 operator*(double s) const;
    vec3 cross(const vec3& o) const;
    double dot(const vec3& o) const;
    double mag2() const;
    double mag() const; 
    void print() const; 
};

struct vec4 {
    double x = 0, y = 0, z = 0, w = 0;
    vec4 operator+(const vec4& o) const;
    vec4 operator-(const vec4& o) const;
    vec4 operator*(double s) const;
    double dot(const vec4& o) const;
    void print();
};


struct matrix {
    matrix(int r=3, int c=3);
    matrix(const matrix& other);
    ~matrix();

    void print(int p = 5);

    // fetch and assignment methods
    double& at(int r, int c); 
    const double& at(int r, int c) const; 

    int rows() const; 
    int cols() const; 

    double mag() const; 

    // identity matrix
    static matrix I(int size);
    
    // transpose
    matrix T() const;

    // inverse 3x3
    matrix inverse();
    double det(); 

    // dot product
    matrix dot(const matrix& m); 
    void eigenvalues(vec3* real, vec3* imag); 

    matrix& operator=(const matrix& other);
    matrix  operator+(const matrix& o) const;
    matrix  operator-(const matrix& o) const;
    matrix  operator*(const matrix& o) const;
    matrix  operator*(double s       ) const;
    vec4    operator*(const vec4& v  ) const; 
    vec3    operator*(const vec3& v  ) const;

    double** data = nullptr;
    int _r = 0, _c = 0;
};

void print(double v, int p = 12); 
bool solve_linear(const matrix& A_in, const matrix& b_in, matrix& x_out); 
bool invert_matrix(const matrix& M, matrix& M_inv); 

#endif

